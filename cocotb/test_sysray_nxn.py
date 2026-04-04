import random
import cocotb
from cocotb.triggers import FallingEdge, ReadOnly
from pathlib import Path
import pytest
from shared import clock_start, reset_sequence
from runner import run_test


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------

def vec_mat_mul_ref(acts, weights):
    """C[j] = sum_i acts[i] * weights[i][j]  (vector @ matrix, column outputs)"""
    N = len(acts)
    return [sum(acts[i] * weights[i][j] for i in range(N)) for j in range(N)]


def mat_mat_mul_ref(act_matrix, weights):
    """Compute act_matrix @ weights row-by-row; returns an N×N output matrix."""
    return [vec_mat_mul_ref(act_row, weights) for act_row in act_matrix]


def tiled_matmul_ref(act_banks, weight_banks):
    """
    Reference for one N×N output tile, accumulating over K inner-dimension tiles.

    Computes: C = sum_k act_banks[k] @ weight_banks[k]

    Each act_banks[k] and weight_banks[k] is an N×N tile representing one slice
    along the shared inner dimension of a larger multiply. Their partial products
    are summed into a single N×N accumulated output tile.
    """
    N = len(act_banks[0])
    K = len(act_banks)
    result = [[0] * N for _ in range(N)]
    for k in range(K):
        partial = mat_mat_mul_ref(act_banks[k], weight_banks[k])
        for m in range(N):
            for j in range(N):
                result[m][j] += partial[m][j]
    return result


def full_matmul_ref(A, W):
    """
    Reference for a full (T*N) × (T*N) matrix multiply, computed directly.

    A and W are 2D lists of shape (T*N) × (T*N).
    Returns C = A @ W as a 2D list of the same shape.
    """
    rows = len(A)
    cols = len(W[0])
    inner = len(W)
    return [
        [sum(A[r][k] * W[k][c] for k in range(inner)) for c in range(cols)]
        for r in range(rows)
    ]


# ---------------------------------------------------------------------------
# Hardware drivers
# ---------------------------------------------------------------------------

async def load_weights(dut, N, weights, sel=0):
    """
    Load a single weight matrix into the systolic array with diagonal column stagger.
    """
    for cycle in range(2 * N - 1):
        await FallingEdge(dut.clk_i)
        for col in range(N):
            row_idx = cycle - col
            if 0 <= row_idx < N:
                row = N - 1 - row_idx
                dut.weight_sel_n_i[col].value   = sel
                dut.weight_n_i[col].value       = weights[row][col]
                dut.weight_valid_n_i[col].value = 1
            else:
                dut.weight_n_i[col].value       = 0
                dut.weight_valid_n_i[col].value = 0

    await FallingEdge(dut.clk_i)
    for col in range(N):
        dut.weight_n_i[col].value       = 0
        dut.weight_valid_n_i[col].value = 0


async def stream_activation_matrix(dut, N, act_matrix, sel=0):
    """
    Stream a single N×N activation matrix through the systolic array.

    Returns an N×N list of output rows.
    """
    results = [[None] * N for _ in range(N)]

    for cycle in range(N + 2 * N - 1):
        await FallingEdge(dut.clk_i)

        for i in range(N):
            m = cycle - i
            dut.act_sel_n_i[i].value = sel
            if 0 <= m < N:
                dut.act_n_i[i].value       = act_matrix[m][i]
                dut.act_valid_n_i[i].value = 1
            else:
                dut.act_n_i[i].value       = 0
                dut.act_valid_n_i[i].value = 0

        await ReadOnly()

        for j in range(N):
            if dut.psum_out_valid_n_o[j].value == 1:
                m = cycle - N - j
                if 0 <= m < N:
                    results[m][j] = dut.psum_out_n_o[j].value.to_signed()

    for r in range(N):
        for j in range(N):
            assert results[r][j] is not None, f"row {r}, col {j}: output not captured"
    for m, row_out in enumerate(results):
        cocotb.log.info(f"  → output row {m}: {row_out}")
    return results


async def load_weight_banks(dut, N, weight_banks):
    """
    Load K weight banks back-to-back with diagonal column stagger and alternating sel.

    Each bank k occupies a window of N cycles staggered per column, overlapping
    with the next bank so that the array is always being fed without gaps.
    """
    K = len(weight_banks)

    for cycle in range((K + 1) * N - 1):
        await FallingEdge(dut.clk_i)
        for col in range(N):
            t = cycle - col
            k = t // N      # bank index
            bk_idx = t % N  # row index within bank k (bottom-to-top sweep)

            if 0 <= t < K * N:
                dut.weight_sel_n_i[col].value   = k % 2
                dut.weight_n_i[col].value       = weight_banks[k][N - 1 - bk_idx][col]
                dut.weight_valid_n_i[col].value = 1
            else:
                dut.weight_n_i[col].value       = 0
                dut.weight_valid_n_i[col].value = 0

    await FallingEdge(dut.clk_i)
    dut.weight_valid_n_i[N - 1].value = 0


async def stream_activation_banks_tiled(dut, N, act_banks):
    """
    Stream K activation banks back-to-back with no inter-bank gap, causing the
    DUT to automatically accumulate all K partial products internally.

    Because the hardware accumulates across back-to-back inputs, only one N×N
    result is produced at the output — the fully-accumulated tile C. This result
    appears N cycles after the last activation row has been consumed, staggered
    by column in the usual diagonal fashion.

    Returns a single N×N accumulated result matrix.
    """
    K = len(act_banks)
    result = [[None] * N for _ in range(N)]

    # K*N cycles to push all activations in, plus the N + N - 1 drain tail.
    total_cycles = K * N + 2 * N - 1

    for cycle in range(total_cycles):
        await FallingEdge(dut.clk_i)

        # Drive activations: all K banks streamed back-to-back, staggered per row.
        for i in range(N):
            t = cycle - i   # diagonal-corrected time for row i
            k = t // N      # which bank
            m = t % N       # which row within that bank

            if 0 <= t < K * N:
                dut.act_sel_n_i[i].value   = k % 2
                dut.act_n_i[i].value       = act_banks[k][m][i]
                dut.act_valid_n_i[i].value = 1
            else:
                dut.act_n_i[i].value       = 0
                dut.act_valid_n_i[i].value = 0

        await ReadOnly()

        # Capture output: accumulated result drains N cycles after the last bank.
        # Output row m at column j: m = cycle - K*N - j.
        for j in range(N):
            if dut.psum_out_valid_n_o[j].value == 1:
                m = cycle - K * N - j
                if 0 <= m < N:
                    result[m][j] = dut.psum_out_n_o[j].value.to_signed()

    for m in range(N):
        for j in range(N):
            assert result[m][j] is not None, \
                f"accumulated output row {m}, col {j}: not captured"
    for m, row_out in enumerate(result):
        cocotb.log.info(f"  → accumulated output row {m}: {row_out}")
    return result



# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def reset_test(dut):
    """Verify that all psum outputs are 0 after reset with no inputs driven."""
    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)
    await FallingEdge(dut.rst_i)


@cocotb.test()
async def test_random_matmul_matrix(dut):
    """
    Random single-tile matrix-matrix multiply: N×N activations × N×N weights.
    """
    N = dut.N.value.to_unsigned()
    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)

    act_matrix = [[random.randint(-128, 127) for _ in range(N)] for _ in range(N)]
    weights    = [[random.randint(-128, 127) for _ in range(N)] for _ in range(N)]
    expected   = mat_mat_mul_ref(act_matrix, weights)

    cocotb.log.info(f"N={N}")
    cocotb.log.info(f"act_matrix={act_matrix}")
    cocotb.log.info(f"weights={weights}")
    cocotb.log.info(f"expected={expected}")

    cocotb.start_soon(load_weights(dut, N, weights))
    for _ in range(N):
        await FallingEdge(dut.clk_i)
    results = await stream_activation_matrix(dut, N, act_matrix)

    for m, (row_got, row_exp) in enumerate(zip(results, expected)):
        for j, (got, exp) in enumerate(zip(row_got, row_exp)):
            cocotb.log.info(f"out[{m}][{j}] = {got}  (expected {exp})")
            assert got == exp, f"row {m}, col {j}: expected {exp}, got {got}"

    await FallingEdge(dut.clk_i)


@cocotb.test()
async def test_tiled_matmul_k(dut):
    """
    Tiled matmul along the K (inner) dimension with automatic on-chip accumulation.

    Computes one N×N output tile C of a larger matrix multiply where both A and W
    have been partitioned into K=8 tiles along their shared inner dimension:

        C = A_0 @ W_0  +  A_1 @ W_1  +  ...  +  A_{K-1} @ W_{K-1}

    All K weight tiles are loaded back-to-back via shadow buffering. All K
    activation tiles are then streamed back-to-back with no inter-tile gap,
    causing the DUT to accumulate the partial products internally. A single
    accumulated N×N result is read from the output and compared against the
    reference.
    """
    N = dut.N.value.to_unsigned()
    K = 8  # number of tiles along the inner (reduction) dimension

    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)

    act_banks    = [[[random.randint(-128, 127) for _ in range(N)] for _ in range(N)] for _ in range(K)]
    weight_banks = [[[random.randint(-128, 127) for _ in range(N)] for _ in range(N)] for _ in range(K)]

    expected = tiled_matmul_ref(act_banks, weight_banks)

    for k in range(K):
        cocotb.log.info(f"act_banks[{k}]={act_banks[k]}")
        cocotb.log.info(f"weight_banks[{k}]={weight_banks[k]}")
    cocotb.log.info(f"expected (accumulated C tile)={expected}")

    cocotb.start_soon(load_weight_banks(dut, N, weight_banks))
    for _ in range(N):
        await FallingEdge(dut.clk_i)
    result = await stream_activation_banks_tiled(dut, N, act_banks)

    cocotb.log.info(f"hardware accumulated result={result}")

    for m in range(N):
        for j in range(N):
            got = result[m][j]
            exp = expected[m][j]
            assert got == exp, f"tile row {m}, col {j}: expected {exp}, got {got}"

    await FallingEdge(dut.clk_i)


@cocotb.test()
async def test_full_square_matmul(dut):
    """
    Full (T*N) × (T*N) square matrix multiply, computed as T×T independent output tiles.

    For each output tile C[i][j]:
      - Load weight banks W[0][j] .. W[T-1][j] (the j-th column of weight tiles)
      - Stream activation banks A[i][0] .. A[i][T-1] back-to-back (the i-th row of act tiles)
      - DUT accumulates internally → one N×N result tile

    This is exactly the test_tiled_matmul_k flow, repeated T×T times.
    """
    N = dut.N.value.to_unsigned()
    T = 4  # tile grid dimension: full matrix is (T*N) × (T*N)

    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)

    # Generate full matrices and slice into N×N tiles.
    # A_tile[i][k]: rows i*N..i*N+N-1, cols k*N..k*N+N-1
    # W_tile[k][j]: rows k*N..k*N+N-1, cols j*N..j*N+N-1
    A_full = [[random.randint(-7, 7) for _ in range(T * N)] for _ in range(T * N)]
    W_full = [[random.randint(-7, 7) for _ in range(T * N)] for _ in range(T * N)]

    A_tile = [
        [
            [A_full[i * N + m][k * N : k * N + N] for m in range(N)]
            for k in range(T)
        ]
        for i in range(T)
    ]
    W_tile = [
        [
            [W_full[k * N + r][j * N : j * N + N] for r in range(N)]
            for j in range(T)
        ]
        for k in range(T)
    ]

    C_ref = full_matmul_ref(A_full, W_full)
    C_hw  = [[None] * T for _ in range(T)]

    cocotb.log.info(f"N={N}, T={T}, full matrix size={T*N}×{T*N}")

    for i in range(T):
        for j in range(T):
            cocotb.log.info(f"--- tile [{i}][{j}] ---")

            # Load the T weight banks for this output tile column j
            weight_banks = [W_tile[k][j] for k in range(T)]
            cocotb.start_soon(load_weight_banks(dut, N, weight_banks))
            for _ in range(N):
                await FallingEdge(dut.clk_i)

            # Stream the T activation banks for this output tile row i
            act_banks = [A_tile[i][k] for k in range(T)]
            C_hw[i][j] = await stream_activation_banks_tiled(dut, N, act_banks)

            cocotb.log.info(f"  C_hw[{i}][{j}] = {C_hw[i][j]}")

    for i in range(T):
        for j in range(T):
            for m in range(N):
                for n in range(N):
                    got = C_hw[i][j][m][n]
                    exp = C_ref[i * N + m][j * N + n]
                    assert got == exp, (
                        f"C[{i*N+m}][{j*N+n}] (tile [{i}][{j}] row {m} col {n}): "
                        f"expected {exp}, got {got}"
                    )

    cocotb.log.info("test_full_square_matmul PASSED")
    await FallingEdge(dut.clk_i)


tests = [
    "reset_test",
    "test_random_matmul_matrix",
    "test_tiled_matmul_k",
    "test_full_square_matmul",
]

proj_path = Path("./src/sysray").resolve()
SOURCES   = [proj_path / "sysray_nxn.sv", proj_path / "pe.sv"]


@pytest.mark.parametrize("N", [2, 8])
@pytest.mark.parametrize("testcase", tests)
def test_sysray_nxn_each(N, testcase):
    run_test(
        sources=SOURCES,
        module_name="test_sysray_nxn",
        hdl_toplevel="sysray_nxn",
        parameters={"N": N},
        testcase=testcase,
        sims=['icarus']
    )


@pytest.mark.parametrize("N", [2, 8])
def test_sysray_nxn_all(N):
    run_test(
        sources=SOURCES,
        module_name="test_sysray_nxn",
        hdl_toplevel="sysray_nxn",
        parameters={"N": N},
        sims=['icarus']
    )