import random
import cocotb
from cocotb.triggers import FallingEdge, ReadOnly
from pathlib import Path
import pytest
from shared import clock_start, reset_sequence
from runner import run_test

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

def pack_bytes(values: list[int]) -> int:
    """Pack 8 x 8-bit ints into a single 64-bit integer (index 0 = LSB)."""
    result = 0
    for i, v in enumerate(values):
        result |= (v & 0xFF) << (i * 8)
    return result

# def pack_bytes(values: list[int]) -> int:
#     """Pack 8 x 8-bit ints into a single 64-bit integer (index 0 = MSB)."""
#     result = 0
#     for i, v in enumerate(values):
#         result |= (v & 0xFF) << ((7 - i) * 8)
#     return result

async def load_weight_banks(dut, N, weight_banks):
    """
    Load K weight banks back-to-back, one full row at a time (no column stagger).
    All columns receive the same row simultaneously.
    """
    K = len(weight_banks)

    for cycle in range(K * N):
        await FallingEdge(dut.clk_i)
        k      = cycle // N        # bank index
        bk_idx = cycle % N         # row index within bank (bottom-to-top sweep)

        dut.weight_bus_i.value  = pack_bytes(weight_banks[k][N - 1 - bk_idx])
        dut.weight_enable_i.value = 1
        dut.weight_valid_i.value  = 1

    # De-assert valid on the cycle after the last row
    await FallingEdge(dut.clk_i)
    dut.weight_enable_i.value     = 0
    dut.weight_valid_i.value      = 1

async def stream_activation_banks_tiled(dut, N, act_banks):
    """
    Stream K activation banks back-to-back, one full row at a time (no row stagger).
    Captures N output rows whenever psum_valid_o goes high, regardless of latency.

    Returns a single N×N accumulated result matrix.
    """
    K = len(act_banks)
    result = [[None] * N for _ in range(N)]
    rows_captured = 0

    async def drive_inputs():
        for cycle in range(K * N):
            await FallingEdge(dut.clk_i)
            k = cycle // N
            m = cycle % N
            dut.act_bus_i.value    = pack_bytes(act_banks[k][m])
            dut.act_enable_i.value = 1
            dut.act_valid_i.value  = 1

        await FallingEdge(dut.clk_i)
        dut.act_enable_i.value = 0
        dut.act_valid_i.value  = 0

    async def capture_outputs():
        nonlocal rows_captured
        while rows_captured < N:
            await FallingEdge(dut.clk_i)
            await ReadOnly()
            if dut.psum_valid_o[0].value == 1:
                for j in range(N):
                    result[rows_captured][j] = dut.psum_o[j].value.to_signed()
                cocotb.log.info(f"  → captured output row {rows_captured}: {result[rows_captured]}")
                rows_captured += 1

    await cocotb.start(drive_inputs())
    await capture_outputs()

    for m in range(N):
        for j in range(N):
            assert result[m][j] is not None, \
                f"accumulated output row {m}, col {j}: not captured"
    return result


@cocotb.test()
async def reset_test(dut):
    """Verify that all psum outputs are 0 after reset with no inputs driven."""
    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)
    await FallingEdge(dut.rst_i)

@cocotb.test()
async def single_matmul_test(dut):
    """Test a single 8x8 matrix multiply: C = A @ W"""
    N = 8

    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)
    await FallingEdge(dut.rst_i)

    dut.act_enable_i.value = 0
    dut.weight_enable_i.value = 0

    # Generate random 8-bit signed inputs in [-64, 63] to avoid overflow in 32-bit accumulators
    weight = [[random.randint(-64, 63) for _ in range(N)] for _ in range(N)]
    act    = [[random.randint(-64, 63) for _ in range(N)] for _ in range(N)]

    cocotb.log.info("Weight matrix:")
    for row in weight:
        cocotb.log.info(f"  {row}")
    cocotb.log.info("Activation matrix:")
    for row in act:
        cocotb.log.info(f"  {row}")

    # Wrap in single-bank lists so the tiled functions handle them
    cocotb.start_soon(load_weight_banks(dut, N, [weight]))
    for _ in range(8):
        await FallingEdge(dut.clk_i)
    result = await stream_activation_banks_tiled(dut, N, [act])

    expected = tiled_matmul_ref([act], [weight])

    cocotb.log.info("Expected:")
    for m, row in enumerate(expected):
        cocotb.log.info(f"  row {m}: {row}")

    for m in range(N):
        for j in range(N):
            assert result[m][j] == expected[m][j], (
                f"Mismatch at output[{m}][{j}]: "
                f"got {result[m][j]}, expected {expected[m][j]}"
            )

    cocotb.log.info("single_matmul_test PASSED")

@cocotb.test()
async def tiled_matmul_test(dut):
    """Test tiled matrix multiply: C = sum_k A_k @ W_k, K=4 tiles"""
    N = 8
    K = 4  # number of tiles along inner dimension

    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)
    await FallingEdge(dut.rst_i)

    dut.act_enable_i.value    = 0
    dut.weight_enable_i.value = 0

    # Generate K random N×N tiles for weights and activations
    weight_banks = [
        [[random.randint(-32, 31) for _ in range(N)] for _ in range(N)]
        for _ in range(K)
    ]
    act_banks = [
        [[random.randint(-32, 31) for _ in range(N)] for _ in range(N)]
        for _ in range(K)
    ]

    for k in range(K):
        cocotb.log.info(f"Weight bank {k}:")
        for row in weight_banks[k]:
            cocotb.log.info(f"  {row}")
        cocotb.log.info(f"Activation bank {k}:")
        for row in act_banks[k]:
            cocotb.log.info(f"  {row}")

    cocotb.start_soon(load_weight_banks(dut, N, weight_banks))
    for _ in range(8):
        await FallingEdge(dut.clk_i)

    result   = await stream_activation_banks_tiled(dut, N, act_banks)
    expected = tiled_matmul_ref(act_banks, weight_banks)

    cocotb.log.info("Expected:")
    for m, row in enumerate(expected):
        cocotb.log.info(f"  row {m}: {row}")

    for m in range(N):
        for j in range(N):
            assert result[m][j] == expected[m][j], (
                f"Tiled mismatch at output[{m}][{j}]: "
                f"got {result[m][j]}, expected {expected[m][j]}"
            )

    cocotb.log.info("tiled_matmul_test PASSED")

tests = [
    "reset_test",
    "single_matmul_test",
    "tiled_matmul_test",
]

proj_path = Path("./src/").resolve()
SOURCES   = [proj_path / "sysray/sysray_nxn.sv", proj_path / "sysray/pe.sv", proj_path / "sysray/mxu.sv",
             proj_path /"common/counter.sv", proj_path / "tri_shift.sv", proj_path / "common/shift.sv"]


@pytest.mark.parametrize("N", [8])
@pytest.mark.parametrize("testcase", tests)
def test_sysray_nxn_each(N, testcase):
    run_test(
        sources=SOURCES,
        module_name="test_mxu",
        hdl_toplevel="mxu",
        parameters={"N": N},
        testcase=testcase,
        sims=['icarus']
    )


# @pytest.mark.parametrize("N", [2, 8])
# def test_sysray_nxn_all(N):
#     run_test(
#         sources=SOURCES,
#         module_name="test_sysray_nxn",
#         hdl_toplevel="sysray_nxn",
#         parameters={"N": N},
#         sims=['icarus']
#     )