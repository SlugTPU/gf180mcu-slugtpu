import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer, RisingEdge, ReadOnly
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
import random

from test_scalar_stage import scalar_pipe_ref

# ---------------------------------------------------------------------------
# Helpers shared with the existing testbenches
# ---------------------------------------------------------------------------

def pack_bytes(values: list[int]) -> int:
    """Pack 8 x 8-bit ints into a single 64-bit integer (index 0 = LSB)."""
    result = 0
    for i, v in enumerate(values):
        result |= (v & 0xFF) << (i * 8)
    return result


def vec_mat_mul_ref(acts, weights):
    N = len(acts)
    return [sum(acts[i] * weights[i][j] for i in range(N)) for j in range(N)]


def mat_mat_mul_ref(act_matrix, weights):
    return [vec_mat_mul_ref(act_row, weights) for act_row in act_matrix]


def tiled_matmul_ref(act_banks, weight_banks):
    N = len(act_banks[0])
    K = len(act_banks)
    result = [[0] * N for _ in range(N)]
    for k in range(K):
        partial = mat_mat_mul_ref(act_banks[k], weight_banks[k])
        for m in range(N):
            for j in range(N):
                result[m][j] += partial[m][j]
    return result


# ---------------------------------------------------------------------------
# SRAM interface helper (mirrors scalar_stage_sram testbench)
# ---------------------------------------------------------------------------

class SramInterface:
    """
    Wraps the act or weight SRAM port signals so tests can reuse
    load_address / read_data / write_data without duplicating driver code.
    """

    def __init__(self, dut, prefix: str):
        """
        prefix is either 'act' or 'weight', matching signal naming in
        compute_core: act_addr_i, act_load_valid_i, weight_addr_i, ...
        """
        self.dut    = dut
        self.prefix = prefix

    def _sig(self, name):
        return getattr(self.dut, f"{self.prefix}_{name}")

    async def load_address(self, rw_mode: str, addr, count):
        assert self._sig("load_ready_o").value == 1
        self._sig("transaction_rw_mode_i").value = 1 if rw_mode == 'w' else 0
        self._sig("addr_i").value                = int(addr)
        self._sig("transaction_amount_i").value  = int(count)
        self._sig("load_valid_i").value           = 1
        await FallingEdge(self.dut.clk_i)
        assert self._sig("load_ready_o").value == 0
        self._sig("load_valid_i").value = 0

    async def read_data(self, expected_vals):
        for val in expected_vals:
            assert self._sig("load_ready_o").value == 0
            assert self._sig("downstream_ready_o").value == 1
            self._sig("rd_ready_i").value = 1
            await FallingEdge(self.dut.clk_i)
            assert self._sig("rd_data_o").value == val
        self._sig("rd_ready_i").value = 0
        assert self._sig("load_ready_o").value == 1
        assert self._sig("downstream_ready_o").value == 0

    async def read_data_no_ready(self, expected_vals):
        for val in expected_vals:
            assert self._sig("load_ready_o").value == 0
            assert self._sig("downstream_ready_o").value == 1
            self._sig("rd_ready_i").value = 0
            await FallingEdge(self.dut.clk_i)
            assert self._sig("rd_data_o").value == val
        self._sig("rd_ready_i").value = 0
        assert self._sig("load_ready_o").value == 1
        assert self._sig("downstream_ready_o").value == 0

    async def write_data(self, data):
        for val in data:
            self._sig("wr_data_i").value   = int(val)
            assert self._sig("load_ready_o").value == 0
            assert self._sig("downstream_ready_o").value == 1
            self._sig("wr_valid_i").value  = 1
            await FallingEdge(self.dut.clk_i)
        self._sig("wr_valid_i").value = 0
        assert self._sig("load_ready_o").value == 1
        assert self._sig("downstream_ready_o").value == 0


# ---------------------------------------------------------------------------
# Reset helper
# ---------------------------------------------------------------------------

async def do_reset(dut):
    # scalar stage controls
    dut.load_bias_en_i.value   = 0
    dut.load_zp_en_i.value     = 0
    dut.load_scale_en_i.value  = 0
    # mxu controls
    dut.act_enable_i.value     = 0
    dut.weight_enable_i.value  = 0
    # act SRAM port
    dut.act_addr_i.value                   = 0
    dut.act_transaction_amount_i.value     = 0
    dut.act_transaction_rw_mode_i.value    = 0
    dut.act_load_valid_i.value             = 0
    dut.act_wr_data_i.value                = 0
    dut.act_wr_valid_i.value               = 0
    dut.act_rd_ready_i.value               = 0
    # weight SRAM port
    dut.weight_addr_i.value                = 0
    dut.weight_transaction_amount_i.value  = 0
    dut.weight_transaction_rw_mode_i.value = 0
    dut.weight_load_valid_i.value          = 0
    dut.weight_wr_data_i.value             = 0
    dut.weight_wr_valid_i.value            = 0
    dut.weight_rd_ready_i.value            = 0

    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)
    await FallingEdge(dut.clk_i)


async def load_scalar_values(dut, scalar_params, start_addr):
    a_mem = SramInterface(dut, "act")
    bias  = scalar_params['bias']
    zp    = scalar_params['zp']
    mul   = scalar_params['mul']

    # --- pack scalar parameters and load into act SRAM ---
    bias_packed = [(b1 | (b2 << 32)) for b1, b2 in zip(bias[0::2], bias[1::2])]
    zp_packed   = [(z1 | (z2 << 32)) for z1, z2 in zip(zp[0::2],   zp[1::2])]
    mul_packed  = [(m1 | (m2 << 32)) for m1, m2 in zip(mul[0::2],  mul[1::2])]

    param_words = bias_packed + zp_packed + mul_packed  # 4+4+4 = 12 words

    await a_mem.load_address('w', start_addr, len(param_words))
    await a_mem.write_data(param_words)

    # Read scalar params back into the scalar_stage registers
    await a_mem.load_address('r', start_addr, 4)
    dut.load_bias_en_i.value = 1
    await a_mem.read_data(bias_packed)
    await FallingEdge(dut.clk_i)
    dut.load_bias_en_i.value = 0

    await a_mem.load_address('r', start_addr+4, 4)
    dut.load_zp_en_i.value = 1
    await a_mem.read_data(zp_packed)
    await FallingEdge(dut.clk_i)
    dut.load_zp_en_i.value = 0

    await a_mem.load_address('r', start_addr+8, 4)
    dut.load_scale_en_i.value = 1
    await a_mem.read_data(mul_packed)
    await FallingEdge(dut.clk_i)
    dut.load_scale_en_i.value = 0

async def load_weight_banks_via_sram(dut, N, weight_banks):
    K     = len(weight_banks)
    w_mem = SramInterface(dut, "weight")

    # Write rows in bottom-to-top order (flipped) so sequential reads
    # come out in the order the MXU expects.
    all_rows_packed = []
    for k in range(K):
        for bk_idx in range(N):
            all_rows_packed.append(pack_bytes(weight_banks[k][N - 1 - bk_idx]))

    total_words = K * N
    await w_mem.load_address('w', 0, total_words)
    await w_mem.write_data(all_rows_packed)

    # Read sequentially from address 0 — no index math needed.
    await w_mem.load_address('r', 0, total_words)
    dut.weight_enable_i.value = 1
    await w_mem.read_data_no_ready(all_rows_packed)
    await FallingEdge(dut.clk_i)
    dut.weight_enable_i.value = 0

async def stream_acts_and_capture(dut, N, act_banks, expected_output, start_addr = 12, write_addr = 100):
    """
    Write activation banks into the act SRAM (via the scalar_stage_sram
    path), assert act_enable_i so the SRAM feeds the MXU, and collect the
    N output rows that come back through the scalar post-processing stage.

    Returns a list of N packed 64-bit words (one per output row) after
    scalar post-processing, matching the format written back to act SRAM.
    """
    a_mem = SramInterface(dut, "act")
    K = len(act_banks)
    all_act_packed = []
    for k in range(K):
        for row in act_banks[k]:
            all_act_packed.append(pack_bytes(row))

    total_act_words = K * N
    await a_mem.load_address('w', start_addr, total_act_words)
    await a_mem.write_data(all_act_packed)

    dut.act_enable_i.value = 1
    await a_mem.load_address('r', start_addr, total_act_words)
    await a_mem.read_data_no_ready(all_act_packed)

    if dut.act_load_ready_o.value != 1:
        await RisingEdge(dut.act_load_ready_o)
    await FallingEdge(dut.clk_i)
    dut.act_enable_i.value = 0
    await a_mem.load_address('w', write_addr, N)

    if dut.act_load_ready_o.value != 1:
        await RisingEdge(dut.act_load_ready_o)
    await FallingEdge(dut.clk_i)

    # Verify the results were actually written back to SRAM
    await a_mem.load_address('r', write_addr, N)
    expected_packed = []
    for row in expected_output:
        expected_packed.append(pack_bytes(row))
    await a_mem.read_data(expected_packed)


# ===========================================================================
# Tests
# ===========================================================================

@cocotb.test()
async def test_reset(dut):
    """Verify the core comes out of reset without assertions firing."""
    await do_reset(dut)
    # Check ready signals are de-asserted in a sensible state
    assert dut.act_load_ready_o.value    == 1
    assert dut.weight_load_ready_o.value == 1

@cocotb.test()
async def test_act_sram_write_read(dut):
    """
    Confirm the activation SRAM (inside scalar_stage_sram) accepts writes
    and returns the same data on a subsequent read, independent of the MXU.
    """
    await do_reset(dut)
    a_mem = SramInterface(dut, "act")
 
    payload = [random.randint(0, 0xFFFFFFFFFFFFFFFF) for _ in range(8)]
 
    await a_mem.load_address('w', 0, 8)
    await a_mem.write_data(payload)
 
    await a_mem.load_address('r', 0, 8)
    await a_mem.read_data(payload)
    await FallingEdge(dut.clk_i)
 
 
@cocotb.test()
async def test_weight_sram_write_read(dut):
    """
    Confirm the weight SRAM accepts writes and returns the same data on a
    subsequent read, independent of the MXU.
    """
    await do_reset(dut)
    w_mem = SramInterface(dut, "weight")
 
    payload = [random.randint(0, 0xFFFFFFFFFFFFFFFF) for _ in range(8)]
 
    await w_mem.load_address('w', 0, 8)
    await w_mem.write_data(payload)
 
    await w_mem.load_address('r', 0, 8)
    await w_mem.read_data(payload)
    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_weight_load_matrix(dut):
    await do_reset(dut)
    N = 8
    weight = [[random.randint(-64, 63) for _ in range(N)] for _ in range(N)]
    await load_weight_banks_via_sram(dut, N, [weight])
    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_single_matmul(dut):
    """
    Test a matmul, with the scalar stages not transforming the output
    ATM, RELU cannot be turned off, so matrices must both be positive only
    """
    await do_reset(dut)
    N = 8

    # weight = [[random.randint(0, 63) for _ in range(N)] for _ in range(N)]
    # act    = [[random.randint(0, 63) for _ in range(N)] for _ in range(N)]
    weight = [[i%2+1 for _ in range(N)] for i in range(N)]
    act    = [[i%2+1 for _ in range(N)] for i in range(N)]

    expected = tiled_matmul_ref([act], [weight])

    bias = [0, 0, 0, 0, 0, 0, 0, 0]
    zp   = [0 for _ in range(8)]
    mul  = [1 << 16 for _ in range(8)]
    scalar_params = {'bias': bias, 'zp': zp, 'mul': mul}

    await load_scalar_values(dut, scalar_params, 0)

    cocotb.start_soon(load_weight_banks_via_sram(dut, N, [weight]))
    for _ in range(9):
        await FallingEdge(dut.clk_i)
    await stream_acts_and_capture(dut, N, [act], expected, 12, 100)
    # for _ in range(30):
    #     await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)

tests = [
    'test_reset',
    'test_act_sram_write_read',
    'test_weight_sram_write_read',
    'test_weight_load_matrix',
    'test_single_matmul',
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "compute_core.sv",
    proj_path / "scalar_units/scalar_stage_sram.sv",
    proj_path / "scalar_units/scalar_stage.sv",
    proj_path / "scalar_units/scalar_pipe.sv",
    proj_path / "scalar_units/add_n.sv",
    proj_path / "scalar_units/scale_n.sv",
    proj_path / "scalar_units/relu_n.sv",
    proj_path / "scalar_units/load_data.sv",
    proj_path / "scalar_units/quantizer_mul.sv",
    proj_path / "common/elastic.sv",
    proj_path / "common/shift.sv",
    proj_path / "sram/memory_transaction.sv",
    proj_path / "sram/sram_8x256_full.sv",
    proj_path / "sram/sram_8x256.sv",
    proj_path / "sysray/sysray_nxn.sv",
    proj_path / "sysray/pe.sv",
    proj_path / "sysray/mxu.sv",
    proj_path / "common/counter.sv",
    proj_path / "tri_shift.sv",
    "./ip/gf180mcu_ocd_ip_sram.git/cells/gf180mcu_ocd_ip_sram__sram256x8m8wm1/gf180mcu_ocd_ip_sram__sram256x8m8wm1.v",
]


@pytest.mark.parametrize("testcase", tests)
def test_compute_core_each(testcase):
    """Runs each test independently. Continues on test failure."""
    run_test(
        parameters={},
        sources=sources,
        module_name="test_compute_core",
        hdl_toplevel="compute_core",
        testcase=testcase,
        sims=['icarus'],
    )


# def test_compute_core_all():
#     """Runs all tests sequentially in one simulation."""
#     run_test(
#         parameters={},
#         sources=sources,
#         module_name="test_compute_core",
#         hdl_toplevel="compute_core",
#         sims=['icarus'],
#     )