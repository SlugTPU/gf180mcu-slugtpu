import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer, RisingEdge
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
import random

from test_scalar_stage import scalar_pipe_ref

class scalar_stage_sram_interface:
    def __init__(self, dut):
        self.dut = dut
    async def load_address(self, rw_mode : str, addr, count):
        assert self.dut.load_ready_o.value == 1
        self.dut.transaction_rw_mode_i.value = 0
        if rw_mode == 'w':
            self.dut.transaction_rw_mode_i.value = 1
        self.dut.addr_i.value = int(addr)
        self.dut.transaction_amount_i.value = int(count)
        self.dut.load_valid_i.value = 1
        await FallingEdge(self.dut.clk_i)
        assert self.dut.load_ready_o.value == 0
        self.dut.load_valid_i.value = 0

    async def read_data(self, expected_vals):
        for val in expected_vals:
            assert self.dut.load_ready_o.value == 0
            assert self.dut.downstream_ready_o.value == 1
            self.dut.rd_ready_i.value = 1
            await FallingEdge(self.dut.clk_i)
            assert self.dut.rd_data_o.value == val
            print(self.dut.rd_data_o.value)
        self.dut.rd_ready_i.value = 0
        assert self.dut.load_ready_o.value == 1
        assert self.dut.downstream_ready_o.value == 0
    
    async def write_data(self, data):
        for val in data:
            self.dut.wr_data_i.value = int(val)
            assert self.dut.load_ready_o.value == 0
            assert self.dut.downstream_ready_o.value == 1
            self.dut.wr_valid_i.value = 1
            await FallingEdge(self.dut.clk_i)
        self.dut.wr_valid_i.value = 0
        assert self.dut.load_ready_o.value == 1
        assert self.dut.downstream_ready_o.value == 0

async def do_reset(dut):
    dut.load_bias_en_i.value = 0
    dut.load_zp_en_i.value = 0
    dut.load_scale_en_i.value = 0
    dut.data_i.value = [0 for i in range(8)]
    dut.data_valid_i.value = 0
    dut.addr_i.value = 0
    dut.transaction_amount_i.value = 0
    dut.transaction_rw_mode_i.value = 0
    dut.load_valid_i.value = 0
    dut.rd_ready_i.value = 0
    dut.wr_data_i.value = 0
    dut.wr_valid_i.value = 0


    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)
    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_reset(dut):
    await do_reset(dut)

@cocotb.test()
async def test_simple(dut):
    await do_reset(dut)
    mem = scalar_stage_sram_interface(dut)

    bias = [1,1,1,1,2,2,2,2]
    bias_packed = [
        (b1 | (b2 << 32))
        for b1, b2 in zip(bias[0::2], bias[1::2])
    ]
    
    zp = [0 for _ in range (8)]
    zp_packed = [0 for _ in range(4)]
    mul = [1 << 16 for _ in range (8)]
    mul_packed = [
        (b1 | (b2 << 32))
        for b1, b2 in zip(mul[0::2], mul[1::2])
    ]

    # for v in mul_packed:
    #     bits = f"{v:064b}"
    #     grouped = " ".join(bits[i:i+8] for i in range(0, 64, 8))
    #     print(f"0b {grouped}")

    data = [
        [1 for _ in range(8)],
        [2 for _ in range(8)],
        [3 for _ in range(8)],
        [4 for _ in range(8)],
        [5 for _ in range(8)],
        [6 for _ in range(8)],
        [7 for _ in range(8)],
        [8 for _ in range(8)],
    ]
    print(data)

    expected_packed = [
        sum((scalar_pipe_ref(data[i], bias, zp, mul)[j] & 0xFF) << (j * 8)
            for j in range(8))
        for i in range(8)
    ]

    for v in expected_packed:
        bits = f"{v:064b}"
        grouped = " ".join(bits[i:i+8] for i in range(0, 64, 8))
        print(f"0b {grouped}")
    # bias at 0, sf at 4, mul at 8
    await mem.load_address('w', 0, 12)
    await mem.write_data(bias_packed + zp_packed + mul_packed)

    await mem.load_address('r', 0, 4)
    dut.load_bias_en_i.value = 1
    await mem.read_data(bias_packed)
    await FallingEdge(dut.clk_i)
    dut.load_bias_en_i.value = 0

    await mem.load_address('r', 4, 4)
    dut.load_zp_en_i.value = 1
    await mem.read_data(zp_packed)
    await FallingEdge(dut.clk_i)
    dut.load_zp_en_i.value = 0

    await mem.load_address('r', 8, 4)
    dut.load_scale_en_i.value = 1
    await mem.read_data(mul_packed)
    await FallingEdge(dut.clk_i)
    dut.load_scale_en_i.value = 0

    await mem.load_address('w', 20, 8)

    for d in data:
        assert dut.data_ready_o.value == 1
        dut.data_i.value = d
        dut.data_valid_i.value = 1
        await FallingEdge(dut.clk_i)
    dut.data_valid_i.value = 0

    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)

    await mem.load_address('r', 20, 8)
    await mem.read_data(expected_packed)
    await FallingEdge(dut.clk_i)




tests =[
    'test_reset',
    'test_simple',
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "scalar_units/scalar_stage_sram.sv",
    proj_path / "scalar_units/scalar_stage.sv",
    proj_path / "scalar_units/scalar_pipe.sv",
    proj_path / "scalar_units/add_n.sv",
    proj_path / "scalar_units/scale_n.sv",
    proj_path / "scalar_units/relu_n.sv",
    proj_path / "scalar_units/load_data.sv",
    proj_path / "quantizer_mul.sv",
    proj_path / "common/elastic.sv",
    proj_path / "common/shift.sv",
    proj_path / "sram/memory_transaction.sv",
    proj_path / "sram/activation_sram.sv",
    proj_path / "sram/sram_8x256.sv",
    "./ip/gf180mcu_ocd_ip_sram.git/cells/gf180mcu_ocd_ip_sram__sram256x8m8wm1/gf180mcu_ocd_ip_sram__sram256x8m8wm1.v",
]

@pytest.mark.parametrize("testcase", tests)
def test_scalar_mem_each(testcase):
    """Runs each test independently. Continues on test failure."""
    run_test(parameters={}, sources=sources, module_name="test_scalar_stage_sram", hdl_toplevel="scalar_stage_sram", testcase=testcase, sims=['icarus'])

def test_scalar_mem_all():
    """Runs all tests sequentially in one simulation."""
    run_test(parameters={}, sources=sources, module_name="test_scalar_stage_sram", hdl_toplevel="scalar_stage_sram", sims=['icarus'])