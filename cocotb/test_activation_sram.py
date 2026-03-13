import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer, RisingEdge
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
import random

class memory_transaction:
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
            self.dut.downstream_ready_i.value = 1
            await FallingEdge(self.dut.clk_i)
            assert self.dut.rd_data_o.value == val
        self.dut.downstream_ready_i.value = 0
        assert self.dut.load_ready_o.value == 1
        assert self.dut.downstream_ready_o.value == 0


    async def write_data(self, data):
        for val in data:
            self.dut.wr_data_i.value = int(val)
            assert self.dut.load_ready_o.value == 0
            assert self.dut.downstream_ready_o.value == 1
            self.dut.downstream_ready_i.value = 1
            await FallingEdge(self.dut.clk_i)
        self.dut.downstream_ready_i.value = 0
        assert self.dut.load_ready_o.value == 1
        assert self.dut.downstream_ready_o.value == 0

    


@cocotb.test()
async def test_simple(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    wr_data = dut.wr_data_i
    rd_data = dut.rd_data_o

    ready_i = dut.downstream_ready_i
    ready_o = dut.downstream_ready_o

    addr_i = dut.addr_i
    rw_mode = dut.transaction_rw_mode_i
    amount = dut.transaction_amount_i
    load_valid_i = dut.load_valid_i
    load_ready_o = dut.load_ready_o

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(clk_i)

    model = memory_transaction(dut)
    count = 8
    start_addr = 8

    await model.load_address('w', start_addr, count)

    data_i = [(i+1)*10 for i in range(8)]
    await model.write_data(data_i)
    await model.load_address('r', start_addr, count)
    await model.read_data(data_i)


    await FallingEdge(clk_i)



tests =[
    'test_simple'
]

proj_path = Path("./rtl").resolve()
sources = [
    proj_path / "sram/memory_transaction.sv",
    proj_path / "sram/activation_sram.sv",
    proj_path / "sram/sram_8x256.sv",
    proj_path / "lib/sram/cells/gf180mcu_ocd_ip_sram__sram256x8m8wm1/gf180mcu_ocd_ip_sram__sram256x8m8wm1.v",
]

@pytest.mark.parametrize("testcase", tests)
def test_mem_each(testcase):
    """Runs each test independently. Continues on test failure."""
    run_test(parameters={}, sources=sources, module_name="test_activation_sram", hdl_toplevel="activation_sram", testcase=testcase, sims=['icarus'])

def test_mem_all():
    """Runs all tests sequentially in one simulation."""
    run_test(parameters={}, sources=sources, module_name="test_activation_sram", hdl_toplevel="activation_sram", sims=['icarus'])