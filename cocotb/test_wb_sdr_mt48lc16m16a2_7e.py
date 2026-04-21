import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, Timer, RisingEdge
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
import random

@cocotb.test()
async def test_reset(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    Clock(clk_i, 100, unit="us").start()
    await reset_sequence(clk_i, rst_i)

@cocotb.test()
async def test_initialization(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    clk_freq_mhz = 100

    Clock(clk_i, 1/clk_freq_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)

    """
    since vanilla Wishbone does not really have a ready_o interface without
    sending a read/write request, waiting 200us will do
    """
    await ClockCycles(clk_i, clk_freq_mhz * 200)

tests =[
    'test_reset',
    'test_initialization',
]

proj_path = Path("./src").resolve()
sources = [ proj_path / "dram" / "wb_sdr_mt48lc16m16a2_7e.sv"]
parameters = {}
module_name = "test_wb_sdr_mt48lc16m16a2_7e"
hdl_toplevel="wb_sdr_mt48lc16m16a_7e"

@pytest.mark.parametrize("testcase", tests)
def test_sdr_ctrl_each(testcase):
    """Runs each test independently. Continues on test failure."""
    run_test(parameters=parameters, sources=sources, module_name=module_name, hdl_toplevel=hdl_toplevel, testcase=testcase, sims=['icarus'])

def test_sdr_ctrl_all():
    """Runs all tests sequentially in one simulation."""
    run_test(parameters=parameters, sources=sources, module_name=module_name, hdl_toplevel=hdl_toplevel, sims=['icarus'])
