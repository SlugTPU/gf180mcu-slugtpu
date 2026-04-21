import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, Timer, RisingEdge, First
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
import random

@cocotb.test()
async def test_reset(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()

    print(f"DEBUG: Passed in clock is {1 // sys_clk_mhz}ns")
    Clock(clk_i, 1/sys_clk_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(clk_i)


@cocotb.test()
async def test_initialization(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    m_we_i = dut.m_we_i
    m_stb_i = dut.m_stb_i
    m_cyc_i = dut.m_cyc_i
    m_sel_i = dut.m_sel_i

    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()

    Clock(clk_i, 1/sys_clk_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)
    m_we_i.value = 0
    m_stb_i.value = 0
    m_cyc_i.value = 0
    m_sel_i.value = 0

    """
    since vanilla Wishbone does not really have a ready_o interface without
    sending a read/write request, waiting 200us will do
    """
    await ClockCycles(clk_i, 200 * sys_clk_mhz)

    await FallingEdge(clk_i)

@cocotb.test()
async def test_read(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    m_we_i = dut.m_we_i
    m_stb_i = dut.m_stb_i
    m_cyc_i = dut.m_cyc_i
    m_sel_i = dut.m_sel_i
    m_ack_o = dut.m_ack_o
    m_adr_i = dut.m_adr_i

    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()

    Clock(clk_i, 1/sys_clk_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)
    m_we_i.value = 0
    m_stb_i.value = 1
    m_cyc_i.value = 1
    m_sel_i.value = 0
    m_adr_i.value = 1

    timeout = Timer(1000, unit="us")
    m_ack = RisingEdge(m_ack_o)

    r = await First(timeout, m_ack)
    if r is timeout:
        assert 1 == 0, "Timed out while waiting for m_ack_o to go high"

    await FallingEdge(clk_i)
    m_we_i.value = 0
    m_stb_i.value = 0
    m_cyc_i.value = 0
    m_sel_i.value = 0
    m_adr_i.value = 0
    await RisingEdge(clk_i)

    # go quiet for a bit for checking autorefresh
    await ClockCycles(clk_i, 20)

@cocotb.test()
async def test_write(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    m_we_i = dut.m_we_i
    m_stb_i = dut.m_stb_i
    m_cyc_i = dut.m_cyc_i
    m_sel_i = dut.m_sel_i
    m_ack_o = dut.m_ack_o
    m_adr_i = dut.m_adr_i
    m_dat_i = dut.m_dat_i

    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()

    Clock(clk_i, 1/sys_clk_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)
    m_we_i.value = 1
    m_stb_i.value = 1
    m_cyc_i.value = 1
    m_adr_i.value = 1
    # ignored
    m_sel_i.value = 0

    timeout = Timer(1000, unit="us")
    m_ack = RisingEdge(m_ack_o)

    r = await First(timeout, m_ack)
    if r is timeout:
        assert 1 == 0, "Timed out while waiting for m_ack_o to go high"

    await FallingEdge(clk_i)
    m_we_i.value = 0
    m_stb_i.value = 1
    m_cyc_i.value = 1
    m_sel_i.value = 1
    m_adr_i.value = 1

    timeout = Timer(1000, unit="us")
    m_ack = RisingEdge(m_ack_o)

    r = await First(timeout, m_ack)
    if r is timeout:
        assert 1 == 0, "Timed out while waiting for m_ack_o to go high"

    await FallingEdge(clk_i)
    m_we_i.value = 0
    m_stb_i.value = 0
    m_cyc_i.value = 0
    m_sel_i.value = 0
    m_adr_i.value = 0

    # go quiet for a bit for checking autorefresh
    await ClockCycles(clk_i, 1000)

tests =[
    'test_reset',
    'test_initialization',
    'test_write',
    'test_read',
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "dram" / "wb_sdr_mt48lc16m16a2_7e.sv",
    proj_path / "dram" / "sdram_model_mt48lc16m16a2.v",
    proj_path / "common" / "shift.sv",
    proj_path / "tb_wb_sdr_mt48lc16m16a_7e.sv"
]
parameters = { "sys_clk_mhz_p": 100 }
module_name = "test_wb_sdr_mt48lc16m16a2_7e"
hdl_toplevel="tb_wb_sdr_mt48lc16m16a_7e"
sims = ['icarus', 'verilator']

@pytest.mark.parametrize("testcase", tests)
def test_sdr_ctrl_each(testcase):
    """Runs each test independently. Continues on test failure."""
    run_test(parameters=parameters, sources=sources, module_name=module_name, hdl_toplevel=hdl_toplevel, testcase=testcase, sims=sims)

def test_sdr_ctrl_all():
    """Runs all tests sequentially in one simulation."""
    run_test(parameters=parameters, sources=sources, module_name=module_name, hdl_toplevel=hdl_toplevel, sims=sims)
