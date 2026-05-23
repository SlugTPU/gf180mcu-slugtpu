import pytest
import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from pathlib import Path
from runner import run_test

def _drive_master(dut, adr, dat_w, we, stb, cyc, sel):
    dut.m_adr_i.value   = adr
    dut.m_dat_w_i.value = dat_w
    dut.m_we_i.value    = we
    dut.m_stb_i.value   = stb
    dut.m_cyc_i.value   = cyc
    dut.m_sel_i.value   = sel


def _drive_slaves(dut, s0_dat_r=0, s0_ack=0, s1_dat_r=0, s1_ack=0):
    dut.s0_dat_r_i.value = s0_dat_r
    dut.s0_ack_i.value   = s0_ack
    dut.s1_dat_r_i.value = s1_dat_r
    dut.s1_ack_i.value   = s1_ack


@cocotb.test()
async def route_to_s0(dut):
    """sel=0, then traffic reaches slave 0, slave 1 sees idle"""
    dut.sel_i.value = 0
    _drive_master(dut, adr=0x100, dat_w=0xDEAD, we=1, stb=1, cyc=1, sel=0xF)
    _drive_slaves(dut, s0_dat_r=0xCAFE, s0_ack=1, s1_dat_r=0xBEEF, s1_ack=1)
    await Timer(1, "ns")

    assert int(dut.s0_adr_o.value)   == 0x100
    assert int(dut.s0_dat_w_o.value) == 0xDEAD
    assert int(dut.s0_stb_o.value)   == 1
    assert int(dut.s0_cyc_o.value)   == 1

    # slave 1 must be idle
    assert int(dut.s1_stb_o.value)   == 0
    assert int(dut.s1_cyc_o.value)   == 0

    # master sees slave 0's response, not slave 1's
    assert int(dut.m_dat_r_o.value)  == 0xCAFE
    assert int(dut.m_ack_o.value)    == 1


@cocotb.test()
async def route_to_s1(dut):
    """sel=1, then traffic reaches slave 1, slave 0 sees idle"""
    dut.sel_i.value = 1
    _drive_master(dut, adr=0x200, dat_w=0xF00D, we=0, stb=1, cyc=1, sel=0x3)
    _drive_slaves(dut, s0_dat_r=0xCAFE, s0_ack=1, s1_dat_r=0xBEEF, s1_ack=1)
    await Timer(1, "ns")

    assert int(dut.s1_adr_o.value)   == 0x200
    assert int(dut.s1_dat_w_o.value) == 0xF00D
    assert int(dut.s1_stb_o.value)   == 1
    assert int(dut.s1_cyc_o.value)   == 1

    # slave 0 must be idle
    assert int(dut.s0_stb_o.value)   == 0
    assert int(dut.s0_cyc_o.value)   == 0

    # master sees slave 1's response
    assert int(dut.m_dat_r_o.value)  == 0xBEEF
    assert int(dut.m_ack_o.value)    == 1


@cocotb.test()
async def idle_master_idles_both_slaves(dut):
    """master cyc=stb=0 then neither slave sees a transaction, both sel values"""
    for sel in (0, 1):
        dut.sel_i.value = sel
        _drive_master(dut, adr=0, dat_w=0, we=0, stb=0, cyc=0, sel=0)
        await Timer(1, "ns")
        assert int(dut.s0_stb_o.value) == 0
        assert int(dut.s0_cyc_o.value) == 0
        assert int(dut.s1_stb_o.value) == 0
        assert int(dut.s1_cyc_o.value) == 0


proj_path = Path("./src").resolve()
sources = [proj_path / "spi" / "wb_demux_1to2.sv"]


def test_wb_demux_1to2():
    run_test(
        parameters={},
        sources=sources,
        module_name="test_wb_demux_1to2",
        hdl_toplevel="wb_demux_1to2",
    )
