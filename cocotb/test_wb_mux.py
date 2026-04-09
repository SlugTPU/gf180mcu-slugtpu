import pytest
import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, ClockCycles
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
import random

async def wb_write(dut, prefix, addr, data, sel=0xF):
    """single write on master port `prefix` (m0 or m1)."""
    adr   = getattr(dut, f"{prefix}_adr")
    dat_w = getattr(dut, f"{prefix}_dat_w")
    we    = getattr(dut, f"{prefix}_we")
    stb   = getattr(dut, f"{prefix}_stb")
    cyc   = getattr(dut, f"{prefix}_cyc")
    sel_s = getattr(dut, f"{prefix}_sel")
    ack   = getattr(dut, f"{prefix}_ack")

    await FallingEdge(dut.clk_i)
    adr.value   = addr
    dat_w.value = data
    we.value    = 1
    stb.value   = 1
    cyc.value   = 1
    sel_s.value = sel

    while True:
        await RisingEdge(dut.clk_i)
        if ack.value == 1:
            break

    await FallingEdge(dut.clk_i)
    stb.value = 0
    cyc.value = 0
    we.value  = 0


async def wb_read(dut, prefix, addr, sel=0xF):
    """single read on master port `prefix`. Returns int."""
    adr   = getattr(dut, f"{prefix}_adr")
    dat_w = getattr(dut, f"{prefix}_dat_w")
    we    = getattr(dut, f"{prefix}_we")
    stb   = getattr(dut, f"{prefix}_stb")
    cyc   = getattr(dut, f"{prefix}_cyc")
    sel_s = getattr(dut, f"{prefix}_sel")
    ack   = getattr(dut, f"{prefix}_ack")
    dat_r = getattr(dut, f"{prefix}_dat_r")

    await FallingEdge(dut.clk_i)
    adr.value   = addr
    dat_w.value = 0
    we.value    = 0
    stb.value   = 1
    cyc.value   = 1
    sel_s.value = sel

    while True:
        await RisingEdge(dut.clk_i)
        if ack.value == 1:
            break

    val = dat_r.value.to_unsigned()
    await FallingEdge(dut.clk_i)
    stb.value = 0
    cyc.value = 0
    return val


def idle_master(dut, prefix):
    getattr(dut, f"{prefix}_adr").value   = 0
    getattr(dut, f"{prefix}_dat_w").value = 0
    getattr(dut, f"{prefix}_we").value    = 0
    getattr(dut, f"{prefix}_stb").value   = 0
    getattr(dut, f"{prefix}_cyc").value   = 0
    getattr(dut, f"{prefix}_sel").value   = 0


async def init(dut):
    await clock_start(dut.clk_i)
    idle_master(dut, "m0")
    idle_master(dut, "m1")
    dut.tpu_active.value = 0
    await reset_sequence(dut.clk_i, dut.rst_i)

# tests

@cocotb.test()
async def test_m0_write_read(dut):
    """Host/spi-side (m0) can write and read back through the mux."""
    await init(dut)
    dut.tpu_active.value = 0

    addr = 0x10
    data = 0xCAFEBABE
    await wb_write(dut, "m0", addr, data)
    got = await wb_read(dut, "m0", addr)
    assert got == data, f"m0: expected {data:#010x}, got {got:#010x}"


@cocotb.test()
async def test_m1_write_read(dut):
    """TPU-side (m1) can write and read back when tpu_active=1."""
    await init(dut)
    dut.tpu_active.value = 1

    addr = 0x20
    data = 0xDEADBEEF
    await wb_write(dut, "m1", addr, data)
    got = await wb_read(dut, "m1", addr)
    assert got == data, f"m1: expected {data:#010x}, got {got:#010x}"


@cocotb.test()
async def test_inactive_port_blocked(dut):
    """Inactive master gets no ack and reads zero."""
    await init(dut)
    dut.tpu_active.value = 0

    # m1 is inactive — ack should stay low
    getattr(dut, "m1_adr").value   = 0x10
    getattr(dut, "m1_stb").value   = 1
    getattr(dut, "m1_cyc").value   = 1
    getattr(dut, "m1_sel").value   = 0xF

    await ClockCycles(dut.clk_i, 4)

    assert dut.m1_ack.value == 0, "Inactive port should never get ack"
    assert dut.m1_dat_r.value == 0, "Inactive port should read zero"

    idle_master(dut, "m1")


@cocotb.test()
async def test_switch_preserves_data(dut):
    """Write via m0, switch to m1, read same address from m1."""
    await init(dut)

    addr = 0x40
    data = 0x12345678

    # write via host side
    dut.tpu_active.value = 0
    await wb_write(dut, "m0", addr, data)

    # switch to TPU side, read back
    await FallingEdge(dut.clk_i)
    dut.tpu_active.value = 1
    await ClockCycles(dut.clk_i, 1)

    got = await wb_read(dut, "m1", addr)
    assert got == data, f"Cross-port: expected {data:#010x}, got {got:#010x}"


@cocotb.test()
async def test_no_crosstalk(dut):
    """write from m0 and m1 to different addresses don't interfere."""
    await init(dut)

    addr_a, data_a = 0x80, 0xAAAAAAAA
    addr_b, data_b = 0x84, 0xBBBBBBBB

    # m0 writes addr_a
    dut.tpu_active.value = 0
    await wb_write(dut, "m0", addr_a, data_a)

    # m1 writes addr_b
    dut.tpu_active.value = 1
    await wb_write(dut, "m1", addr_b, data_b)

    #verify
    got_b = await wb_read(dut, "m1", addr_b)
    assert got_b == data_b, f"addr_b: expected {data_b:#010x}, got {got_b:#010x}"

    dut.tpu_active.value = 0
    got_a = await wb_read(dut, "m0", addr_a)
    assert got_a == data_a, f"addr_a: expected {data_a:#010x}, got {got_a:#010x}"


@cocotb.test()
async def test_byte_sel(dut):
    """Byte-lane select writes partial words correctly."""
    await init(dut)
    dut.tpu_active.value = 0

    addr = 0xC0

    # fill with known pattern
    await wb_write(dut, "m0", addr, 0xFFFFFFFF)

    # overwrite only byte 1
    await wb_write(dut, "m0", addr, 0x00AA0000, sel=0x4)  # sel[2] = byte 2

    got = await wb_read(dut, "m0", addr)
    assert got == 0xFFAAFFFF, f"byte_sel: expected 0xFFAAFFFF, got {got:#010x}"


# Runner

tests = [
    "test_m0_write_read",
    "test_m1_write_read",
    "test_inactive_port_blocked",
    "test_switch_preserves_data",
    "test_no_crosstalk",
    "test_byte_sel",
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "dram/wb_mux_2to1.sv",
    proj_path / "dram/wb_mem_model.sv",
    proj_path / "dram/wb_mux_tb_top.sv",
]

@pytest.mark.parametrize("testcase", tests)
def test_wb_mux_each(testcase):
    run_test(
        parameters={},
        sources=sources,
        module_name="test_wb_mux",
        hdl_toplevel="wb_mux_tb_top",
        testcase=testcase,
        sims=["icarus"],
    )

def test_wb_mux_all():
    run_test(
        parameters={},
        sources=sources,
        module_name="test_wb_mux",
        hdl_toplevel="wb_mux_tb_top",
        sims=["icarus"],
    )