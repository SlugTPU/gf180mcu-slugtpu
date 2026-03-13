import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
import random


async def write(dut, addr, data):
    await FallingEdge(dut.clk_i)
    dut.addr_i.value    = addr
    dut.wr_data_i.value = data
    dut.rw_mode_i.value = 1
    dut.en_i.value      = 1
    await FallingEdge(dut.clk_i)
    dut.en_i.value      = 0

async def read(dut, addr):
    await FallingEdge(dut.clk_i)
    dut.addr_i.value    = addr
    dut.rw_mode_i.value = 0
    dut.en_i.value      = 1
    await FallingEdge(dut.clk_i)
    dut.en_i.value      = 0
    return dut.rd_data_o.value.to_unsigned()


@cocotb.test()
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)


@cocotb.test()
async def test_single_write_read(dut):
    """Write a known value to one address and read it back."""
    clk_i     = dut.clk_i
    rst_i     = dut.rst_i

    dut.en_i.value      = 0
    dut.rw_mode_i.value = 0
    dut.addr_i.value    = 0
    dut.wr_data_i.value = 0

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)

    addr = 0x05
    data = 0xDEADBEEFCAFEBABE

    await write(dut, addr, data)
    got = await read(dut, addr)

    # print(f"expected {data:#018x}, got {got:#018x}")
    assert got == data, f"Expected {data:#018x}, got {got:#018x}"
    await FallingEdge(clk_i)


@cocotb.test()
async def test_sequential_write_read(dut):
    """Write to all 256 addresses then read them all back."""
    clk_i     = dut.clk_i
    rst_i     = dut.rst_i

    dut.en_i.value      = 0
    dut.rw_mode_i.value = 0
    dut.addr_i.value    = 0
    dut.wr_data_i.value = 0

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)

    ref = {}
    for addr in range(256):
        data      = random.randint(0, 2**64 - 1)
        ref[addr] = data
        await write(dut, addr, data)

    errors = 0
    for addr in range(256):
        got = await read(dut, addr)
        exp = ref[addr]
        # print(f"addr={addr:#04x} expected={exp:#018x} got={got:#018x}")
        if got != exp:
            errors += 1

    assert errors == 0, f"Sequential write/read test failed with {errors} errors"
    await FallingEdge(clk_i)


@cocotb.test()
async def test_random_access(dut):
    """Random address/data writes then verify with reads."""
    clk_i     = dut.clk_i
    rst_i     = dut.rst_i

    dut.en_i.value      = 0
    dut.rw_mode_i.value = 0
    dut.addr_i.value    = 0
    dut.wr_data_i.value = 0

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)

    ref     = {}
    NUM_OPS = 64

    for _ in range(NUM_OPS):
        addr      = random.randint(0, 255)
        data      = random.randint(0, 2**64 - 1)
        ref[addr] = data
        await write(dut, addr, data)

    errors = 0
    for addr, exp in ref.items():
        got = await read(dut, addr)
        # print(f"addr={addr:#04x} expected={exp:#018x} got={got:#018x}")
        if got != exp:
            errors += 1

    assert errors == 0, f"Random access test failed with {errors} errors"
    await FallingEdge(clk_i)


@cocotb.test()
async def test_overwrite(dut):
    """Write to same address twice, verify second write wins."""
    clk_i     = dut.clk_i
    rst_i     = dut.rst_i

    dut.en_i.value      = 0
    dut.rw_mode_i.value = 0
    dut.addr_i.value    = 0
    dut.wr_data_i.value = 0

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)

    addr   = 0x10
    first  = 0x1111111111111111
    second = 0x2222222222222222

    await write(dut, addr, first)
    await write(dut, addr, second)
    got = await read(dut, addr)

    # print(f"expected={second:#018x} got={got:#018x}")
    assert got == second, f"Expected {second:#018x} after overwrite, got {got:#018x}"
    await FallingEdge(clk_i)


tests = [
    "reset_test",
    "test_single_write_read",
    "test_sequential_write_read",
    "test_random_access",
    "test_overwrite",
]

proj_path = Path("./rtl").resolve()
sources = [
    proj_path / "sram/sram_8x256.sv",
    proj_path / "lib/sram/cells/gf180mcu_ocd_ip_sram__sram256x8m8wm1/gf180mcu_ocd_ip_sram__sram256x8m8wm1.v",
]

@pytest.mark.parametrize("testcase", tests)
def test_sram_each(testcase):
    """Runs each test independently. Continues on test failure."""
    run_test(parameters={}, sources=sources, module_name="test_sram", hdl_toplevel="sram_8x256", testcase=testcase, sims=['icarus'])

def test_sram_all():
    """Runs all tests sequentially in one simulation."""
    run_test(parameters={}, sources=sources, module_name="test_sram", hdl_toplevel="sram_8x256", sims=['icarus'])