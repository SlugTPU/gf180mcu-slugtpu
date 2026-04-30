import random
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

from runner import run_test
from spibone_bfm import SpiboneBFM


CLK_PERIOD_NS = 10  # 100 MHz. is much more bigger than the 4x SCK lower bound
SCK_HALF_NS   = 250 # SCK = 2 MHz


async def setup(dut) -> SpiboneBFM:
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD_NS, unit="ns").start())

    bfm = SpiboneBFM(dut, clk_period_ns=CLK_PERIOD_NS, sck_half_ns=SCK_HALF_NS)
    bfm.idle()

    dut.rst_i.value = 1
    await ClockCycles(dut.clk_i, 5)
    dut.rst_i.value = 0
    await ClockCycles(dut.clk_i, 3)

    return bfm


# Tests

@cocotb.test()
async def test_single_write_then_read(dut):
    """write one word, read back in a separate transaction"""
    bfm = await setup(dut)

    addr = 0x40
    word = 0xDEADBEEF

    await bfm.write(addr, [word])
    got = await bfm.read(addr, 1)
    assert got[0] == word, f"got 0x{got[0]:08X} expected 0x{word:08X}"


@cocotb.test()
async def test_burst_write_then_burst_read(dut):
    """burst write 8 words, burst read them back. Address auto-increments."""
    bfm = await setup(dut)

    base = 0x100
    words = [random.randint(0, 2**32 - 1) for _ in range(8)]

    await bfm.write(base, words)
    got = await bfm.read(base, len(words))
    assert got == words, f"\nwrote {[hex(w) for w in words]}\nread  {[hex(w) for w in got]}"


@cocotb.test()
async def test_individual_writes_burst_read(dut):
    """Each word written in its own SPI transaction. read back as one burst."""
    bfm = await setup(dut)

    base = 0x200
    words = [random.randint(0, 2**32 - 1) for _ in range(4)]

    for i, w in enumerate(words):
        await bfm.write(base + i * 4, [w])

    got = await bfm.read(base, len(words))
    assert got == words


@cocotb.test()
async def test_overwrite(dut):
    """second write to the same address replaces the first."""
    bfm = await setup(dut)

    addr = 0x80
    await bfm.write(addr, [0xAAAAAAAA])
    await bfm.write(addr, [0x55555555])
    got = await bfm.read(addr, 1)
    assert got[0] == 0x55555555, f"got 0x{got[0]:08X}"


# Runner

tests = [
    "test_single_write_then_read",
    "test_burst_write_then_burst_read",
    "test_individual_writes_burst_read",
    "test_overwrite",
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "spi/spibone_wb.sv",
    proj_path / "dram/wb_mem_model.sv",
    proj_path / "dram/spibone_dram_tb_top.sv",
]


@pytest.mark.parametrize("testcase", tests)
def test_spibone_dram_each(testcase):
    run_test(
        parameters={},
        sources=sources,
        module_name="test_spibone_dram",
        hdl_toplevel="spibone_dram_tb_top",
        testcase=testcase,
        sims=["icarus"],
    )


def test_spibone_dram_all():
    run_test(
        parameters={},
        sources=sources,
        module_name="test_spibone_dram",
        hdl_toplevel="spibone_dram_tb_top",
        sims=["icarus"],
    )