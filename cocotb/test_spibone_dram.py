import random
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from cocotbext.spi import SpiMaster, SpiBus, SpiConfig
from shared import reset_sequence, clock_start
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ReadOnly
from spibone_bfm import SpiboneBFM

from runner import run_test


clock_freq_mhz = 100
clock_period = 1/(clock_freq_mhz*10**6)*10**9
spi_config = SpiConfig(
    word_width = 8,     # all parameters optional
    sclk_freq  = (clock_freq_mhz/6) * 10**6,   # these are the defaults
    cpol       = False,
    cpha       = False,
    msb_first  = True,
    cs_active_low = True # optional (assumed True)
)

# Tests

@cocotb.test()
async def reset_test(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    await clock_start(clk_i, period_ns=clock_period)
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(clk_i)


@cocotb.test()
async def test_single_write_then_read(dut):
    """write one word, read back in a separate transaction"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    spi_cs_ni = dut.spi_cs_n_i

    spi_bus = SpiBus.from_entity(
        dut,
        sclk_name="spi_sck_i",
        mosi_name="spi_mosi_i",
        miso_name="spi_miso_o",
        cs_name=None,
    )
    spi_master = SpiMaster(spi_bus, spi_config)
    bfm = SpiboneBFM(dut, clk_i, spi_cs_ni, spi_master)

    await clock_start(clk_i, period_ns=clock_period)
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)

    expected_val = [0x00, 0x00, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF]

    await bfm.write(starting_address=[0x00, 0x00, 0x00, 0x04], payloads=[expected_val])
    recv = (await bfm.read(starting_address=[0x00, 0x00, 0x00, 0x04]))[0]

    print(f"recv is {recv}")

    for i in range(len(expected_val)):
        assert recv[i] == expected_val[i]

    await ClockCycles(clk_i, 100)

    await FallingEdge(clk_i)

@cocotb.test()
async def test_burst_write_then_burst_read(dut):
    """burst write 8 words, burst read them back. Address auto-increments."""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    spi_cs_ni = dut.spi_cs_n_i

    spi_bus = SpiBus.from_entity(
        dut,
        sclk_name="spi_sck_i",
        mosi_name="spi_mosi_i",
        miso_name="spi_miso_o",
        cs_name=None,
    )
    spi_master = SpiMaster(spi_bus, spi_config)
    bfm = SpiboneBFM(dut, clk_i, spi_cs_ni, spi_master)

    await clock_start(clk_i, period_ns=clock_period)
    await reset_sequence(clk_i, rst_i)

    starting_addr = [0x00, 0x00, 0x40, 0x00]
    expected_vals = [[0x00, 0x00, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF],
                     [0x00, 0x00, 0x00, 0x00, 0xCA, 0xFE, 0xBA, 0xBE],
                     [0xAF, 0xAB, 0xEF, 0xEA, 0x00, 0xFF, 0xBF, 0xFF],
                     [0x01, 0x02, 0x03, 0x04, 0x05, 0xAB, 0xCD, 0xEF],
                     [0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD],
                     [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11],
                     [0x00, 0x11, 0x22, 0x33, 0xAA, 0xBB, 0xCC, 0xDD],
                     [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xAC, 0xEF]]

    await bfm.write(starting_address=starting_addr,
                    payloads=expected_vals)

    recv = await bfm.read(starting_address=starting_addr, count=8)

    for i in range(len(expected_vals)):
        for j in range(len(expected_vals[0])):
            assert recv[i][j] == expected_vals[i][j]

    await FallingEdge(clk_i)


@cocotb.test()
async def test_individual_writes_burst_read(dut):
    """Each word written in its own SPI transaction. read back as one burst."""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    spi_cs_ni = dut.spi_cs_n_i

    spi_bus = SpiBus.from_entity(
        dut,
        sclk_name="spi_sck_i",
        mosi_name="spi_mosi_i",
        miso_name="spi_miso_o",
        cs_name=None,
    )
    spi_master = SpiMaster(spi_bus, spi_config)
    bfm = SpiboneBFM(dut, clk_i, spi_cs_ni, spi_master)

    await clock_start(clk_i, period_ns=clock_period)
    await reset_sequence(clk_i, rst_i)

    base = 0x200
    words = [[random.randint(0, 2**8- 1) for _ in range(8)] for _ in range(8)]

    for i, w in enumerate(words):
        await bfm.write(list((base + i).to_bytes(4, byteorder='big')), [w])

    got = await bfm.read(list(base.to_bytes(4, byteorder='big')), len(words))
    for i in range(len(got)):
        for j in range(len(words[0])):
            assert got[i][j] == words[i][j]


@cocotb.test()
async def test_overwrite(dut):
    """second write to the same address replaces the first."""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    spi_cs_ni = dut.spi_cs_n_i

    spi_bus = SpiBus.from_entity(
        dut,
        sclk_name="spi_sck_i",
        mosi_name="spi_mosi_i",
        miso_name="spi_miso_o",
        cs_name=None,
    )
    spi_master = SpiMaster(spi_bus, spi_config)
    bfm = SpiboneBFM(dut, clk_i, spi_cs_ni, spi_master)

    await clock_start(clk_i, period_ns=clock_period)
    await reset_sequence(clk_i, rst_i)

    addr = [0x00, 0x00, 0x00, 0x80]
    await bfm.write(addr, [[0x00, 0x00, 0x00, 0x00, 0xAA, 0xAA, 0xAA, 0xAA]])
    await bfm.write(addr, [[0x00, 0x00, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55]])
    got = await bfm.read(addr, 1)
    expected_val = [0x00, 0x00, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55]

    for i in range(len(expected_val)):
        assert got[0][i] == expected_val[i]


# Runner

tests = [
    "test_single_write_then_read",
    "test_burst_write_then_burst_read",
    "test_individual_writes_burst_read",
    "test_overwrite",
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "spi/spi_slave.sv",
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
