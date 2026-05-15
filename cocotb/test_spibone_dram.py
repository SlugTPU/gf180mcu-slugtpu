import random
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
from cocotbext.spi import SpiMaster, SpiBus, SpiConfig
from shared import reset_sequence, clock_start
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ReadOnly

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

    spi_bus = SpiBus.from_entity(
        dut,
        sclk_name="spi_sck_i",
        mosi_name="spi_mosi_i",
        miso_name="spi_miso_o",
        cs_name="spi_cs_n_i",
    )
    spi_master = SpiMaster(spi_bus, spi_config)

    await clock_start(clk_i, period_ns=clock_period)
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)

    # SINGLE_WRITE, addr->[0x00, 0x00, 0x00, 0x40], data->[0x00, 0x00, 0x00, 0x00, 0xDE,0xAD,0xBE,0xEF]
    await spi_master.write([0x20,
                            0x00, 0x00, 0x00, 0x40,
                            0x00, 0x00, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF],
                           burst=True)
    spi_master.clear()
    await spi_master.write([0x00])
    recv = (await spi_master.read(count=1))[0]
    count = 0
    while (recv == 0xFF and count < 20):
        cocotb.log.info(f"Received {hex(recv)}; waiting...")
        await spi_master.write([0x00])
        recv = (await spi_master.read(count=1))[0]
        count += 1

    if (count > 20):
        assert 1 == 0, "Timed out while waiting for ACK from slave"

    cocotb.log.info(f"Received {hex(recv)}")
    assert recv == 0xAC, f"Expected 0xAC, got {hex(recv)}"

    # SINGLE_READ, addr->[0x00, 0x00, 0x00, 0x40], data->[0x00, 0x00, 0x00, 0x00, 0xDE,0xAD,0xBE,0xEF]
    await spi_master.write([0x10,
                            0x00, 0x00, 0x00, 0x40],
                           burst=True)
    spi_master.clear()
    await spi_master.write([0x00])
    recv = (await spi_master.read(count=1))[0]
    count = 0
    while (recv == 0xFF and count < 20):
        cocotb.log.info(f"Received {hex(recv)}; waiting...")
        await spi_master.write([0x00])
        recv = (await spi_master.read(count=1))[0]
        count += 1
    cocotb.log.info(f"Received {hex(recv)}")
    assert recv == 0xAC, f"Expected 0xAC, got {hex(recv)}"

    await spi_master.write([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], burst=True)
    recv = await spi_master.read(count=8)
    expected_bytes = [0x00, 0x00, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF]
    for i in range(len(expected_bytes)):
        assert recv[i] == expected_bytes[i], f"Expected byte {expected_bytes[i]}, got {recv[i]}"
        cocotb.log.info(f"Received {hex(recv[i])}")

    await FallingEdge(clk_i)

# @cocotb.test()
# async def test_burst_write_then_burst_read(dut):
#     """burst write 8 words, burst read them back. Address auto-increments."""
#     bfm = await setup(dut)

#     base = 0x100
#     words = [random.randint(0, 2**32 - 1) for _ in range(8)]

#     await bfm.write(base, words)
#     got = await bfm.read(base, len(words))
#     assert got == words, f"\nwrote {[hex(w) for w in words]}\nread  {[hex(w) for w in got]}"


# @cocotb.test()
# async def test_individual_writes_burst_read(dut):
#     """Each word written in its own SPI transaction. read back as one burst."""
#     bfm = await setup(dut)

#     base = 0x200
#     words = [random.randint(0, 2**32 - 1) for _ in range(4)]

#     for i, w in enumerate(words):
#         await bfm.write(base + i * 4, [w])

#     got = await bfm.read(base, len(words))
#     assert got == words


# @cocotb.test()
# async def test_overwrite(dut):
#     """second write to the same address replaces the first."""
#     bfm = await setup(dut)

#     addr = 0x80
#     await bfm.write(addr, [0xAAAAAAAA])
#     await bfm.write(addr, [0x55555555])
#     got = await bfm.read(addr, 1)
#     assert got[0] == 0x55555555, f"got 0x{got[0]:08X}"


# Runner

tests = [
    "test_single_write_then_read",
    # "test_burst_write_then_burst_read",
    # "test_individual_writes_burst_read",
    # "test_overwrite",
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
