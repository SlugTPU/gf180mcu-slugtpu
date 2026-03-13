# sim/test_spi.py
from __future__ import annotations

import random
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, ReadOnly

import pytest
from runner import run_test


#* Drive helper
#* Ensures signal updates occur outside ReadOnly phase
async def drive(sig, val):
    sig.value = int(val)
    await Timer(1, unit="ps")


def u8(sig) -> int:
    return int(sig.value) & 0xFF


#* Wait for N clk cycles
async def settle_clk(dut, n=2):
    for _ in range(n):
        await RisingEdge(dut.clk)


#* Initialize DUT and apply reset
async def init_dut(dut, clk_period_ns=10):
    cocotb.start_soon(Clock(dut.clk, clk_period_ns, unit="ns").start())

    await drive(dut.cs_n, 1)
    await drive(dut.sclk, 0)
    await drive(dut.mosi, 0)
    await drive(dut.tx_valid, 0)
    await drive(dut.tx_data, 0)

    await drive(dut.rst, 1)
    await settle_clk(dut, 5)
    await drive(dut.rst, 0)
    await settle_clk(dut, 5)


#* Queue next transmit byte into slave
async def queue_tx_byte(dut, b):
    await drive(dut.tx_data, b)
    await drive(dut.tx_valid, 1)
    await settle_clk(dut, 2)
    await drive(dut.tx_valid, 0)


#* SPI Mode 0 transfer
#* MOSI valid while SCLK low
#* MISO sampled on rising edge
async def spi_transfer_byte(dut, mosi_byte):
    miso = 0

    await drive(dut.cs_n, 0)
    await settle_clk(dut, 3)

    for bit in range(8):
        bit_val = (mosi_byte >> (7 - bit)) & 1

        await drive(dut.mosi, bit_val)
        await settle_clk(dut, 1)

        await drive(dut.sclk, 1)
        await settle_clk(dut, 1)

        await ReadOnly()
        miso = ((miso << 1) | (int(dut.miso.value) & 1)) & 0xFF
        await Timer(1, unit="ps")

        await drive(dut.sclk, 0)
        await settle_clk(dut, 1)

    await drive(dut.cs_n, 1)
    await settle_clk(dut, 3)

    return miso


#* Wait until slave reports received byte
async def wait_for_rx(dut, timeout_cycles=200):
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        await ReadOnly()
        if dut.rx_valid.value:
            await Timer(1, unit="ps")
            return u8(dut.rx_data)

    raise AssertionError("Timeout waiting for rx_valid")


#* Cocotb tests

@cocotb.test()
async def reset_test(dut):
    await init_dut(dut)
    await ReadOnly()
    assert dut.rx_valid.value == 0


@cocotb.test()
async def spi_single_byte_test(dut):
    await init_dut(dut)

    val = 0xA5
    _ = await spi_transfer_byte(dut, val)
    rx = await wait_for_rx(dut)

    assert rx == val, f"Expected 0x{val:02X}, got 0x{rx:02X}"


@cocotb.test()
async def spi_multiple_bytes_test(dut):
    await init_dut(dut)

    data = [random.randrange(0, 256) for _ in range(8)]
    for b in data:
        _ = await spi_transfer_byte(dut, b)
        rx = await wait_for_rx(dut)
        assert rx == b, f"Expected 0x{b:02X}, got 0x{rx:02X}"


@cocotb.test()
async def spi_tx_next_byte_test(dut):
    await init_dut(dut)

    tx0 = 0x3C
    tx1 = 0xC3

    await queue_tx_byte(dut, tx0)
    miso0 = await spi_transfer_byte(dut, 0x00)

    await queue_tx_byte(dut, tx1)
    miso1 = await spi_transfer_byte(dut, 0x00)

    assert miso0 == tx0, f"MISO0 expected 0x{tx0:02X}, got 0x{miso0:02X}"
    assert miso1 == tx1, f"MISO1 expected 0x{tx1:02X}, got 0x{miso1:02X}"


#* Pytest wrappers (runner.py integration)

tests = [
    "reset_test",
    "spi_single_byte_test",
    "spi_multiple_bytes_test",
    "spi_tx_next_byte_test",
]


@pytest.mark.parametrize("testcase", tests)
def test_spi_each(testcase):
    sources = [Path("./rtl/spi_slave.sv").resolve()]
    try:
        run_test(
            parameters={},
            sources=sources,
            module_name="test_spi",
            hdl_toplevel="spi_slave",
            testcase=testcase,
        )
    except SystemExit as exc:
        if exc.code != 0:
            pytest.fail(f"cocotb test '{testcase}' failed (exit code {exc.code})")


def test_spi_all():
    sources = [Path("./rtl/spi_slave.sv").resolve()]
    try:
        run_test(
            parameters={},
            sources=sources,
            module_name="test_spi",
            hdl_toplevel="spi_slave",
        )
    except SystemExit as exc:
        if exc.code != 0:
            pytest.fail(f"cocotb tests failed (exit code {exc.code})")
