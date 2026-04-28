import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, Timer, RisingEdge, First
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
from cocotbext.spi import SpiMaster, SpiBus, SpiConfig
import random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_spi(dut):
    bus = SpiBus(
        entity=dut,
        prefix="spi",
        sclk_name="clk_i",
        mosi_name="mosi_i",
        miso_name="miso_o",
        cs_name="cs_ni",
    )
    cfg = SpiConfig(
        word_width=(1+4+8)*8,   # cmd(1) + addr(4) + data/dummy(8) = 13 bytes
        sclk_freq=20_000_000,   # 1 MHz; clk_i=100 MHz => 100x >= 4x minimum
        cpol=False,
        cpha=False,
        msb_first=True,
        cs_active_low=True,
        frame_spacing_ns=30,    # >= 3 sys_clk cycles so cs_deassert propagates
    )
    return SpiMaster(bus, cfg)


async def spibone_write(spi, addr: int, data: int):
    """Send a single 64-bit spibone write: cmd(1) + addr(4) + data(8) bytes."""
    payload = bytes([0x00]) + addr.to_bytes(4, 'big') + data.to_bytes(8, 'big')
    await spi.write([int.from_bytes(payload, 'big')])
    await spi.read(1)  # drain RX queue; SPI is full-duplex so write also captures MISO


async def spibone_read(spi, addr: int) -> int:
    """Send a single 64-bit spibone read; return the 64-bit response."""
    payload = bytes([0x01]) + addr.to_bytes(4, 'big') + bytes(8)
    await spi.write([int.from_bytes(payload, 'big')])
    result = await spi.read(1)
    return result[0] & 0xFFFF_FFFF_FFFF_FFFF


async def init(dut):
    dut.sdr_dq_i.value = 0          # SDRAM data bus not used in these tests
    await clock_start(dut.clk_i, period_ns=10)   # 100 MHz
    await reset_sequence(dut.clk_i, dut.rst_i)
    return make_spi(dut)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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
async def test_simple(dut):
    """Write CTRL=1 (tpu_enable), read back to verify it latches,
    then clear it and verify it reads 0.
    """
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()
    spi = await init(dut)

    await spibone_write(spi, 0x1000_0000, 0x1)
    val = await spibone_read(spi, 0x1000_0000)
    cocotb.log.info(f"got {hex(val)}, expected 0x00000001")
    assert (val & 0xFFFF_FFFF) == 0x1, \
        f"CTRL readback after set: got {val:#010x}, expected 0x00000001"

    await spibone_write(spi, 0x1000_0000, 0x0)
    val = await spibone_read(spi, 0x1000_0000)
    cocotb.log.info(f"got {hex(val)}, expected 0x00000000")
    assert (val & 0xFFFF_FFFF) == 0x0, \
        f"CTRL readback after clear: got {val:#010x}, expected 0x00000000"

    await FallingEdge(clk_i)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

tests = [
    'test_reset',
    'test_simple',
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "tpu_soc.sv",
    proj_path / "spi" / "spibone_wb.sv",
    proj_path / "spi" / "wb_decoder.sv",
    proj_path / "spi" / "tpu_regs.sv",
    proj_path / "dram" / "wb_dma_master.sv",
    proj_path / "dram" / "wb_mux_2to1.sv",
    proj_path / "dram" / "wb_sdr_mt48lc16m16a2_7e.sv",
    proj_path / "common" / "shift.sv",
]
parameters = {"sys_clk_mhz_p": 100}
module_name = "test_tpu_soc"
hdl_toplevel = "tpu_soc"
sims = ["icarus"]
# note: verilator doesn't like specify blocks in the SDRAM controller


@pytest.mark.parametrize("testcase", tests)
def test_tpu_soc_each(testcase):
    """Runs each test independently. Continues on test failure."""
    run_test(
        parameters=parameters,
        sources=sources,
        module_name=module_name,
        hdl_toplevel=hdl_toplevel,
        testcase=testcase,
        sims=sims,
    )


def test_tpu_soc_all():
    """Runs all tests sequentially in one simulation."""
    run_test(
        parameters=parameters,
        sources=sources,
        module_name=module_name,
        hdl_toplevel=hdl_toplevel,
        sims=sims,
    )
