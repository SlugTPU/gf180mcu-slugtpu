from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge, ReadOnly
import pytest
from runner import run_test
from shared import reset_sequence, clock_start
from cocotbext.spi import SpiMaster, SpiBus, SpiConfig

clock_freq_mhz = 100
spi_config = SpiConfig(
    word_width = 8,     # all parameters optional
    sclk_freq  = (clock_freq_mhz/6) * 10**6,   # these are the defaults
    cpol       = False,
    cpha       = False,
    msb_first  = True,
    cs_active_low = True # optional (assumed True)
)
@cocotb.test()
async def reset_test(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(clk_i)


@cocotb.test()
async def spi_single_byte_test(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    spi_bus = SpiBus.from_entity(
        dut,
        sclk_name="spi_sclk_i",
        mosi_name="spi_mosi_i",
        miso_name="spi_miso_o",
        cs_name="spi_cs_ni",
    )
    spi_master = SpiMaster(spi_bus, spi_config)

    await clock_start(clk_i, period_ns=1/(clock_freq_mhz*10**6)*10**9)
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)
    await spi_master.write([0xAB, 0xCD, 0xDE, 0xAD])
    await spi_master.read(count=3)
    await spi_master.write([0x00, 0x00, 0x00])
    for _ in range(4):
        cocotb.log.info(hex((await spi_master.read(count=1))[0]))

    await FallingEdge(clk_i)

tests = [
    "reset_test",
    "spi_single_byte_test",
]


proj_path = Path("./src").resolve()
sources = [proj_path / "spi" / "spi_slave.sv",
           proj_path / "spi" / "tb_spi_slave.sv"]
hdl_toplevel="tb_spi_slave"
module_name="test_spi"

@pytest.mark.parametrize("testcase", tests)
def test_spi_each(testcase):
    run_test(
        parameters={},
        sources=sources,
        module_name=module_name,
        hdl_toplevel=hdl_toplevel,
        testcase=testcase,
    )


def test_spi_all():
    run_test(
        parameters={},
        sources=sources,
        module_name=module_name,
        hdl_toplevel=hdl_toplevel,
    )
