import pytest
import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles
from pathlib import Path
from runner import run_test
from shared import clock_start, reset_sequence


# WB driver matching tpu_regs's _i/_o suffix style
class WbMasterI:
    def __init__(self, dut):
        self.dut = dut

    def idle(self):
        self.dut.wb_adr_i.value = 0
        self.dut.wb_dat_i.value = 0
        self.dut.wb_we_i.value  = 0
        self.dut.wb_stb_i.value = 0
        self.dut.wb_cyc_i.value = 0

    async def write(self, addr, data):
        await FallingEdge(self.dut.clk_i)
        self.dut.wb_adr_i.value = addr
        self.dut.wb_dat_i.value = data
        self.dut.wb_we_i.value  = 1
        self.dut.wb_stb_i.value = 1
        self.dut.wb_cyc_i.value = 1
        while True:
            await RisingEdge(self.dut.clk_i)
            if int(self.dut.wb_ack_o.value) == 1:
                break
        await FallingEdge(self.dut.clk_i)
        self.dut.wb_stb_i.value = 0
        self.dut.wb_cyc_i.value = 0
        self.dut.wb_we_i.value  = 0

    async def read(self, addr):
        await FallingEdge(self.dut.clk_i)
        self.dut.wb_adr_i.value = addr
        self.dut.wb_we_i.value  = 0
        self.dut.wb_stb_i.value = 1
        self.dut.wb_cyc_i.value = 1
        while True:
            await RisingEdge(self.dut.clk_i)
            if int(self.dut.wb_ack_o.value) == 1:
                data = int(self.dut.wb_dat_o.value)
                break
        await FallingEdge(self.dut.clk_i)
        self.dut.wb_stb_i.value = 0
        self.dut.wb_cyc_i.value = 0
        return data


REG_CTRL = 0x08


async def _setup(dut):
    m = WbMasterI(dut)
    m.idle()
    dut.tpu_state_i.value = 0b01  # set to idle = keep done detector quiet
    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)
    return m


@cocotb.test()
async def test_mode_set_and_clear(dut):
    """CTRL[1] toggles test_mode_o; CTRL[0] continues to drive tpu_enable_o."""
    m = await _setup(dut)

    # reset: both bits low
    assert int(dut.test_mode_o.value)  == 0
    assert int(dut.tpu_enable_o.value) == 0

    # set test_mode alone
    await m.write(REG_CTRL, 0b10)
    await ClockCycles(dut.clk_i, 2)
    assert int(dut.test_mode_o.value)  == 1, "test_mode_o did not assert"
    assert int(dut.tpu_enable_o.value) == 0, "tpu_enable_o false set"

    # set both
    await m.write(REG_CTRL, 0b11)
    await ClockCycles(dut.clk_i, 2)
    assert int(dut.test_mode_o.value)  == 1
    assert int(dut.tpu_enable_o.value) == 1

    # clear both
    await m.write(REG_CTRL, 0b00)
    await ClockCycles(dut.clk_i, 2)
    assert int(dut.test_mode_o.value)  == 0, "test_mode_o did not clear"
    assert int(dut.tpu_enable_o.value) == 0
    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_mode_readback(dut):
    """CTRL readback exposes both bits in their declared positions"""
    m = await _setup(dut)
    await m.write(REG_CTRL, 0b11)
    await ClockCycles(dut.clk_i, 2)
    got = await m.read(REG_CTRL)
    assert (got & 0b11) == 0b11, f"CTRL readback: got {got:#x}"

    await m.write(REG_CTRL, 0b10)
    await ClockCycles(dut.clk_i, 2)
    got = await m.read(REG_CTRL)
    # bit 1 = test_mode, bit 0 = tpu_enable
    assert (got & 0b11) == 0b10, f"CTRL readback after test_mode-only: got {got:#x}"
    await FallingEdge(dut.clk_i)

proj_path = Path("./src").resolve()
sources = [proj_path / "spi" / "tpu_regs.sv"]


def test_tpu_regs_test_mode():
    run_test(
        parameters={},
        sources=sources,
        module_name="test_tpu_regs_test_mode",
        hdl_toplevel="tpu_regs",
    )