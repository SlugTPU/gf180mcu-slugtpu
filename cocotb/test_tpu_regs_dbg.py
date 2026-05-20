import pytest
import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles
from pathlib import Path
from runner import run_test
from shared import clock_start, reset_sequence


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


REG_CTRL     = 0x08
REG_DBG_ADDR = 0x0C
REG_DBG_DATA = 0x10


async def _setup(dut):
    m = WbMasterI(dut)
    m.idle()
    dut.tpu_state_i.value = 0b01  # set to idle = keep done detector quiet
    dut.dbg_word_i.value = 0
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
    assert int(dut.tpu_enable_o.value) == 0, "tpu_enable_o spuriously set"

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
    assert (got & 0b11) == 0b10, f"CTRL readback after test_mode-only: got {got:#x}"
    await FallingEdge(dut.clk_i)

def _pack_bytes(byte_list, n_words):
    """Pack a list of byte values into the dbg_word_i bus (LSB first)."""
    v = 0
    for i, b in enumerate(byte_list[:n_words]):
        v |= (b & 0xFF) << (i * 8)
    return v


@cocotb.test()
async def dbg_register_round_trip(dut):
    """set DBG_ADDR, read DBG_DATA; each address returns right byte"""
    m = await _setup(dut)
    n_words = int(dut.DBG_N_WORDS.value)

    pattern = [(i + 1) & 0xFF for i in range(n_words)]
    dut.dbg_word_i.value = _pack_bytes(pattern, n_words)
    await ClockCycles(dut.clk_i, 2)

    for addr in range(n_words):
        await m.write(REG_DBG_ADDR, addr)
        await ClockCycles(dut.clk_i, 1)
        got = await m.read(REG_DBG_DATA)
        assert (got & 0xFF) == pattern[addr], (
            f"DBG_DATA[{addr:#04x}]: got {got & 0xFF:#04x}, want {pattern[addr]:#04x}"
        )
    await FallingEdge(dut.clk_i)


@cocotb.test()
async def dbg_addr_readback(dut):
    """DBG_ADDR is host-readable (confirms the address actually latched)"""
    m = await _setup(dut)
    await m.write(REG_DBG_ADDR, 0x42)
    await ClockCycles(dut.clk_i, 2)
    got = await m.read(REG_DBG_ADDR)
    assert (got & 0xFF) == 0x42, f"DBG_ADDR readback: got {got:#04x}"
    await FallingEdge(dut.clk_i)


@cocotb.test()
async def dbg_oor_returns_sentinel(dut):
    """reading past the populated debug map returns 0xDE"""
    m = await _setup(dut)
    n_words = int(dut.DBG_N_WORDS.value)
    dut.dbg_word_i.value = 0
    await ClockCycles(dut.clk_i, 2)

    await m.write(REG_DBG_ADDR, n_words + 5)
    await ClockCycles(dut.clk_i, 1)
    got = await m.read(REG_DBG_DATA)
    assert (got & 0xFF) == 0xDE, f"OOR sentinel: got {got & 0xFF:#04x}"
    await FallingEdge(dut.clk_i)


proj_path = Path("./src").resolve()
sources = [
    proj_path / "spi" / "tpu_regs.sv",
    proj_path / "debug_mux.sv",
]


def test_tpu_regs_test_mode():
    run_test(
        parameters={},
        sources=sources,
        module_name="test_tpu_regs_dbg",
        hdl_toplevel="tpu_regs",
    )