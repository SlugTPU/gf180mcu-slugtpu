import pytest
import cocotb
from cocotb.triggers import RisingEdge, FallingEdge
from pathlib import Path
from runner import run_test
from shared import clock_start, reset_sequence


# WB master driver matching tpu_regs-style _i/_o suffixes
class WbMasterI:
    def __init__(self, dut):
        self.dut = dut
        self._sel_all = (1 << len(dut.wb_sel_i)) - 1

    def idle(self):
        self.dut.wb_adr_i.value = 0
        self.dut.wb_dat_i.value = 0
        self.dut.wb_sel_i.value = 0
        self.dut.wb_we_i.value  = 0
        self.dut.wb_stb_i.value = 0
        self.dut.wb_cyc_i.value = 0

    async def write(self, addr, data, sel=None):
        await FallingEdge(self.dut.clk_i)
        self.dut.wb_adr_i.value = addr
        self.dut.wb_dat_i.value = data
        self.dut.wb_sel_i.value = self._sel_all if sel is None else sel
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

    async def read(self, addr, sel=None):
        await FallingEdge(self.dut.clk_i)
        self.dut.wb_adr_i.value = addr
        self.dut.wb_sel_i.value = self._sel_all if sel is None else sel
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


async def _setup(dut):
    m = WbMasterI(dut)
    m.idle()
    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)
    return m


@cocotb.test()
async def round_trip(dut):
    """write each word and read back"""
    m = await _setup(dut)
    n_words = int(dut.N_WORDS.value)
    pattern = [(0xCAFE0000 | i) for i in range(n_words)]
    for i, v in enumerate(pattern):
        await m.write(i * 4, v)
    for i, v in enumerate(pattern):
        got = await m.read(i * 4)
        assert got == v, f"word {i}: got {got:#010x}, want {v:#010x}"
    await FallingEdge(dut.clk_i)


@cocotb.test()
async def byte_strobe(dut):
    """partial writes only modify the strobed bytes"""
    m = await _setup(dut)
    await m.write(0, 0xAABBCCDD, sel=0b1111)
    # Overwrite only bytes 1 and 3 (sel=0b1010).
    await m.write(0, 0x11223344, sel=0b1010)
    got = await m.read(0)
    assert got == 0x11BB33DD, f"byte-strobed write wrong: got {got:#010x}"
    await FallingEdge(dut.clk_i)

@cocotb.test()
async def address_alignment(dut):
    """byte offset dropped. adr=0 and adr=3 both hit word 0"""
    m = await _setup(dut)
    await m.write(0, 0xDEADBEEF)
    await m.write(3, 0x12345678) # same word and overwrites
    got = await m.read(2) # still same
    assert got == 0x12345678, f"alignment wrong: got {got:#010x}"
    await FallingEdge(dut.clk_i)


@cocotb.test()
async def boundary(dut):
    """highest word is addressable"""
    m = await _setup(dut)
    n_words = int(dut.N_WORDS.value)
    top = (n_words - 1) * 4
    await m.write(top, 0xFEEDFACE)
    got = await m.read(top)
    assert got == 0xFEEDFACE, f"boundary word: got {got:#010x}"
    await FallingEdge(dut.clk_i)


proj_path = Path("./src").resolve()
sources = [proj_path / "spi" / "wb_test_ram.sv"]


@pytest.mark.parametrize("n_words", [16, 128])
def test_wb_test_ram(n_words):
    run_test(
        parameters={"N_WORDS": n_words},
        sources=sources,
        module_name="test_wb_test_ram",
        hdl_toplevel="wb_test_ram",
    )