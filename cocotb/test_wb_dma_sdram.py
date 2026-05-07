# DMA master against actual SDRAM controller + Micron SDRAM model. similar to test_wb_dma.py (preload via host, kick DMA, verify via stream or via host read-back) but adapted for the real SDRAM

# kind of mirrors how tpu_soc.sv wires DMA -> mux -> sdram controller -> sdram model

import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge, ClockCycles
from pathlib import Path
from shared import reset_sequence
from runner import run_test
from wb_bfm import WishboneMaster
import random

SYS_CLK_MHZ   = 100
CLK_PERIOD_NS = 1000 // SYS_CLK_MHZ 
INIT_WAIT_US  = 200
INIT_CYCLES   = INIT_WAIT_US * SYS_CLK_MHZ 

# DMA stride is BYTE_W = DataW/8 = 8 (DataW=64). Each beat advances the SDRAM word address by 8, so preload at the same stride
DMA_STRIDE = 8

# Worst case is prob bank precharge + activate + CL + burst + the occasional auto-refresh stall
PER_WORD_CYCLES = 1000


async def init(dut):
    """Start clk, reset, idle all driven inputs, wait out SDRAM init"""
    Clock(dut.clk_i, CLK_PERIOD_NS, units="ns").start()

    host = WishboneMaster(dut, "m0", dut.clk_i)
    host.idle()
    dut.tpu_active.value     = 0
    dut.dma_start.value      = 0
    dut.dma_we.value         = 0
    dut.dma_start_addr.value = 0
    dut.dma_word_count.value = 0
    dut.dma_rd_ready.value   = 0
    dut.dma_wr_data.value    = 0
    dut.dma_wr_valid.value   = 0

    await reset_sequence(dut.clk_i, dut.rst_i)
    await ClockCycles(dut.clk_i, INIT_CYCLES)
    return host


async def preload(dut, host, words, base):
    """Host bulk-write into DRAM at the DMA's stride"""
    dut.tpu_active.value = 0
    for i, w in enumerate(words):
        await host.write(base + i * DMA_STRIDE, w)


async def kick_dma(dut, addr, count, we):
    await FallingEdge(dut.clk_i)
    dut.tpu_active.value     = 1
    dut.dma_start_addr.value = addr
    dut.dma_word_count.value = count
    dut.dma_we.value         = we
    dut.dma_start.value      = 1
    await FallingEdge(dut.clk_i)
    dut.dma_start.value      = 0


async def wait_done(dut, n_words):
    timeout = max(n_words * PER_WORD_CYCLES, PER_WORD_CYCLES)
    for _ in range(timeout):
        if dut.dma_busy.value == 0:
            return
        await RisingEdge(dut.clk_i)
    raise AssertionError(f"DMA did not complete within {timeout} cycles")


def rand_word():
    # 64-bit payload to match the bus
    return random.randint(0, 2**64 - 1)


# read path

@cocotb.test()
async def test_dma_read_basic(dut):
    """Preload n words, DMA reads them back, consumer always ready"""
    host = await init(dut)
    n = 4
    base = 0x40
    expected = [rand_word() for _ in range(n)]
    await preload(dut, host, expected, base)

    await kick_dma(dut, base, n, we=0)
    dut.dma_rd_ready.value = 1

    got = []
    for _ in range(n * PER_WORD_CYCLES):
        await RisingEdge(dut.clk_i)
        if dut.dma_rd_valid.value == 1 and dut.dma_rd_ready.value == 1:
            got.append(dut.dma_rd_data.value.to_unsigned())
        if len(got) == n:
            break

    assert got == expected, f"\nexpected {[hex(x) for x in expected]}\ngot      {[hex(x) for x in got]}"
    await ClockCycles(dut.clk_i, 1)
    assert dut.dma_busy.value == 0, "DMA should be idle after last word"
    await FallingEdge(dut.clk_i)


@cocotb.test()
async def test_dma_read_backpressure(dut):
    """here we have consumer randomly stall. Data integrity needs to hold"""
    host = await init(dut)
    n = 4
    base = 0x100
    expected = [rand_word() for _ in range(n)]
    await preload(dut, host, expected, base)

    await kick_dma(dut, base, n, we=0)

    got = []
    for _ in range(n * PER_WORD_CYCLES):
        await FallingEdge(dut.clk_i)
        dut.dma_rd_ready.value = 1 if random.random() > 0.3 else 0
        await RisingEdge(dut.clk_i)
        if dut.dma_rd_valid.value == 1 and dut.dma_rd_ready.value == 1:
            got.append(dut.dma_rd_data.value.to_unsigned())
        if len(got) == n:
            break

    assert got == expected, f"\nexpected {[hex(x) for x in expected]}\ngot      {[hex(x) for x in got]}"
    await FallingEdge(dut.clk_i)


# write path

@cocotb.test()
async def test_dma_write_basic(dut):
    """DMA writes a stream into SDRAM, host reads back to verify"""
    host = await init(dut)
    n = 4
    base = 0x80
    words = [rand_word() for _ in range(n)]

    await kick_dma(dut, base, n, we=1)

    # present word, wait for ready/valid coincidence, advance.
    idx = 0
    await FallingEdge(dut.clk_i)
    dut.dma_wr_valid.value = 1
    dut.dma_wr_data.value  = words[0]
    for _ in range(n * PER_WORD_CYCLES):
        await RisingEdge(dut.clk_i)
        if dut.dma_wr_valid.value == 1 and dut.dma_wr_ready.value == 1:
            idx += 1
            await FallingEdge(dut.clk_i)
            if idx < n:
                dut.dma_wr_data.value = words[idx]
            else:
                dut.dma_wr_valid.value = 0
                break

    await wait_done(dut, n)

    # hand the bus back, verify each word
    await FallingEdge(dut.clk_i)
    dut.tpu_active.value = 0
    await ClockCycles(dut.clk_i, 2)
    for i, w in enumerate(words):
        got = await host.read(base + i * DMA_STRIDE)
        assert got == w, f"addr {base + i*DMA_STRIDE:#x}: expected {w:#018x}, got {got:#018x}"
    await FallingEdge(dut.clk_i)


@cocotb.test()
async def test_dma_write_producer_stalls(dut):
    """prod holds wr_valid low between words; transfer must still complete."""
    host = await init(dut)
    n = 3
    base = 0x200
    words = [rand_word() for _ in range(n)]

    await kick_dma(dut, base, n, we=1)

    idx = 0
    for _ in range(n * PER_WORD_CYCLES):
        if idx < n and random.random() > 0.4:
            await FallingEdge(dut.clk_i)
            dut.dma_wr_valid.value = 1
            dut.dma_wr_data.value  = words[idx]
        else:
            await FallingEdge(dut.clk_i)
            dut.dma_wr_valid.value = 0
        await RisingEdge(dut.clk_i)
        if dut.dma_wr_valid.value == 1 and dut.dma_wr_ready.value == 1:
            idx += 1
        if idx == n:
            await FallingEdge(dut.clk_i)
            dut.dma_wr_valid.value = 0
            break

    await wait_done(dut, n)

    await FallingEdge(dut.clk_i)
    dut.tpu_active.value = 0
    await ClockCycles(dut.clk_i, 2)
    for i, w in enumerate(words):
        got = await host.read(base + i * DMA_STRIDE)
        assert got == w, f"addr {base + i*DMA_STRIDE:#x}: expected {w:#018x}, got {got:#018x}"
    await FallingEdge(dut.clk_i)


@cocotb.test()
async def test_dma_write_then_read(dut):
    """DMA writes a block, then reads it back. round trip for tpu"""
    await init(dut)
    n = 4
    base = 0x300
    words = [rand_word() for _ in range(n)]

    # write
    await kick_dma(dut, base, n, we=1)
    await FallingEdge(dut.clk_i)
    dut.dma_wr_valid.value = 1
    dut.dma_wr_data.value  = words[0]
    idx = 0
    for _ in range(n * PER_WORD_CYCLES):
        await RisingEdge(dut.clk_i)
        if dut.dma_wr_valid.value == 1 and dut.dma_wr_ready.value == 1:
            idx += 1
            await FallingEdge(dut.clk_i)
            if idx < n:
                dut.dma_wr_data.value = words[idx]
            else:
                dut.dma_wr_valid.value = 0
                break
    await wait_done(dut, n)

    # read
    await kick_dma(dut, base, n, we=0)
    dut.dma_rd_ready.value = 1
    got = []
    for _ in range(n * PER_WORD_CYCLES):
        await RisingEdge(dut.clk_i)
        if dut.dma_rd_valid.value == 1 and dut.dma_rd_ready.value == 1:
            got.append(dut.dma_rd_data.value.to_unsigned())
        if len(got) == n:
            break

    assert got == words, f"\nwrote {[hex(x) for x in words]}\nread  {[hex(x) for x in got]}"
    await FallingEdge(dut.clk_i)


# runner

tests = [
    "test_dma_read_basic",
    "test_dma_read_backpressure",
    "test_dma_write_basic",
    "test_dma_write_producer_stalls",
    "test_dma_write_then_read",
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "common"  / "shift.sv",
    proj_path / "dram" / "wb_dma_master.sv",
    proj_path / "dram" / "wb_mux_2to1.sv",
    proj_path / "dram" / "wb_sdr_mt48lc16m16a2_7e.sv",
    proj_path / "dram" / "sdram_model_mt48lc16m16a2.v",
    proj_path / "dram" / "wb_dma_sdram_tb_top.sv",
]
parameters    = {"sys_clk_mhz_p": SYS_CLK_MHZ}
module_name   = "test_wb_dma_sdram"
hdl_toplevel  = "wb_dma_sdram_tb_top"
sims          = ["icarus"]   # Verilator chokes on the Micron model's `specify` blocks

@pytest.mark.parametrize("testcase", tests)
def test_wb_dma_sdram_each(testcase):
    run_test(parameters=parameters, sources=sources, module_name=module_name,
             hdl_toplevel=hdl_toplevel, testcase=testcase, sims=sims)


def test_wb_dma_sdram_all():
    run_test(parameters=parameters, sources=sources, module_name=module_name,
             hdl_toplevel=hdl_toplevel, sims=sims)