# host preloads DRAM via m0, TPU DMA r/w via m1 through mux, host verifies via m0.
# test read path, write path, stream backpressure, consecutive kicks.

import pytest
import cocotb
from cocotb.triggers import FallingEdge, RisingEdge, ClockCycles
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
from wb_bfm import WishboneMaster
import random


async def init(dut):
    await clock_start(dut.clk_i)
    host = WishboneMaster(dut, "m0", dut.clk_i)
    host.idle()
    dut.tpu_active.value = 0
    dut.dma_start.value = 0
    dut.dma_we.value = 0
    dut.dma_start_addr.value = 0
    dut.dma_word_count.value = 0
    dut.dma_rd_ready.value = 0
    dut.dma_wr_data.value = 0
    dut.dma_wr_valid.value = 0
    await reset_sequence(dut.clk_i, dut.rst_i)
    return host


async def preload(dut, host, words, base):
    """host bulk write into DRAM"""
    dut.tpu_active.value = 0
    for i, w in enumerate(words):
        await host.write(base + i*4, w)


async def kick_dma(dut, addr, count, we):
    await FallingEdge(dut.clk_i)
    dut.tpu_active.value = 1
    dut.dma_start_addr.value = addr
    dut.dma_word_count.value = count
    dut.dma_we.value = we
    dut.dma_start.value = 1
    await FallingEdge(dut.clk_i)
    dut.dma_start.value = 0


async def wait_done(dut, timeout=2000):
    # done_o is a 1-cycle pulse
    for _ in range(timeout):
        if dut.dma_busy.value == 0:
            return
        await RisingEdge(dut.clk_i)
    raise AssertionError("DMA did not complete in time")


# read path

@cocotb.test()
async def test_dma_read_basic(dut):
    """preload n words, DMA reads them back, consumer always ready"""
    host = await init(dut)
    n = 16
    base = 0x40
    expected = [random.randint(0, 2**32 - 1) for _ in range(n)]
    await preload(dut, host, expected, base)

    await kick_dma(dut, base, n, we=0)
    dut.dma_rd_ready.value = 1

    got = []
    for _ in range(n * 10):  # kind of generous upper bound
        await RisingEdge(dut.clk_i)
        if dut.dma_rd_valid.value == 1 and dut.dma_rd_ready.value == 1:
            got.append(dut.dma_rd_data.value.to_unsigned())
        if len(got) == n:
            break

    assert got == expected, f"\nexpected {expected}\ngot      {got}"
    await ClockCycles(dut.clk_i, 1)
    assert dut.dma_busy.value == 0, "DMA should be idle after last word"
    await FallingEdge(dut.clk_i)


@cocotb.test()
async def test_dma_read_backpressure(dut):
    """if consumer stalls, check data integrity preserved"""
    host = await init(dut)
    n = 12
    base = 0x100
    expected = [random.randint(0, 2**32 - 1) for _ in range(n)]
    await preload(dut, host, expected, base)

    await kick_dma(dut, base, n, we=0)

    got = []
    for _ in range(n * 40):
        await FallingEdge(dut.clk_i)
        dut.dma_rd_ready.value = 1 if random.random() > 0.3 else 0
        await RisingEdge(dut.clk_i)
        if dut.dma_rd_valid.value == 1 and dut.dma_rd_ready.value == 1:
            got.append(dut.dma_rd_data.value.to_unsigned())
        if len(got) == n:
            break

    assert got == expected, f"\nexpected {expected}\ngot      {got}"
    await FallingEdge(dut.clk_i)


# write path

@cocotb.test()
async def test_dma_write_basic(dut):
    """DMA write a stream into DRAM, verify via host read"""
    host = await init(dut)
    n = 8
    base = 0x80
    words = [random.randint(0, 2**32 - 1) for _ in range(n)]

    await kick_dma(dut, base, n, we=1)

    # feed stream: present word, wait for ready/valid cycle, advance
    idx = 0
    await FallingEdge(dut.clk_i)
    dut.dma_wr_valid.value = 1
    dut.dma_wr_data.value  = words[0]
    for _ in range(n * 10):
        await RisingEdge(dut.clk_i)
        if dut.dma_wr_valid.value == 1 and dut.dma_wr_ready.value == 1:
            idx += 1
            await FallingEdge(dut.clk_i)
            if idx < n:
                dut.dma_wr_data.value = words[idx]
            else:
                dut.dma_wr_valid.value = 0
                break

    await wait_done(dut)

    # give bus back to host, read each word
    await FallingEdge(dut.clk_i)
    dut.tpu_active.value = 0
    await ClockCycles(dut.clk_i, 2)
    for i, w in enumerate(words):
        got = await host.read(base + i*4)
        assert got == w, f"addr {base+i*4:#x}: expected {w:#010x}, got {got:#010x}"
    await FallingEdge(dut.clk_i)


@cocotb.test()
async def test_dma_write_producer_stalls(dut):
    """prod holds wr_valid low between words check if transfer still completes"""
    host = await init(dut)
    n = 6
    base = 0x200
    words = [random.randint(0, 2**32 - 1) for _ in range(n)]

    await kick_dma(dut, base, n, we=1)

    idx = 0
    for _ in range(n * 40):
        # randomly present data
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

    await wait_done(dut)

    await FallingEdge(dut.clk_i)
    dut.tpu_active.value = 0
    await ClockCycles(dut.clk_i, 2)
    for i, w in enumerate(words):
        got = await host.read(base + i*4)
        assert got == w, f"addr {base+i*4:#x}: expected {w:#010x}, got {got:#010x}"
    await FallingEdge(dut.clk_i)


@cocotb.test()
async def test_dma_write_then_read(dut):
    """DMA writes a block, then reads it back. TPU-side round trip."""
    host = await init(dut)
    n = 10
    base = 0x300
    words = [random.randint(0, 2**32 - 1) for _ in range(n)]

    # write
    await kick_dma(dut, base, n, we=1)
    await FallingEdge(dut.clk_i)
    dut.dma_wr_valid.value = 1
    dut.dma_wr_data.value  = words[0]
    idx = 0
    for _ in range(n * 10):
        await RisingEdge(dut.clk_i)
        if dut.dma_wr_valid.value == 1 and dut.dma_wr_ready.value == 1:
            idx += 1
            await FallingEdge(dut.clk_i)
            if idx < n:
                dut.dma_wr_data.value = words[idx]
            else:
                dut.dma_wr_valid.value = 0
                break
    await wait_done(dut)

    # read
    await kick_dma(dut, base, n, we=0)
    dut.dma_rd_ready.value = 1
    got = []
    for _ in range(n * 10):
        await RisingEdge(dut.clk_i)
        if dut.dma_rd_valid.value == 1 and dut.dma_rd_ready.value == 1:
            got.append(dut.dma_rd_data.value.to_unsigned())
        if len(got) == n:
            break

    assert got == words, f"\nwrote {words}\nread  {got}"
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
    proj_path / "dram/wb_mux_2to1.sv",
    proj_path / "dram/wb_mem_model.sv",
    proj_path / "dram/wb_dma_master.sv",
    proj_path / "dram/wb_dma_tb_top.sv",
]

@pytest.mark.parametrize("testcase", tests)
def test_wb_dma_each(testcase):
    run_test(
        parameters={},
        sources=sources,
        module_name="test_wb_dma",
        hdl_toplevel="wb_dma_tb_top",
        testcase=testcase,
        sims=["icarus"],
    )

def test_wb_dma_all():
    run_test(
        parameters={},
        sources=sources,
        module_name="test_wb_dma",
        hdl_toplevel="wb_dma_tb_top",
        sims=["icarus"],
    )