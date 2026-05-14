import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, Timer, RisingEdge
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
from spibone_master import SpiboneMaster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def spibone_write(spi: SpiboneMaster, addr: int, data: int):
    await spi.write(addr, data)


async def spibone_read(spi: SpiboneMaster, addr: int) -> int:
    return await spi.read(addr)


async def spibone_burst_read(spi: SpiboneMaster, addr: int, n_words: int) -> list:
    return await spi.burst_read(addr, n_words)


async def init(dut) -> SpiboneMaster:
    await clock_start(dut.clk_i, period_ns=10)   # 100 MHz
    await reset_sequence(dut.clk_i, dut.rst_i)
    return SpiboneMaster(dut)


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
    """Write a distinctive 64-bit pattern to DRAM word 0, read back to verify,
    then overwrite with a second pattern and verify the update.
    Uses values that CTRL register [1:0] cannot reproduce, so a mis-route to
    the register file would be caught immediately.
    """
    clk_i = dut.clk_i
    spi = await init(dut)

    # Wait for SDRAM initialisation (~100 µs × 100 MHz = 10000 cycles)

    pattern_a = 0xDEAD_BEEF_CAFE_1234
    pattern_b = 0x0123_4567_89AB_CDEF
    dram_addr = 0x0000_0000

    await spibone_write(spi, dram_addr, pattern_a)
    val = await spibone_read(spi, dram_addr)
    cocotb.log.info(f"got {hex(val)}, expected {hex(pattern_a)}")
    assert val == pattern_a, \
        f"DRAM readback A: got {val:#018x}, expected {pattern_a:#018x}"

    await spibone_write(spi, dram_addr, pattern_b)
    val = await spibone_read(spi, dram_addr)
    cocotb.log.info(f"got {hex(val)}, expected {hex(pattern_b)}")
    assert val == pattern_b, \
        f"DRAM readback B: got {val:#018x}, expected {pattern_b:#018x}"

    await FallingEdge(clk_i)


@cocotb.test()
async def test_multi_addr(dut):
    """Write distinctive patterns to several consecutive DRAM word addresses,
    then read them all back to verify each address is independent.
    Word size = 64 bits = 8 bytes, so byte addresses step by 8.
    """
    spi = await init(dut)
    await ClockCycles(dut.clk_i, 11000)

    addrs   = [0x00, 0x08, 0x10, 0x18, 0x20]
    patterns = [0xAAAA_AAAA_AAAA_AAAA,
                0x5555_5555_5555_5555,
                0xDEAD_BEEF_CAFE_1234,
                0x0000_0000_0000_0001,
                0xFFFF_FFFF_FFFF_FFFF]

    for addr, pat in zip(addrs, patterns):
        await spibone_write(spi, addr, pat)

    for addr, pat in zip(addrs, patterns):
        val = await spibone_read(spi, addr)
        cocotb.log.info(f"addr={addr:#010x} got {val:#018x}, expected {pat:#018x}")
        assert val == pat, \
            f"addr {addr:#010x}: got {val:#018x}, expected {pat:#018x}"

    await FallingEdge(dut.clk_i)


@cocotb.test()
async def test_burst_read(dut):
    """Write 4 words individually, then burst-read all 4 back in one SPI
    transaction with CS held asserted throughout.  Verifies that S_TX_BURST
    correctly auto-increments the address and the READY+data sequence appears
    for each word.  Burst mode is enabled/disabled around the burst read.

    Note: burst writes (multiple words in one CS assertion) are not yet
    implemented in spibone_wb; the write phase uses single-word transactions.
    """
    spi = await init(dut)
    await ClockCycles(dut.clk_i, 11000)

    n_words  = 4
    base_addr = 0x0000_0000
    patterns = [
        0xAAAA_AAAA_AAAA_AAAA,
        0x5555_5555_5555_5555,
        0xDEAD_BEEF_CAFE_1234,
        0x0123_4567_89AB_CDEF,
    ]
    addrs = [base_addr + i * 8 for i in range(n_words)]

    for addr, pat in zip(addrs, patterns):
        await spibone_write(spi, addr, pat)

    got = await spibone_burst_read(spi, base_addr, n_words)

    for i, (pat, val) in enumerate(zip(patterns, got)):
        cocotb.log.info(
            f"word {i} addr={addrs[i]:#010x}: got {val:#018x} expected {pat:#018x}"
        )
        assert val == pat, \
            f"burst word {i} (addr {addrs[i]:#010x}): got {val:#018x}, expected {pat:#018x}"

    await FallingEdge(dut.clk_i)


@cocotb.test()
async def test_burst_then_single(dut):
    """Burst-read 4 words, then immediately write and single-read a DRAM word
    without disabling burst_en between the burst and the single transactions.
    Verifies that cs_deassert cleanly resets the FSM (S_TX_BURST → S_IDLE) so
    any subsequent transaction type works correctly.
    """
    spi = await init(dut)
    await ClockCycles(dut.clk_i, 11000)

    n_words   = 4
    base_addr = 0x0000_0000
    patterns  = [
        0xAAAA_AAAA_AAAA_AAAA,
        0x5555_5555_5555_5555,
        0xDEAD_BEEF_CAFE_1234,
        0x0123_4567_89AB_CDEF,
    ]
    new_val   = 0x1111_2222_3333_4444

    for i, pat in enumerate(patterns):
        await spibone_write(spi, base_addr + i * 8, pat)

    got = await spibone_burst_read(spi, base_addr, n_words)
    for i, (pat, val) in enumerate(zip(patterns, got)):
        assert val == pat, \
            f"burst word {i}: got {val:#018x}, expected {pat:#018x}"

    await spibone_write(spi, base_addr, new_val)
    val = await spibone_read(spi, base_addr)
    assert val == new_val, \
        f"single read after burst: got {val:#018x}, expected {new_val:#018x}"

    # verify adjacent word unchanged
    val = await spibone_read(spi, base_addr + 8)
    assert val == patterns[1], \
        f"word 1 after write: got {val:#018x}, expected {patterns[1]:#018x}"

    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_tpu_load_pc(dut):
    """
    Load in PC and see what happens
    """
    spi = await init(dut)

    await spibone_write(spi, 0x1000_0000, 0x0000_0000_0000_0003)

    await ClockCycles(dut.clk_i, 1000)

    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_tpu_mem_basic(dut):
    """
    Test SPI -> DRAM, SPI PC -> TPU, TPU <-> DRAM
    """
    spi = await init(dut)
    n_words   = 4
    base_addr = 0x0000_0000
    patterns  = [0xFFFF_FFFF_FFFF_FFFF - i for i in range (33)]

    for i, pat in enumerate(patterns):
        await spibone_write(spi, base_addr + i * 8, pat)

    await spibone_write(spi, 0x1000_0000, 0x0000_0000_0000_0001)

    await ClockCycles(dut.clk_i, 1000)

    await FallingEdge(dut.clk_i)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

tests = [
    'test_reset',
    'test_simple',
    'test_multi_addr',
    'test_burst_read',
    'test_burst_then_single',
    'test_tpu_load_pc',
    'test_tpu_mem_basic',
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "tpu_soc.sv",
    proj_path / "spi" / "spibone_wb.sv",
    proj_path / "spi" / "wb_decoder.sv",
    proj_path / "spi" / "tpu_regs.sv",
    proj_path / "dram" / "tpu_soc_with_dram.sv",
    proj_path / "dram" / "wb_dma_master.sv",
    proj_path / "dram" / "wb_mux_2to1.sv",
    proj_path / "dram" / "wb_sdr_mt48lc16m16a2_7e.sv",
    proj_path / "dram" / "sdram_model_mt48lc16m16a2.v",
    proj_path / "common" / "shift.sv",
    proj_path / "common" / "counter.sv",
    proj_path / "common" / "elastic.sv",
    proj_path / "control" / "control_top.sv",
    proj_path / "control" / "control_decoder.sv",
    proj_path / "control" / "control_buffer.sv",
    proj_path / "control" / "control_sram.sv",
    proj_path / "compute_core.sv",
    proj_path / "sram" / "sram_1x256.sv",
    proj_path / "sram" / "sram_8x256.sv",
    proj_path / "sram" / "sram_8x256_full.sv",
    proj_path / "sram" / "memory_transaction.sv",
    proj_path / "sysray" / "mxu.sv",
    proj_path / "sysray" / "sysray_nxn.sv",
    proj_path / "sysray" / "pe.sv",
    proj_path / "scalar_units" / "add_n.sv",
    proj_path / "scalar_units" / "relu_n.sv",
    proj_path / "scalar_units" / "scale_n.sv",
    proj_path / "scalar_units" / "load_data.sv",
    proj_path / "scalar_units" / "scalar_pipe.sv",
    proj_path / "scalar_units" / "scalar_stage.sv",
    proj_path / "scalar_units" / "quantizer_mul.sv",
    proj_path / "scalar_units" / "scalar_stage_sram.sv",
    proj_path / "tri_shift.sv",
    "ip/gf180mcu_ocd_ip_sram/cells/gf180mcu_ocd_ip_sram__sram256x8m8wm1/gf180mcu_ocd_ip_sram__sram256x8m8wm1.v"
]
parameters = {"sys_clk_mhz_p": 100}
module_name = "test_tpu_soc_with_dram"
hdl_toplevel = "tpu_soc_sdram_tb"
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
        sim_flag=2,
    )


# def test_tpu_soc_all():
#     """Runs all tests sequentially in one simulation."""
#     run_test(
#         parameters=parameters,
#         sources=sources,
#         module_name=module_name,
#         hdl_toplevel=hdl_toplevel,
#         sims=sims,
#         sim_flag=2,
#     )
