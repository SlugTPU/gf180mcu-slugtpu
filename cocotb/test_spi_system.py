import pytest
import cocotb
from cocotb.triggers import RisingEdge, Timer, ClockCycles
from cocotb.clock import Clock
from pathlib import Path
from runner import run_test
import random

# ---------------------------------------------------------------------------
# Timing constants
# System clock: 100 MHz (10 ns period)
# SPI SCK:      2 MHz (250 ns half-period)
# clk_i must be >= 4x SCK — satisfied here (100 MHz >> 4 * 12.5 MHz)
# ---------------------------------------------------------------------------
CLK_PERIOD_NS = 10
SCK_HALF_NS   = 250
SCK_HALF_CYCLES = SCK_HALF_NS // CLK_PERIOD_NS

# Address map (must match wb_decoder.sv)
DRAM_BASE       = 0x00000000
TPU_CTRL        = 0x10000000
TPU_STATUS      = 0x10000004
TPU_INPUT_ADDR  = 0x10000008
TPU_OUTPUT_ADDR = 0x1000000C
TPU_LENGTH      = 0x10000010
UNMAPPED_ADDR   = 0xF0000000

STATUS_BUSY = 0x1
STATUS_DONE = 0x2


# ---------------------------------------------------------------------------
# AXI4 slave model
# Runs as a background coroutine. Stores writes in mem dict, serves reads.
# ---------------------------------------------------------------------------
class AXISlaveModel:
    def __init__(self, dut):
        self.dut = dut
        self.mem = {}

    async def run(self):
        dut = self.dut
        dut.axi_awready.value = 0
        dut.axi_wready.value  = 0
        dut.axi_bvalid.value  = 0
        dut.axi_bresp.value   = 0
        dut.axi_arready.value = 0
        dut.axi_rvalid.value  = 0
        dut.axi_rdata.value   = 0
        dut.axi_rlast.value   = 0
        dut.axi_rresp.value   = 0

        while True:
            await RisingEdge(dut.clk_i)

            # ---- Write path ----
            aw_valid = int(dut.axi_awvalid.value)
            w_valid  = int(dut.axi_wvalid.value)

            if aw_valid and w_valid:
                dut.axi_awready.value = 1
                dut.axi_wready.value  = 1
                addr = int(dut.axi_awaddr.value)
                data = int(dut.axi_wdata.value)
                self.mem[addr] = data
                cocotb.log.debug(
                    f"  [AXI] WRITE addr=0x{addr:08X} data=0x{data:08X}"
                )
                await RisingEdge(dut.clk_i)
                dut.axi_awready.value = 0
                dut.axi_wready.value  = 0
                dut.axi_bvalid.value  = 1
                dut.axi_bresp.value   = 0
                await RisingEdge(dut.clk_i)
                dut.axi_bvalid.value  = 0

            elif aw_valid and not w_valid:
                dut.axi_awready.value = 1
                addr = int(dut.axi_awaddr.value)
                await RisingEdge(dut.clk_i)
                dut.axi_awready.value = 0
                while not int(dut.axi_wvalid.value):
                    await RisingEdge(dut.clk_i)
                dut.axi_wready.value = 1
                data = int(dut.axi_wdata.value)
                self.mem[addr] = data
                await RisingEdge(dut.clk_i)
                dut.axi_wready.value  = 0
                dut.axi_bvalid.value  = 1
                dut.axi_bresp.value   = 0
                await RisingEdge(dut.clk_i)
                dut.axi_bvalid.value  = 0

            # ---- Read path ----
            if int(dut.axi_arvalid.value):
                dut.axi_arready.value = 1
                ar_addr = int(dut.axi_araddr.value)
                await RisingEdge(dut.clk_i)
                dut.axi_arready.value = 0
                rdata = self.mem.get(ar_addr, 0xDEADBEEF)
                dut.axi_rvalid.value  = 1
                dut.axi_rdata.value   = rdata
                dut.axi_rlast.value   = 1
                dut.axi_rresp.value   = 0
                cocotb.log.debug(
                    f"  [AXI] READ  addr=0x{ar_addr:08X} data=0x{rdata:08X}"
                )
                await RisingEdge(dut.clk_i)
                dut.axi_rvalid.value = 0
                dut.axi_rlast.value  = 0


# ---------------------------------------------------------------------------
# SPI helpers
# ---------------------------------------------------------------------------

async def _spi_byte(dut, tx: int) -> int:
    """Shift one byte MSB-first, synchronized to clk_i.

    spibone_wb samples MOSI through a registered path (mosi_r) on detected
    SCK rising edges. Driving with absolute Timer delays can place SCK/MOSI
    transitions near clk_i edges and cause bit slips. This helper keeps each
    SPI edge aligned to integer clk_i cycles so sampled bits are deterministic.
    """
    rx = 0
    for bit in range(7, -1, -1):
        dut.spi_mosi_i.value = (tx >> bit) & 1
        # Give mosi_r at least one full clk edge to capture this bit before SCK rise.
        await ClockCycles(dut.clk_i, SCK_HALF_CYCLES)
        # Sample MISO at end of low phase, right before clocking next edge.
        # With the DUT's registered edge-detect pipeline this avoids grabbing
        # the already-shifted next bit (which appears as a left-shifted word).
        await Timer(1, unit="ps")
        rx = (rx << 1) | int(dut.spi_miso_o.value)
        dut.spi_sck_i.value = 1
        await ClockCycles(dut.clk_i, SCK_HALF_CYCLES)
        dut.spi_sck_i.value = 0
        await Timer(1, unit="ps")
    return rx


async def spi_write(dut, addr: int, words: list) -> None:
    """
    spibone burst write.
    Protocol: 0x00 | addr[31:0] | word[0] | word[1] | ...
    CS held low for the entire transfer (burst mode).
    """
    dut.spi_cs_n_i.value = 0
    await Timer(SCK_HALF_NS, unit="ns")

    await _spi_byte(dut, 0x00)
    await _spi_byte(dut, (addr >> 24) & 0xFF)
    await _spi_byte(dut, (addr >> 16) & 0xFF)
    await _spi_byte(dut, (addr >>  8) & 0xFF)
    await _spi_byte(dut, (addr      ) & 0xFF)

    for word in words:
        await _spi_byte(dut, (word >> 24) & 0xFF)
        await _spi_byte(dut, (word >> 16) & 0xFF)
        await _spi_byte(dut, (word >>  8) & 0xFF)
        await _spi_byte(dut, (word      ) & 0xFF)

    await Timer(SCK_HALF_NS, unit="ns")
    dut.spi_cs_n_i.value = 1
    await Timer(SCK_HALF_NS * 4, unit="ns")


async def spi_read(dut, addr: int, num_words: int) -> list:
    """
    spibone burst read.
    Protocol: 0x01 | addr[31:0] | 0xFF*(4*num_words) -> MISO returns data
    """
    dut.spi_cs_n_i.value = 0
    await Timer(SCK_HALF_NS, unit="ns")

    await _spi_byte(dut, 0x01)
    await _spi_byte(dut, (addr >> 24) & 0xFF)
    await _spi_byte(dut, (addr >> 16) & 0xFF)
    await _spi_byte(dut, (addr >>  8) & 0xFF)
    await _spi_byte(dut, (addr      ) & 0xFF)

    results = []
    for i in range(num_words):
        # Read bursts are pipelined in RTL; after each 32-bit word, the next
        # word may need one extra command byte time to be fetched from WB/AXI.
        if i > 0:
            await _spi_byte(dut, 0xFF)
        b3 = await _spi_byte(dut, 0xFF)
        b2 = await _spi_byte(dut, 0xFF)
        b1 = await _spi_byte(dut, 0xFF)
        b0 = await _spi_byte(dut, 0xFF)
        results.append((b3 << 24) | (b2 << 16) | (b1 << 8) | b0)

    await Timer(SCK_HALF_NS, unit="ns")
    dut.spi_cs_n_i.value = 1
    await Timer(SCK_HALF_NS * 4, unit="ns")
    return results


# ---------------------------------------------------------------------------
# Setup fixture
# ---------------------------------------------------------------------------

async def setup(dut):
    """Start clock, reset DUT, launch AXI slave model. Returns AXISlaveModel."""
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD_NS, unit="ns").start())

    dut.spi_sck_i.value  = 0
    dut.spi_mosi_i.value = 0
    dut.spi_cs_n_i.value = 1
    dut.tpu_busy_i.value = 0
    dut.tpu_done_i.value = 0

    dut.axi_awready.value = 0
    dut.axi_wready.value  = 0
    dut.axi_bvalid.value  = 0
    dut.axi_bresp.value   = 0
    dut.axi_arready.value = 0
    dut.axi_rvalid.value  = 0
    dut.axi_rdata.value   = 0
    dut.axi_rlast.value   = 0
    dut.axi_rresp.value   = 0

    dut.rst_i.value = 1
    await ClockCycles(dut.clk_i, 5)
    dut.rst_i.value = 0
    await ClockCycles(dut.clk_i, 3)

    axi_slave = AXISlaveModel(dut)
    cocotb.start_soon(axi_slave.run())
    return axi_slave


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_single_dram_write(dut):
    """Single 32-bit write to DRAM — verify AXI slave receives correct addr+data."""
    axi = await setup(dut)
    await spi_write(dut, DRAM_BASE, [0xDEADBEEF])
    await ClockCycles(dut.clk_i, 20)
    assert axi.mem.get(DRAM_BASE) == 0xDEADBEEF, \
        f"DRAM write failed: got 0x{axi.mem.get(DRAM_BASE, 0):08X}"
    cocotb.log.info("test_single_dram_write passed")


@cocotb.test()
async def test_single_dram_read(dut):
    """Single 32-bit read from DRAM — verify correct data returns over MISO."""
    axi = await setup(dut)
    axi.mem[DRAM_BASE] = 0xCAFEBABE
    data = await spi_read(dut, DRAM_BASE, 1)
    assert data[0] == 0xCAFEBABE, \
        f"DRAM read failed: got 0x{data[0]:08X} expected 0xCAFEBABE"
    cocotb.log.info("test_single_dram_read passed")


@cocotb.test()
async def test_burst_write_4_words(dut):
    """Burst write 4 words — address auto-increments, CS held low."""
    axi = await setup(dut)
    words = [0xAABBCC00, 0xAABBCC01, 0xAABBCC02, 0xAABBCC03]
    await spi_write(dut, 0x00000010, words)
    await ClockCycles(dut.clk_i, 20)
    for i, expected in enumerate(words):
        addr = 0x00000010 + i * 4
        got  = axi.mem.get(addr, 0)
        assert got == expected, \
            f"burst write word[{i}] @ 0x{addr:08X}: got 0x{got:08X} expected 0x{expected:08X}"
    cocotb.log.info("test_burst_write_4_words passed")


@cocotb.test()
async def test_burst_read_4_words(dut):
    """Burst read 4 words — all return correctly over MISO."""
    axi = await setup(dut)
    expected = [0x11111111, 0x22222222, 0x33333333, 0x44444444]
    for i, val in enumerate(expected):
        axi.mem[0x00000010 + i * 4] = val
    data = await spi_read(dut, 0x00000010, 4)
    for i in range(4):
        assert data[i] == expected[i], \
            f"burst read word[{i}]: got 0x{data[i]:08X} expected 0x{expected[i]:08X}"
    cocotb.log.info("test_burst_read_4_words passed")


@cocotb.test()
async def test_tpu_input_addr_reg(dut):
    """Write TPU_INPUT_ADDR register — verify tpu_input_addr_o output."""
    await setup(dut)
    await spi_write(dut, TPU_INPUT_ADDR, [0x00001000])
    await ClockCycles(dut.clk_i, 10)
    got = int(dut.tpu_input_addr_o.value)
    assert got == 0x00001000, \
        f"TPU_INPUT_ADDR: got 0x{got:08X} expected 0x00001000"
    cocotb.log.info("test_tpu_input_addr_reg passed")


@cocotb.test()
async def test_tpu_output_addr_reg(dut):
    """Write TPU_OUTPUT_ADDR register — verify tpu_output_addr_o output."""
    await setup(dut)
    await spi_write(dut, TPU_OUTPUT_ADDR, [0x00100000])
    await ClockCycles(dut.clk_i, 10)
    got = int(dut.tpu_output_addr_o.value)
    assert got == 0x00100000, \
        f"TPU_OUTPUT_ADDR: got 0x{got:08X} expected 0x00100000"
    cocotb.log.info("test_tpu_output_addr_reg passed")


@cocotb.test()
async def test_tpu_length_reg(dut):
    """Write TPU_LENGTH register — verify tpu_length_o output."""
    await setup(dut)
    await spi_write(dut, TPU_LENGTH, [0x00000100])
    await ClockCycles(dut.clk_i, 10)
    got = int(dut.tpu_length_o.value)
    assert got == 0x00000100, \
        f"TPU_LENGTH: got 0x{got:08X} expected 0x00000100"
    cocotb.log.info("test_tpu_length_reg passed")


@cocotb.test()
async def test_tpu_status_idle(dut):
    """STATUS reads 0x0 when TPU is idle."""
    await setup(dut)
    dut.tpu_busy_i.value = 0
    dut.tpu_done_i.value = 0
    data = await spi_read(dut, TPU_STATUS, 1)
    assert data[0] == 0, \
        f"TPU STATUS idle: got 0x{data[0]:08X} expected 0x00000000"
    cocotb.log.info("test_tpu_status_idle passed")


@cocotb.test()
async def test_tpu_status_busy(dut):
    """STATUS[0] reflects tpu_busy_i."""
    await setup(dut)
    dut.tpu_busy_i.value = 1
    dut.tpu_done_i.value = 0
    data = await spi_read(dut, TPU_STATUS, 1)
    assert data[0] & STATUS_BUSY, \
        f"TPU STATUS busy bit: got 0x{data[0]:08X} expected bit 0 set"
    cocotb.log.info("test_tpu_status_busy passed")


@cocotb.test()
async def test_tpu_status_done(dut):
    """STATUS[1] reflects tpu_done_i."""
    await setup(dut)
    dut.tpu_busy_i.value = 0
    dut.tpu_done_i.value = 1
    data = await spi_read(dut, TPU_STATUS, 1)
    assert data[0] & STATUS_DONE, \
        f"TPU STATUS done bit: got 0x{data[0]:08X} expected bit 1 set"
    cocotb.log.info("test_tpu_status_done passed")


@cocotb.test()
async def test_tpu_start_pulse(dut):
    """CTRL[0]=1 produces a single-cycle tpu_start_o pulse that auto-clears."""
    await setup(dut)
    write_task = cocotb.start_soon(spi_write(dut, TPU_CTRL, [0x00000001]))
    pulse_seen = False
    for _ in range(1000):
        await RisingEdge(dut.clk_i)
        if int(dut.tpu_start_o.value) == 1:
            pulse_seen = True
            await RisingEdge(dut.clk_i)
            assert int(dut.tpu_start_o.value) == 0, \
                "tpu_start_o did not auto-clear after one cycle"
            break
    await write_task
    assert pulse_seen, "tpu_start_o never pulsed high"
    cocotb.log.info("test_tpu_start_pulse passed")


@cocotb.test()
async def test_tpu_reset(dut):
    """CTRL[1]=1 asserts tpu_reset_o."""
    await setup(dut)
    write_task = cocotb.start_soon(spi_write(dut, TPU_CTRL, [0x00000002]))
    reset_seen = False
    for _ in range(1000):
        await RisingEdge(dut.clk_i)
        if int(dut.tpu_reset_o.value) == 1:
            reset_seen = True
            break
    await write_task
    assert reset_seen, "tpu_reset_o never asserted"
    cocotb.log.info("test_tpu_reset passed")


@cocotb.test()
async def test_unmapped_address(dut):
    """Read from unmapped address returns 0xDEADBEEF without hanging."""
    await setup(dut)
    data = await spi_read(dut, UNMAPPED_ADDR, 1)
    assert data[0] == 0xDEADBEEF, \
        f"unmapped addr: got 0x{data[0]:08X} expected 0xDEADBEEF"
    cocotb.log.info("test_unmapped_address passed")


@cocotb.test()
async def test_wb_decoder_dram_route(dut):
    """Address in DRAM range routes to AXI bridge (not TPU regs)."""
    axi = await setup(dut)
    await spi_write(dut, 0x08000000, [0xABCD1234])
    await ClockCycles(dut.clk_i, 20)
    got = axi.mem.get(0x08000000, 0)
    assert got == 0xABCD1234, \
        f"decoder DRAM route: got 0x{got:08X} expected 0xABCD1234"
    cocotb.log.info("test_wb_decoder_dram_route passed")


@cocotb.test()
async def test_1kb_burst_write(dut):
    """Write 256 words (1KB) in one burst and spot-check first, mid, last."""
    axi = await setup(dut)
    base  = 0x00000200
    words = [0xA0000000 | i for i in range(256)]
    await spi_write(dut, base, words)
    await ClockCycles(dut.clk_i, 50)
    assert axi.mem.get(base + 0,    0) == 0xA0000000, "1KB burst write [0] failed"
    assert axi.mem.get(base + 508,  0) == 0xA000007F, "1KB burst write [127] failed"
    assert axi.mem.get(base + 1020, 0) == 0xA00000FF, "1KB burst write [255] failed"
    cocotb.log.info("test_1kb_burst_write passed")


@cocotb.test()
async def test_1kb_burst_read(dut):
    """Read 256 words (1KB) in one burst and verify all 256 values."""
    axi = await setup(dut)
    base = 0x00000200
    for i in range(256):
        axi.mem[base + i * 4] = 0xB0000000 | i
    data = await spi_read(dut, base, 256)
    mismatches = [i for i in range(256) if data[i] != (0xB0000000 | i)]
    assert len(mismatches) == 0, \
        f"1KB burst read: {len(mismatches)} mismatches, first at word {mismatches[0]}"
    cocotb.log.info("test_1kb_burst_read passed — all 256 words correct")


@cocotb.test()
async def test_full_inference_flow(dut):
    """End-to-end: load 1KB → configure TPU → start → poll done → read results."""
    axi = await setup(dut)

    INPUT_ADDR  = 0x00000000
    OUTPUT_ADDR = 0x00100000
    N_WORDS     = 256

    # 1. Load input data into DRAM
    input_words = [0xDA7A0000 | i for i in range(N_WORDS)]
    await spi_write(dut, INPUT_ADDR, input_words)
    await ClockCycles(dut.clk_i, 50)
    assert axi.mem.get(INPUT_ADDR) == 0xDA7A0000, "inference: input load failed"

    # 2. Configure TPU registers
    await spi_write(dut, TPU_INPUT_ADDR,  [INPUT_ADDR])
    await spi_write(dut, TPU_OUTPUT_ADDR, [OUTPUT_ADDR])
    await spi_write(dut, TPU_LENGTH,      [N_WORDS])
    await ClockCycles(dut.clk_i, 10)
    assert int(dut.tpu_input_addr_o.value)  == INPUT_ADDR,  "inference: input addr reg wrong"
    assert int(dut.tpu_output_addr_o.value) == OUTPUT_ADDR, "inference: output addr reg wrong"
    assert int(dut.tpu_length_o.value)      == N_WORDS,     "inference: length reg wrong"

    # 3. Kick off TPU
    start_task = cocotb.start_soon(spi_write(dut, TPU_CTRL, [0x1]))
    pulse_seen = False
    for _ in range(1000):
        await RisingEdge(dut.clk_i)
        if int(dut.tpu_start_o.value):
            pulse_seen = True
            break
    await start_task
    assert pulse_seen, "inference: tpu_start_o never pulsed"

    # 4. Simulate TPU busy then done
    dut.tpu_busy_i.value = 1
    await ClockCycles(dut.clk_i, 20)
    dut.tpu_busy_i.value = 0
    dut.tpu_done_i.value = 1

    # Poll STATUS until done bit set
    done = False
    for _ in range(20):
        status = await spi_read(dut, TPU_STATUS, 1)
        if status[0] & STATUS_DONE:
            done = True
            break
    assert done, "inference: STATUS done bit never seen"

    # 5. Read results back from DRAM
    for i in range(64):
        axi.mem[OUTPUT_ADDR + i * 4] = 0x0C070000 | i
    results = await spi_read(dut, OUTPUT_ADDR, 64)
    assert results[0]  == 0x0C070000, f"inference result[0]: 0x{results[0]:08X}"
    assert results[63] == 0x0C07003F, f"inference result[63]: 0x{results[63]:08X}"
    cocotb.log.info("test_full_inference_flow passed")


@cocotb.test()
async def test_random_write_read(dut):
    """10 random write-then-read round trips through SPI -> WB -> AXI."""
    axi = await setup(dut)
    for i in range(10):
        addr  = (random.randint(0, 0x00FFFFFF) & ~3)  # DRAM range, word-aligned
        value = random.randint(0, 0xFFFFFFFF)
        await spi_write(dut, addr, [value])
        await ClockCycles(dut.clk_i, 20)
        assert axi.mem.get(addr) == value, \
            f"iter {i}: write addr=0x{addr:08X} value=0x{value:08X} not in AXI mem"
        rx = await spi_read(dut, addr, 1)
        assert rx[0] == value, \
            f"iter {i}: read back mismatch addr=0x{addr:08X} " \
            f"expected=0x{value:08X} got=0x{rx[0]:08X}"
        cocotb.log.info(f"  random iter {i}: 0x{addr:08X} = 0x{value:08X} OK")
    cocotb.log.info("test_random_write_read passed")


# ---------------------------------------------------------------------------
# Pytest boilerplate — matches the pattern used in test_house.py
# ---------------------------------------------------------------------------
tests = [
    "test_single_dram_write",
    "test_single_dram_read",
    "test_burst_write_4_words",
    "test_burst_read_4_words",
    "test_tpu_input_addr_reg",
    "test_tpu_output_addr_reg",
    "test_tpu_length_reg",
    "test_tpu_status_idle",
    "test_tpu_status_busy",
    "test_tpu_status_done",
    "test_tpu_start_pulse",
    "test_tpu_reset",
    "test_unmapped_address",
    "test_wb_decoder_dram_route",
    "test_1kb_burst_write",
    "test_1kb_burst_read",
    "test_full_inference_flow",
    "test_random_write_read",
]

SOURCES = [
    Path("./src/spi/spibone_wb.sv").resolve(),
    Path("./src/spi/wb_decoder.sv").resolve(),
    Path("./src/spi/tpu_regs.sv").resolve(),
    Path("./src/spi/wb_to_axi4.sv").resolve(),
    Path("./src/spi/spi_top2.sv").resolve(),
    Path("./src/spi/tb_top.sv").resolve(),
]


@pytest.mark.parametrize("testcase", tests)
def test_spi_system_each(testcase):
    run_test(
        sources=SOURCES,
        module_name="test_spi_system",
        hdl_toplevel="tb_top",
        parameters={},
        testcase=testcase,
        sims=["icarus"],
    )


def test_spi_system_all():
    run_test(
        sources=SOURCES,
        module_name="test_spi_system",
        hdl_toplevel="tb_top",
        parameters={},
        sims=["icarus"],
    )