import pytest
import cocotb
from cocotb.triggers import RisingEdge, Timer, ClockCycles
from cocotb.clock import Clock
from pathlib import Path
from runner import run_test

# ---------------------------------------------------------------------------
# Timing constants — 100 MHz system clock, 2 MHz SPI SCK
# ---------------------------------------------------------------------------
CLK_PERIOD_NS   = 10
SCK_HALF_NS     = 250
SCK_HALF_CYCLES = SCK_HALF_NS // CLK_PERIOD_NS

# Register map (must match tpu_regs.sv and wb_decoder.sv)
TPU_BASE   = 0x10000000
TPU_PC     = TPU_BASE + 0x00   # RW  PC_ADDR
TPU_STATUS = TPU_BASE + 0x04   # RO  tpu_state_i
TPU_CTRL   = TPU_BASE + 0x08   # RW  [0]=tpu_enable

# tpu_state_i encoding (matches control_top)
TPU_IDLE    = 0b01
TPU_RUNNING = 0b10   # placeholder — adjust to your actual encoding


# ---------------------------------------------------------------------------
# SPI helpers (unchanged protocol from original testbench)
# ---------------------------------------------------------------------------

async def _spi_byte(dut, tx: int) -> int:
    rx = 0
    for bit in range(7, -1, -1):
        dut.spi_mosi_i.value = (tx >> bit) & 1
        await ClockCycles(dut.clk_i, SCK_HALF_CYCLES)
        await Timer(1, unit="ps")
        rx = (rx << 1) | int(dut.spi_miso_o.value)
        dut.spi_sck_i.value = 1
        await ClockCycles(dut.clk_i, SCK_HALF_CYCLES)
        dut.spi_sck_i.value = 0
        await Timer(1, unit="ps")
    return rx


async def spi_write(dut, addr: int, words: list) -> None:
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
    dut.spi_cs_n_i.value = 0
    await Timer(SCK_HALF_NS, unit="ns")
    await _spi_byte(dut, 0x01)
    await _spi_byte(dut, (addr >> 24) & 0xFF)
    await _spi_byte(dut, (addr >> 16) & 0xFF)
    await _spi_byte(dut, (addr >>  8) & 0xFF)
    await _spi_byte(dut, (addr      ) & 0xFF)
    results = []
    for i in range(num_words):
        if i > 0:
            # burst pipeline byte between words
            await _spi_byte(dut, 0xFF)
        # spibone emits 0x00 wait bytes then 0x01 ready before data
        ready = 0
        while ready != 0x01:
            ready = await _spi_byte(dut, 0xFF)
        # data is MSB-first: b3=highest byte, b0=lowest
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
# Setup
# ---------------------------------------------------------------------------

async def setup(dut):
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD_NS, unit="ns").start())

    dut.spi_sck_i.value  = 0
    dut.spi_mosi_i.value = 0
    dut.spi_cs_n_i.value = 1
    dut.tpu_state_i.value = TPU_IDLE   # start in IDLE

    dut.rst_i.value = 1
    await ClockCycles(dut.clk_i, 5)
    dut.rst_i.value = 0
    await ClockCycles(dut.clk_i, 3)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_pc_write_strobe(dut):
    """Writing PC_ADDR pulses tpu_pc_stb_o high then low."""
    await setup(dut)

    pulse_seen = False
    pulse_cleared = False

    async def monitor():
        nonlocal pulse_seen, pulse_cleared
        for _ in range(10000):
            await RisingEdge(dut.clk_i)
            if int(dut.tpu_pc_stb_o.value) == 1:
                pulse_seen = True
            if pulse_seen and int(dut.tpu_pc_stb_o.value) == 0:
                pulse_cleared = True
                return

    mon = cocotb.start_soon(monitor())
    await spi_write(dut, TPU_PC, [0xDEAD1234])
    await ClockCycles(dut.clk_i, 10)
    await mon

    assert pulse_seen,    "tpu_pc_stb_o never pulsed after PC write"
    assert pulse_cleared, "tpu_pc_stb_o never cleared after pulsing"
    cocotb.log.info("test_pc_write_strobe passed")


@cocotb.test()
async def test_pc_addr_readback(dut):
    """PC_ADDR register holds written value and reads back correctly."""
    await setup(dut)
    await spi_write(dut, TPU_PC, [0xCAFE0000])
    await ClockCycles(dut.clk_i, 10)
    data = await spi_read(dut, TPU_PC, 1)
    assert data[0] == 0xCAFE0000, \
        f"PC readback: got 0x{data[0]:08X} expected 0xCAFE0000"
    cocotb.log.info("test_pc_addr_readback passed")


@cocotb.test()
async def test_status_reflects_state(dut):
    """STATUS register exposes tpu_state_i directly."""
    await setup(dut)

    for state in [0b00, 0b01, 0b10, 0b11]:
        dut.tpu_state_i.value = state
        await ClockCycles(dut.clk_i, 2)
        data = await spi_read(dut, TPU_STATUS, 1)
        assert (data[0] & 0x3) == state, \
            f"STATUS for state {state:02b}: got 0x{data[0]:08X}"

    cocotb.log.info("test_status_reflects_state passed")


@cocotb.test()
async def test_ctrl_enable(dut):
    """Writing CTRL[0] sets tpu_enable_o; clearing it deasserts."""
    await setup(dut)

    await spi_write(dut, TPU_CTRL, [0x1])
    await ClockCycles(dut.clk_i, 10)
    assert int(dut.tpu_enable_o.value) == 1, "tpu_enable_o not set"

    await spi_write(dut, TPU_CTRL, [0x0])
    await ClockCycles(dut.clk_i, 10)
    assert int(dut.tpu_enable_o.value) == 0, "tpu_enable_o not cleared"

    cocotb.log.info("test_ctrl_enable passed")


@cocotb.test()
async def test_done_fires_after_pc_load(dut):
    """tpu_done_o pulses when TPU returns to IDLE after a PC write."""
    await setup(dut)

    # Start IDLE, write PC to arm the detector
    dut.tpu_state_i.value = TPU_IDLE
    await spi_write(dut, TPU_PC, [0x00001000])
    await ClockCycles(dut.clk_i, 5)

    # TPU leaves IDLE (starts running)
    dut.tpu_state_i.value = TPU_RUNNING
    await ClockCycles(dut.clk_i, 20)

    # TPU returns to IDLE — done should fire
    dut.tpu_state_i.value = TPU_IDLE

    done_seen = False
    for _ in range(10):
        await RisingEdge(dut.clk_i)
        if int(dut.tpu_done_o.value) == 1:
            done_seen = True
            # Verify it clears next cycle
            await RisingEdge(dut.clk_i)
            assert int(dut.tpu_done_o.value) == 0, \
                "tpu_done_o did not clear after one cycle"
            break

    assert done_seen, "tpu_done_o never pulsed after TPU returned to IDLE"
    cocotb.log.info("test_done_fires_after_pc_load passed")


@cocotb.test()
async def test_done_no_fire_without_pc_load(dut):
    """tpu_done_o must NOT fire if no PC was written before IDLE transition."""
    await setup(dut)

    # Drive IDLE -> running -> IDLE without any PC write
    dut.tpu_state_i.value = TPU_IDLE
    await ClockCycles(dut.clk_i, 5)
    dut.tpu_state_i.value = TPU_RUNNING
    await ClockCycles(dut.clk_i, 20)
    dut.tpu_state_i.value = TPU_IDLE
    await ClockCycles(dut.clk_i, 10)

    assert int(dut.tpu_done_o.value) == 0, \
        "tpu_done_o fired spuriously without a PC load"
    cocotb.log.info("test_done_no_fire_without_pc_load passed")


@cocotb.test()
async def test_done_disarms_after_firing(dut):
    """After tpu_done_o fires once, a second IDLE transition without a new PC
    write must NOT fire done again."""
    await setup(dut)

    # First run — arm with PC write, let it complete
    dut.tpu_state_i.value = TPU_IDLE
    await spi_write(dut, TPU_PC, [0x00002000])
    await ClockCycles(dut.clk_i, 5)
    dut.tpu_state_i.value = TPU_RUNNING
    await ClockCycles(dut.clk_i, 10)
    dut.tpu_state_i.value = TPU_IDLE
    await ClockCycles(dut.clk_i, 10)  # done fires and disarms here

    # Second IDLE transition without a new PC write
    dut.tpu_state_i.value = TPU_RUNNING
    await ClockCycles(dut.clk_i, 10)
    dut.tpu_state_i.value = TPU_IDLE
    await ClockCycles(dut.clk_i, 10)

    assert int(dut.tpu_done_o.value) == 0, \
        "tpu_done_o fired a second time without a new PC load"
    cocotb.log.info("test_done_disarms_after_firing passed")


@cocotb.test()
async def test_reset_clears_state(dut):
    """After rst_i, tpu_done_o, tpu_pc_stb_o, tpu_enable_o are all low."""
    await setup(dut)

    # Set some state
    await spi_write(dut, TPU_PC,   [0xABCD0000])
    await spi_write(dut, TPU_CTRL, [0x1])
    await ClockCycles(dut.clk_i, 5)

    # Reset
    dut.rst_i.value = 1
    await ClockCycles(dut.clk_i, 5)
    dut.rst_i.value = 0
    await ClockCycles(dut.clk_i, 3)

    assert int(dut.tpu_done_o.value)    == 0, "tpu_done_o not cleared by reset"
    assert int(dut.tpu_pc_stb_o.value)  == 0, "tpu_pc_stb_o not cleared by reset"
    assert int(dut.tpu_enable_o.value)  == 0, "tpu_enable_o not cleared by reset"
    cocotb.log.info("test_reset_clears_state passed")


@cocotb.test()
async def test_full_inference_flow(dut):
    """End-to-end: write PC, enable TPU, simulate run, observe done pulse,
    verify STATUS transitions, poll STATUS after done."""
    await setup(dut)

    PC_VALUE = 0x00001000

    # 1. Enable TPU bus mux
    await spi_write(dut, TPU_CTRL, [0x1])
    await ClockCycles(dut.clk_i, 5)
    assert int(dut.tpu_enable_o.value) == 1, "tpu_enable_o not set"

    # 2. Load PC — arms done detector
    dut.tpu_state_i.value = TPU_IDLE
    await spi_write(dut, TPU_PC, [PC_VALUE])
    await ClockCycles(dut.clk_i, 5)

    # Verify PC readback
    data = await spi_read(dut, TPU_PC, 1)
    assert data[0] == PC_VALUE, f"PC readback wrong: 0x{data[0]:08X}"

    # 3. Simulate TPU starting
    dut.tpu_state_i.value = TPU_RUNNING
    await ClockCycles(dut.clk_i, 5)

    status = await spi_read(dut, TPU_STATUS, 1)
    assert (status[0] & 0x3) == TPU_RUNNING, \
        f"STATUS during run: 0x{status[0]:08X}"

    # 4. Simulate TPU finishing — returns to IDLE
    dut.tpu_state_i.value = TPU_IDLE

    done_seen = False
    for _ in range(10):
        await RisingEdge(dut.clk_i)
        if int(dut.tpu_done_o.value):
            done_seen = True
            break

    assert done_seen, "tpu_done_o never fired at end of inference"

    # 5. STATUS should now show IDLE
    status = await spi_read(dut, TPU_STATUS, 1)
    assert (status[0] & 0x3) == TPU_IDLE, \
        f"STATUS after done: 0x{status[0]:08X}"

    cocotb.log.info("test_full_inference_flow passed")


# ---------------------------------------------------------------------------
# Pytest boilerplate
# ---------------------------------------------------------------------------
tests = [
    "test_pc_write_strobe",
    "test_pc_addr_readback",
    "test_status_reflects_state",
    "test_ctrl_enable",
    "test_done_fires_after_pc_load",
    "test_done_no_fire_without_pc_load",
    "test_done_disarms_after_firing",
    "test_reset_clears_state",
    "test_full_inference_flow",
]

SOURCES = [
    Path("./src/spi/spibone_wb.sv").resolve(),
    Path("./src/spi/wb_decoder.sv").resolve(),
    Path("./src/spi/tpu_regs.sv").resolve(),
    Path("./src/spi/tb_top.sv").resolve(),
]


@pytest.mark.parametrize("testcase", tests)
def test_tpu_regs_each(testcase):
    run_test(
        sources=SOURCES,
        module_name="test_tpu_regs",
        hdl_toplevel="tb_top",
        parameters={},
        testcase=testcase,
        sims=["icarus"],
    )


def test_tpu_regs_all():
    run_test(
        sources=SOURCES,
        module_name="test_tpu_regs",
        hdl_toplevel="tb_top",
        parameters={},
        sims=["icarus"],
    )