import pytest
import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from pathlib import Path
from runner import run_test
from cocotbext.spi import SpiMaster, SpiBus, SpiConfig
import random
from collections import deque


# ---------------------------------------------------------------------------
# SPI config — Mode 0: CPOL=0, CPHA=0, RTL samples SDI on posedge SCK
# ---------------------------------------------------------------------------
spi_config = SpiConfig(
    word_width    = 8,
    sclk_freq     = 1e6,
    cpol          = False,
    cpha          = False,
    msb_first     = True,
    cs_active_low = True,
)

# RTL state encoding from housekeeping_spi.v
STATE_COMMAND  = 0b000
STATE_ADDRESS  = 0b001
STATE_DATA     = 0b010
STATE_USERPASS = 0b100
STATE_MGMTPASS = 0b101

# At 1 MHz SCK, one 3-byte transaction = 24 bits = 24 µs.
# Allow 10x that before declaring a hang.
CAPTURE_TIMEOUT_NS = 300_000


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------
class HousekeepingSPIModel:
    def __init__(self):
        self.q = deque()

    def consume(self, addr, data):
        self.q.append((addr, data))

    def produce(self, addr, data):
        exp_addr, exp_data = self.q.popleft()
        assert addr == exp_addr, \
            f"oaddr: expected {hex(exp_addr)}, got {hex(addr)}"
        assert data == exp_data, \
            f"odata: expected {hex(exp_data)}, got {hex(data)}"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
async def setup(dut):
    dut.reset.value = 1
    dut.CSB.value   = 1
    dut.SCK.value   = 0
    dut.SDI.value   = 0
    await Timer(200, unit="ns")
    dut.reset.value = 0
    await Timer(200, unit="ns")

    spi_bus = SpiBus(
        entity    = dut,
        prefix    = None,
        sclk_name = "SCK",
        mosi_name = "SDI",
        miso_name = "SDO",
        cs_name   = "CSB",
    )
    return SpiMaster(spi_bus, spi_config)


# ---------------------------------------------------------------------------
# Timeout watchdog
#
# Sets timed_out["v"] = True after timeout_ns nanoseconds.
# Run as a background task alongside a capture task; the capture task checks
# this flag after each RisingEdge(SCK) to know whether to abort.
# ---------------------------------------------------------------------------
async def timeout_watchdog(timeout_ns, timed_out):
    await Timer(timeout_ns, unit="ns")
    timed_out["v"] = True


# ---------------------------------------------------------------------------
# Core capture task
#
# Capture (oaddr, odata) on posedge SCK where state==DATA and count==7.
# Uses a shared timed_out dict set by a parallel timeout_watchdog task.
# ---------------------------------------------------------------------------
async def write_capture_task(dut, num_data_bytes, results, timed_out):
    captured = 0
    while captured < num_data_bytes:
        if timed_out["v"]:
            raise RuntimeError(
                f"write_capture_task timed out (captured {captured}/{num_data_bytes})"
            )
        await RisingEdge(dut.SCK)
        if timed_out["v"]:
            raise RuntimeError(
                f"write_capture_task timed out (captured {captured}/{num_data_bytes})"
            )
        await Timer(1, unit="ns")   # let always@(posedge SCK) registers settle
        state = int(dut.state.value)
        count = int(dut.count.value)
        if state == STATE_DATA and count == 7:
            results.append((int(dut.oaddr.value), int(dut.odata.value)))
            captured += 1


async def rdstb_capture_task(dut, timed_out):
    while True:
        if timed_out["v"]:
            raise RuntimeError("rdstb_capture_task timed out")
        await RisingEdge(dut.SCK)
        if timed_out["v"]:
            raise RuntimeError("rdstb_capture_task timed out")
        await Timer(1, unit="ns")
        state = int(dut.state.value)
        count = int(dut.count.value)
        if state == STATE_ADDRESS and count == 7:
            return int(dut.rdstb.value)


# ---------------------------------------------------------------------------
# Transaction helpers
# ---------------------------------------------------------------------------
async def spi_write(spi_master, addr, data_bytes):
    await spi_master.write([0x80, addr] + list(data_bytes))
    await spi_master.wait()


async def spi_read(spi_master, addr):
    await spi_master.write([0x40, addr, 0x00])
    await spi_master.wait()
    result = await spi_master.read()
    return result[2]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def reset_test(dut):
    """Verify housekeeping SPI is idle after reset."""
    dut.reset.value = 1
    dut.CSB.value   = 1
    dut.SCK.value   = 0
    dut.SDI.value   = 0
    await Timer(200, unit="ns")
    dut.reset.value = 0
    await Timer(200, unit="ns")

    assert dut.wrstb.value  == 0, "wrstb should be 0 after reset"
    assert dut.rdstb.value  == 0, "rdstb should be 0 after reset"
    assert dut.sdoenb.value == 1, "sdoenb should be 1 after reset"
    cocotb.log.info("Reset test passed")


@cocotb.test()
async def test_write_instruction(dut):
    """Send a write transaction; capture oaddr/odata when state=DATA, count=7."""
    spi_master = await setup(dut)
    model = HousekeepingSPIModel()

    addr, data = 0x08, 0x07
    model.consume(addr, data)

    results   = []
    timed_out = {"v": False}
    cocotb.start_soon(timeout_watchdog(CAPTURE_TIMEOUT_NS, timed_out))
    capture = cocotb.start_soon(write_capture_task(dut, 1, results, timed_out))

    await spi_write(spi_master, addr, [data])
    await capture
    await Timer(500, unit="ns")

    assert len(results) == 1, f"Expected 1 capture, got {len(results)}"
    model.produce(results[0][0], results[0][1])
    cocotb.log.info(f"Write passed: addr={hex(addr)} data={hex(data)}")


@cocotb.test()
async def test_wrstb_fires_once(dut):
    """Exactly one data byte → capture fires exactly once."""
    spi_master = await setup(dut)

    results   = []
    timed_out = {"v": False}
    cocotb.start_soon(timeout_watchdog(CAPTURE_TIMEOUT_NS, timed_out))
    capture = cocotb.start_soon(write_capture_task(dut, 1, results, timed_out))

    await spi_write(spi_master, 0x08, [0x07])
    await capture
    await Timer(500, unit="ns")

    assert len(results) == 1, f"Expected 1 capture, got {len(results)}"
    cocotb.log.info("wrstb fired exactly once")


@cocotb.test()
async def test_auto_increment(dut):
    """Multi-byte write: oaddr must increment by 1 per data byte."""
    spi_master = await setup(dut)

    start_addr = 0x08
    data_bytes = [0x01, 0x02, 0x03, 0x04]

    results   = []
    timed_out = {"v": False}
    cocotb.start_soon(timeout_watchdog(CAPTURE_TIMEOUT_NS, timed_out))
    capture = cocotb.start_soon(
        write_capture_task(dut, len(data_bytes), results, timed_out)
    )

    await spi_write(spi_master, start_addr, data_bytes)
    await capture
    await Timer(500, unit="ns")

    got_addrs = [r[0] for r in results]
    got_data  = [r[1] for r in results]
    exp_addrs = [start_addr + i for i in range(len(data_bytes))]

    assert got_addrs == exp_addrs, \
        f"oaddr seq wrong: got {[hex(a) for a in got_addrs]}, " \
        f"expected {[hex(a) for a in exp_addrs]}"
    assert got_data == data_bytes, \
        f"odata seq wrong: got {[hex(d) for d in got_data]}, " \
        f"expected {[hex(d) for d in data_bytes]}"
    cocotb.log.info("Auto-increment passed")


@cocotb.test()
async def test_read_transaction(dut):
    """Read transaction: rdstb must be 1 when state=ADDRESS, count=7."""
    spi_master = await setup(dut)

    timed_out = {"v": False}
    cocotb.start_soon(timeout_watchdog(CAPTURE_TIMEOUT_NS, timed_out))
    rdstb_task = cocotb.start_soon(rdstb_capture_task(dut, timed_out))

    await spi_read(spi_master, 0x08)
    rdstb_val = await rdstb_task
    await Timer(500, unit="ns")

    assert rdstb_val == 1, \
        f"rdstb was {rdstb_val} at state=ADDRESS,count=7; expected 1"
    cocotb.log.info("rdstb fired correctly")


@cocotb.test()
async def test_csb_reset(dut):
    """Asserting CSB resets wrstb, rdstb, and sdoenb to idle."""
    spi_master = await setup(dut)   # noqa: F841
    dut.CSB.value = 1
    await Timer(200, unit="ns")

    assert dut.wrstb.value  == 0, "wrstb should be 0 after CSB"
    assert dut.rdstb.value  == 0, "rdstb should be 0 after CSB"
    assert dut.sdoenb.value == 1, "sdoenb should be 1 after CSB"
    cocotb.log.info("CSB reset test passed")


@cocotb.test()
async def test_random_instructions(dut):
    """10 random write transactions; verify oaddr and odata on each."""
    spi_master = await setup(dut)

    for i in range(10):
        addr = random.randint(0x08, 0xEF)
        data = random.randint(0x00, 0xFF)

        results   = []
        timed_out = {"v": False}
        cocotb.start_soon(timeout_watchdog(CAPTURE_TIMEOUT_NS, timed_out))
        capture = cocotb.start_soon(write_capture_task(dut, 1, results, timed_out))

        await spi_write(spi_master, addr, [data])
        await capture
        await Timer(500, unit="ns")

        assert len(results) == 1, f"iter {i}: no capture"
        got_addr, got_data = results[0]
        assert got_addr == addr, \
            f"iter {i}: oaddr mismatch: expected {hex(addr)}, got {hex(got_addr)}"
        assert got_data == data, \
            f"iter {i}: odata mismatch: expected {hex(data)}, got {hex(got_data)}"

    cocotb.log.info("Random instruction test passed")


@cocotb.test()
async def test_writemode_set(dut):
    """writemode latches 1 on the first posedge SCK (count=0, state=COMMAND)."""
    spi_master = await setup(dut)
    seen_high = {"v": False}

    async def watch():
        await FallingEdge(dut.CSB)
        await RisingEdge(dut.SCK)
        await Timer(1, unit="ns")
        if int(dut.writemode.value) == 1:
            seen_high["v"] = True

    cocotb.start_soon(watch())
    await spi_write(spi_master, 0x08, [0xAB])
    await Timer(500, unit="ns")

    assert seen_high["v"], "writemode never went high"
    assert int(dut.writemode.value) == 0, "writemode should clear after CSB"
    cocotb.log.info("writemode set/clear passed")


@cocotb.test()
async def test_debug_csb(dut):
    """Sanity: CSB goes low, SCK toggles, state/count-based capture works."""
    spi_master = await setup(dut)

    addr, data = 0x08, 0x07
    results   = []
    timed_out = {"v": False}
    cocotb.start_soon(timeout_watchdog(CAPTURE_TIMEOUT_NS, timed_out))
    capture = cocotb.start_soon(write_capture_task(dut, 1, results, timed_out))

    obs = {"csb_low": False, "sck": False}

    async def monitor():
        prev = 0
        while True:
            await Timer(10, unit="ns")
            if int(dut.CSB.value) == 0:
                obs["csb_low"] = True
            cur = int(dut.SCK.value)
            if cur != prev:
                obs["sck"] = True
            prev = cur

    cocotb.start_soon(monitor())
    await spi_write(spi_master, addr, [data])
    await capture
    await Timer(500, unit="ns")

    cocotb.log.info(f"CSB went low: {obs['csb_low']}")
    cocotb.log.info(f"SCK toggled:  {obs['sck']}")
    cocotb.log.info(f"Captured:     {[(hex(a), hex(d)) for a,d in results]}")

    assert obs["csb_low"], "CSB never went low"
    assert obs["sck"],     "SCK never toggled"
    assert len(results) == 1, "oaddr/odata not captured"
    assert results[0] == (addr, data), \
        f"captured {results[0]}, expected {(addr, data)}"


# ---------------------------------------------------------------------------
# Pytest boilerplate
# ---------------------------------------------------------------------------
tests = [
    "reset_test",
    "test_write_instruction",
    "test_wrstb_fires_once",
    "test_auto_increment",
    "test_read_transaction",
    "test_csb_reset",
    "test_random_instructions",
    "test_writemode_set",
    "test_debug_csb",
]

SOURCES = [Path("./rtl/housekeeping_spi.v").resolve()]


@pytest.mark.parametrize("testcase", tests)
def test_housekeeping_spi_each(testcase):
    run_test(
        sources=SOURCES,
        module_name="test_house",
        hdl_toplevel="housekeeping_spi",
        parameters={},
        testcase=testcase,
        sims=["icarus"],
    )


def test_housekeeping_spi_all():
    run_test(
        sources=SOURCES,
        module_name="test_house",
        hdl_toplevel="housekeeping_spi",
        parameters={},
        sims=["icarus"],
    )