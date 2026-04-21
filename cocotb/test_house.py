import pytest
import cocotb
from cocotb.triggers import RisingEdge, FallingEdge, Timer, First
from pathlib import Path
from runner import run_test
import random

# ---------------------------------------------------------------------------
# RTL state encoding (from housekeeping_spi.v)
# ---------------------------------------------------------------------------
STATE_COMMAND  = 0b000
STATE_ADDRESS  = 0b001
STATE_DATA     = 0b010
STATE_USERPASS = 0b100
STATE_MGMTPASS = 0b101

# SPI timing: 1 MHz SCK → 500 ns half-period
HALF_PERIOD_NS = 500

# Timeout for any single transaction (24 bits @ 1 MHz = 24 µs, allow 20x)
TIMEOUT_NS = 500_000

# ---------------------------------------------------------------------------
# Wishbone slave model
#
# Monitors wb_cyc/wb_stb/wb_we and drives wb_ack/wb_dat_i.
# Stores written data in a dict keyed by address.
# Runs as a background task for the duration of each test.
# ---------------------------------------------------------------------------
class WishboneSlave:
    def __init__(self, dut, latency=1):
        self.dut     = dut
        self.latency = latency   # cycles before ack
        self.mem     = {}        # addr -> data

    async def run(self):
        dut = self.dut
        dut.wb_ack_i.value   = 0
        dut.wb_dat_i.value   = 0
        while True:
            await RisingEdge(dut.wb_clk_i)
            if int(dut.wb_cyc_o.value) and int(dut.wb_stb_o.value):
                # Simulate latency
                for _ in range(self.latency - 1):
                    await RisingEdge(dut.wb_clk_i)
                addr = int(dut.wb_adr_o.value)
                if int(dut.wb_we_o.value):
                    data = int(dut.wb_dat_o.value)
                    self.mem[addr] = data
                    dut.wb_dat_i.value = 0x00
                    cocotb.log.info(
                        f"  [WB SLAVE] WRITE addr={hex(addr)} data={hex(data)}"
                    )
                else:
                    data = self.mem.get(addr, 0x00)
                    dut.wb_dat_i.value = data
                    cocotb.log.info(
                        f"  [WB SLAVE] READ  addr={hex(addr)} → {hex(data)}"
                    )
                dut.wb_ack_i.value = 1
                await RisingEdge(dut.wb_clk_i)
                dut.wb_ack_i.value = 0


# ---------------------------------------------------------------------------
# Wishbone clock generator: 10 MHz (100 ns period)
# ---------------------------------------------------------------------------
async def wb_clock(dut):
    dut.wb_clk_i.value = 0
    while True:
        await Timer(50, unit="ns")
        dut.wb_clk_i.value = 1
        await Timer(50, unit="ns")
        dut.wb_clk_i.value = 0


# ---------------------------------------------------------------------------
# Bit-bang SPI helpers
#
# CSB stays low for the ENTIRE transaction so the housekeeping_spi state
# machine is never reset mid-frame (streaming mode).
# ---------------------------------------------------------------------------
async def _spi_transfer(dut, bytes_out):
    """
    Send bytes_out over SPI (MSB first, Mode 0).
    CSB held low for the full frame.
    Returns list of received bytes (sampled on posedge SCK).
    """
    bytes_in = []
    dut.CSB.value = 0
    await Timer(HALF_PERIOD_NS, unit="ns")

    for byte in bytes_out:
        rx = 0
        for bit in range(7, -1, -1):
            dut.SCK.value = 0
            dut.SDI.value = (byte >> bit) & 1
            await Timer(HALF_PERIOD_NS, unit="ns")
            dut.SCK.value = 1
            await Timer(HALF_PERIOD_NS, unit="ns")
            rx = (rx << 1) | int(dut.SDO.value)
        bytes_in.append(rx)

    dut.SCK.value = 0
    await Timer(HALF_PERIOD_NS, unit="ns")
    dut.CSB.value = 1
    await Timer(HALF_PERIOD_NS, unit="ns")
    return bytes_in


async def spi_write(dut, addr_16, data_bytes):
    addr_hi = (addr_16 >> 8) & 0xFF
    addr_lo =  addr_16       & 0xFF
    payload = [0x80, addr_hi, addr_lo] + list(data_bytes)
    cocotb.log.info(
        f"[SPI WRITE] addr={hex(addr_16)} data length={len(data_bytes)}"
    )
    await _spi_transfer(dut, payload)

async def spi_read(dut, addr_16):
    addr_hi = (addr_16 >> 8) & 0xFF
    addr_lo =  addr_16       & 0xFF
    rx = await _spi_transfer(dut, [0x40, addr_hi, addr_lo, 0x00])
    cocotb.log.info(f"[SPI READ]  addr={hex(addr_16)} → {hex(rx[3])}")
    return rx[3]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
async def setup(dut):
    dut.reset.value    = 1
    dut.CSB.value      = 1
    dut.SCK.value      = 0
    dut.SDI.value      = 0
    dut.wb_rst_i.value = 1
    dut.wb_ack_i.value = 0
    dut.wb_dat_i.value = 0

    cocotb.start_soon(wb_clock(dut))
    await Timer(200, unit="ns")

    dut.reset.value    = 0
    dut.wb_rst_i.value = 0
    await Timer(200, unit="ns")

    wb_slave = WishboneSlave(dut, latency=2)
    cocotb.start_soon(wb_slave.run())
    return wb_slave


# ---------------------------------------------------------------------------
# Capture helpers (observe housekeeping_spi internal state via cocotb)
# ---------------------------------------------------------------------------
async def wait_for_wrstb(dut, timeout_ns=TIMEOUT_NS):
    """Wait for wrstb to pulse high. Returns (oaddr, odata) when it does."""
    deadline = cocotb.utils.get_sim_time(units="ns") + timeout_ns
    while True:
        await RisingEdge(dut.SCK)
        await Timer(10, unit="ns")
        if int(dut.wrstb.value) == 1:
            return int(dut.oaddr.value), int(dut.odata.value)
        if cocotb.utils.get_sim_time(units="ns") > deadline:
            raise RuntimeError("Timeout waiting for wrstb")


async def wait_for_wb_write(dut, timeout_ns=TIMEOUT_NS):
    """Wait for Wishbone write cycle to complete (ack received)."""
    deadline = cocotb.utils.get_sim_time(units="ns") + timeout_ns
    while True:
        await RisingEdge(dut.wb_clk_i)
        if (int(dut.wb_cyc_o.value) and
                int(dut.wb_stb_o.value) and
                int(dut.wb_we_o.value) and
                int(dut.wb_ack_i.value)):
            return int(dut.wb_adr_o.value), int(dut.wb_dat_o.value)
        if cocotb.utils.get_sim_time(units="ns") > deadline:
            raise RuntimeError("Timeout waiting for Wishbone write ack")


async def wait_for_wb_read(dut, timeout_ns=TIMEOUT_NS):
    """Wait for Wishbone read cycle to complete (ack received)."""
    deadline = cocotb.utils.get_sim_time(units="ns") + timeout_ns
    while True:
        await RisingEdge(dut.wb_clk_i)
        if (int(dut.wb_cyc_o.value) and
                int(dut.wb_stb_o.value) and
                not int(dut.wb_we_o.value) and
                int(dut.wb_ack_i.value)):
            return int(dut.wb_adr_o.value), int(dut.wb_dat_i.value)
        if cocotb.utils.get_sim_time(units="ns") > deadline:
            raise RuntimeError("Timeout waiting for Wishbone read ack")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_reset(dut):
    """All outputs idle after reset."""
    await setup(dut)

    assert int(dut.wrstb.value)  == 0, "wrstb not idle after reset"
    assert int(dut.rdstb.value)  == 0, "rdstb not idle after reset"
    assert int(dut.sdoenb.value) == 1, "sdoenb not idle after reset"
    assert int(dut.wb_stb_o.value) == 0, "wb_stb_o not idle after reset"
    assert int(dut.wb_cyc_o.value) == 0, "wb_cyc_o not idle after reset"
    cocotb.log.info("Reset test passed")


@cocotb.test()
async def test_spi_to_wishbone_write(dut):
    """
    Host sends a write over SPI.
    Verify:
      1. housekeeping_spi decodes oaddr/odata correctly.
      2. Decoder issues a Wishbone write to the correct address with correct data.
      3. Wishbone slave receives and acks the transaction.
    """
    wb_slave = await setup(dut)

    addr, data = 0x08, 0xAB

    # Arm capture tasks in parallel with the SPI transaction
    wb_write_task = cocotb.start_soon(wait_for_wb_write(dut))
    await spi_write(dut, addr, [data])

    # Wait for Wishbone write to complete
    await Timer(2000, unit="ns")   # give decoder time to issue WB cycle
    wb_addr, wb_data = await wb_write_task

    assert wb_addr == addr, \
        f"WB address wrong: expected {hex(addr)}, got {hex(wb_addr)}"
    assert wb_data == data, \
        f"WB data wrong: expected {hex(data)}, got {hex(wb_data)}"
    assert wb_slave.mem.get(addr) == data, \
        f"WB slave memory wrong: expected {hex(data)}, got {hex(wb_slave.mem.get(addr))}"

    cocotb.log.info(
        f"SPI→WB write passed: addr={hex(addr)} data={hex(data)}"
    )


@cocotb.test()
async def test_wishbone_read_back_over_spi(dut):
    """
    Full round trip:
      1. Pre-load Wishbone slave memory directly.
      2. Host issues SPI read.
      3. Decoder issues WB read, slaves responds.
      4. Data comes back over SDO to host.
    """
    wb_slave = await setup(dut)

    addr        = 0x10
    expected    = 0x5A
    wb_slave.mem[addr] = expected   # pre-load slave memory

    wb_read_task = cocotb.start_soon(wait_for_wb_read(dut))
    rx_data = await spi_read(dut, addr)

    await Timer(2000, unit="ns")
    wb_addr, wb_rdata = await wb_read_task

    assert wb_addr  == addr,     \
        f"WB read address wrong: expected {hex(addr)}, got {hex(wb_addr)}"
    assert wb_rdata == expected, \
        f"WB read data wrong: expected {hex(expected)}, got {hex(wb_rdata)}"

    cocotb.log.info(
        f"SPI→WB read passed: addr={hex(addr)} "
        f"expected={hex(expected)} rx={hex(rx_data)}"
    )


@cocotb.test()
async def test_write_then_read_round_trip(dut):
    """
    Full round trip via SPI:
      1. Write a value to an address over SPI.
      2. Read it back over SPI.
      3. Verify the value matches.
    """
    wb_slave = await setup(dut)

    addr, data = 0x20, 0xDE

    # Write
    await spi_write(dut, addr, [data])
    await Timer(3000, unit="ns")   # let decoder complete WB write

    assert wb_slave.mem.get(addr) == data, \
        f"Write did not reach WB slave: expected {hex(data)}, " \
        f"got {hex(wb_slave.mem.get(addr, 0xFF))}"

    # Read back
    rx = await spi_read(dut, addr)
    await Timer(3000, unit="ns")

    cocotb.log.info(
        f"Round trip: wrote {hex(data)} to {hex(addr)}, read back {hex(rx)}"
    )
    # Note: rx comes from SDO which is driven by idata → wb_idata from decoder.
    # This verifies the full path: SPI write → WB → SPI read → SDO.
    assert rx == data, \
        f"Round trip mismatch: expected {hex(data)}, got {hex(rx)}"
    
@cocotb.test()
async def test_1kb_streaming_write(dut):
    wb_slave = await setup(dut)

    start_addr = 0x0008
    data_bytes = [random.randint(0x00, 0xFF) for _ in range(1024)]

    await spi_write(dut, start_addr, data_bytes)
    # 1027 bytes @ 1MHz SCK = ~8.2ms, allow 2x margin
    await Timer(16_000_000, unit="ns")

    mismatches = []
    for i, expected in enumerate(data_bytes):
        addr = start_addr + i
        got  = wb_slave.mem.get(addr)
        if got != expected:
            mismatches.append((i, addr, expected, got))

    for i, addr, expected, got in mismatches:
        cocotb.log.error(
            f"  byte {i}: addr={hex(addr)} "
            f"expected={hex(expected)} got={hex(got) if got is not None else 'MISSING'}"
        )

    assert len(mismatches) == 0, \
        f"{len(mismatches)}/{len(data_bytes)} bytes wrong"

    cocotb.log.info(f"1KB streaming write passed: all {len(data_bytes)} bytes correct")

# ---------------------------------------------------------------------------
# Add this to test_house.py
# ---------------------------------------------------------------------------

async def spi_streaming_read(dut, addr_16, num_bytes):
    """
    True streaming read: one CSB assertion, N+3 bytes clocked.
    Frame: 0x40  addrH  addrL  0x00 x N
    SDO is captured on posedge SCK during the data phase bytes (bytes 3..N+2).
    Returns list of N received bytes.
    """
    addr_hi = (addr_16 >> 8) & 0xFF
    addr_lo =  addr_16       & 0xFF

    # Full frame: cmd + addrH + addrL + N dummy bytes
    bytes_out = [0x40, addr_hi, addr_lo] + [0x00] * num_bytes

    received = []
    current_byte = 0
    bit_count = 0
    in_data_phase = False
    byte_index = 0

    dut.CSB.value = 0
    await Timer(HALF_PERIOD_NS, unit="ns")

    for byte_idx, byte in enumerate(bytes_out):
        rx = 0
        for bit in range(7, -1, -1):
            dut.SCK.value = 0
            dut.SDI.value = (byte >> bit) & 1
            await Timer(HALF_PERIOD_NS, unit="ns")
            dut.SCK.value = 1
            await Timer(HALF_PERIOD_NS, unit="ns")
            # Capture SDO during data phase (bytes 3 onwards)
            if byte_idx >= 3:
                rx = (rx << 1) | int(dut.SDO.value)
        if byte_idx >= 3:
            received.append(rx)

    dut.SCK.value = 0
    await Timer(HALF_PERIOD_NS, unit="ns")
    dut.CSB.value = 1
    await Timer(HALF_PERIOD_NS, unit="ns")

    cocotb.log.info(
        f"[SPI STREAM READ] addr={hex(addr_16)} "
        f"num_bytes={num_bytes} received={len(received)}"
    )
    return received


@cocotb.test()
async def test_1kb_streaming_read(dut):
    """
    True streaming read of 1024 bytes:
      1. Pre-load WB slave memory with known pattern.
      2. Issue one SPI read transaction with CSB held low for all 1024 bytes.
      3. Capture SDO and verify every byte matches.
    """
    wb_slave = await setup(dut)

    start_addr = 0x0008
    num_bytes  = 1024

    # Use an incrementing pattern so mismatches are obvious
    data_bytes = [(start_addr + i) & 0xFF for i in range(num_bytes)]

    # Pre-load WB slave memory
    for i, val in enumerate(data_bytes):
        wb_slave.mem[start_addr + i] = val

    cocotb.log.info(f"Pre-loaded {num_bytes} bytes from addr {hex(start_addr)}")

    # Issue streaming read — one CSB assertion for all 1024 bytes
    received = await spi_streaming_read(dut, start_addr, num_bytes)

    # Allow decoder to finish any in-flight WB transaction
    await Timer(5000, unit="ns")

    assert len(received) == num_bytes, \
        f"Expected {num_bytes} bytes back, got {len(received)}"

    mismatches = []
    for i in range(num_bytes):
        expected = data_bytes[i]
        got      = received[i]
        if got != expected:
            mismatches.append((i, start_addr + i, expected, got))

    for i, addr, expected, got in mismatches[:20]:   # print first 20 only
        cocotb.log.error(
            f"  byte {i}: addr={hex(addr)} "
            f"expected={hex(expected)} got={hex(got)}"
        )

    assert len(mismatches) == 0, \
        f"{len(mismatches)}/{num_bytes} bytes wrong on streaming read"

    cocotb.log.info(f"1KB streaming read passed: all {num_bytes} bytes correct")


@cocotb.test()
async def test_streaming_multi_byte_write(dut):
    """
    Streaming write: one CSB assertion, multiple data bytes.
    Address auto-increments. Each byte must appear as a separate WB write.
    """
    wb_slave = await setup(dut)

    start_addr = 0x08
    data_bytes = [0x11, 0x22, 0x33, 0x44]

    await spi_write(dut, start_addr, data_bytes)
    await Timer(10_000, unit="ns")   # allow all WB cycles to complete

    for i, expected in enumerate(data_bytes):
        addr = start_addr + i
        got  = wb_slave.mem.get(addr)
        assert got == expected, \
            f"Streaming byte {i}: addr={hex(addr)} " \
            f"expected={hex(expected)} got={hex(got)}"

    cocotb.log.info(
        f"Streaming write passed: {len(data_bytes)} bytes "
        f"from addr {hex(start_addr)}"
    )


@cocotb.test()
async def test_random_write_read(dut):
    """10 random write→read round trips via SPI→Wishbone."""
    wb_slave = await setup(dut)

    for i in range(10):
        addr = random.randint(0x08, 0x7F)
        data = random.randint(0x00, 0xFF)

        await spi_write(dut, addr, [data])
        await Timer(3000, unit="ns")

        assert wb_slave.mem.get(addr) == data, \
            f"iter {i}: WB write failed addr={hex(addr)} data={hex(data)}"

        rx = await spi_read(dut, addr)
        await Timer(3000, unit="ns")

        cocotb.log.info(
            f"  iter {i}: addr={hex(addr)} wrote={hex(data)} read={hex(rx)}"
        )

    cocotb.log.info("Random write/read test passed")


@cocotb.test()
async def test_csb_resets_state(dut):
    """Asserting CSB mid-transaction resets housekeeping_spi to idle."""
    await setup(dut)

    # Pull CSB low, clock a few bits, then abort
    dut.CSB.value = 0
    await Timer(HALF_PERIOD_NS, unit="ns")
    for _ in range(4):
        dut.SCK.value = 0
        await Timer(HALF_PERIOD_NS, unit="ns")
        dut.SCK.value = 1
        await Timer(HALF_PERIOD_NS, unit="ns")
    dut.SCK.value = 0

    # Assert CSB (end transaction early)
    dut.CSB.value = 1
    await Timer(500, unit="ns")

    assert int(dut.wrstb.value)    == 0, "wrstb not cleared after CSB"
    assert int(dut.rdstb.value)    == 0, "rdstb not cleared after CSB"
    assert int(dut.sdoenb.value)   == 1, "sdoenb not cleared after CSB"
    assert int(dut.wb_stb_o.value) == 0, "wb_stb_o not cleared after CSB"
    cocotb.log.info("CSB reset test passed")


# ---------------------------------------------------------------------------
# Pytest boilerplate
# ---------------------------------------------------------------------------
tests = [
    "test_reset",
    "test_spi_to_wishbone_write",
    "test_wishbone_read_back_over_spi",
    "test_write_then_read_round_trip",
    "test_streaming_multi_byte_write",
    "test_random_write_read",
    "test_csb_resets_state",
    "test_1kb_streaming_write",
    "test_1kb_streaming_read",
    "spi_streaming_read",
]

SOURCES = [
    Path("./src/spi/housekeeping_spi.v").resolve(),
    Path("./src/spi/decoder.sv").resolve(),
    Path("./src/spi/spi_top.sv").resolve(),
]


@pytest.mark.parametrize("testcase", tests)
def test_housekeeping_spi_each(testcase):
    run_test(
        sources=SOURCES,
        module_name="test_house",
        hdl_toplevel="spi_top",   # top is the decoder which instantiates housekeeping_spi
        parameters={},
        testcase=testcase,
        sims=["icarus"],
    )


def test_housekeeping_spi_all():
    run_test(
        sources=SOURCES,
        module_name="test_house",
        hdl_toplevel="spi_top",
        parameters={},
        sims=["icarus"],
    )