import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, Event
from pathlib import Path
from shared import reset_sequence, clock_start, random_binary_driver, handshake
from runner import run_test
from collections import deque
import random

# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------

class ControlSramModel():
    """
    Behavioural model of a read-priority FIFO.

    Mirrors the DUT: every cycle that a valid write handshake fires, the
    word is pushed onto the tail; every cycle that a valid read handshake
    fires, the word at the head is compared against the DUT output.
    Read takes priority, so if both happen in the same cycle the read
    drains the older entry before the write is counted.
    """
    def __init__(self):
        self.q = deque()

    def write(self, data: int):
        self.q.append(data)
        # cocotb.log.info(f"[MODEL] write {data:#04x}  depth={len(self.q)}")

    def read(self, dut) -> None:
        rd_data_o = dut.rd_data_o
        got = rd_data_o.value.to_unsigned()
        expected = self.q.popleft()
        # cocotb.log.info(f"[MODEL] read  got={got:#04x}  expected={expected:#04x}  depth={len(self.q)}")
        assert got == expected, (
            f"rd_data_o mismatch: got {got:#04x}, expected {expected:#04x}"
        )

    def depth(self) -> int:
        return len(self.q)


class ControlSramModelRunner():
    """
    Attaches the model to the DUT by monitoring handshakes.

    Write handshake : valid_i & ready_o on rising edge
    Read  handshake : valid_o & ready_i on rising edge
    Read priority   : if both fire together, read is processed first.
    """
    def __init__(self, dut, model: ControlSramModel):
        self.dut   = dut
        self.model = model

    def start(self):
        cocotb.start_soon(self._monitor())

    async def _monitor(self):
        clk_i   = self.dut.clk_i
        rst_i   = self.dut.rst_i
        valid_i = self.dut.valid_i
        ready_o = self.dut.ready_o
        valid_o = self.dut.valid_o
        ready_i = self.dut.ready_i

        await FallingEdge(rst_i)

        while True:
            await RisingEdge(clk_i)
            wr_fire = (valid_i.value == 1 and ready_o.value == 1)
            rd_fire = (valid_o.value == 1 and ready_i.value == 1)

            # read priority: drain before accepting new write
            if rd_fire:
                self.model.read(self.dut)
            if wr_fire:
                self.model.write(self.dut.wr_data_i.value.to_unsigned())

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def reset_test(dut):
    """After reset: ready_o/valid_o/is_full_o must be in sane initial states."""
    clk_i    = dut.clk_i
    rst_i    = dut.rst_i
    valid_o  = dut.valid_o
    is_full_o = dut.is_full_o

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)
    await FallingEdge(clk_i)

    assert valid_o.value   == 0, "valid_o should be 0 after reset (FIFO empty)"
    assert is_full_o.value == 0, "is_full_o should be 0 after reset"
    cocotb.log.info("[RESET TEST] passed")


@cocotb.test()
async def write_read_simple_test(dut):
    """
    Fill the FIFO completely, then drain it.

    Assumes the synthesis parameter exposes a DEPTH_P or similar;
    if not, we probe is_full_o to discover the depth dynamically.
    """
    clk_i     = dut.clk_i
    rst_i     = dut.rst_i
    valid_i   = dut.valid_i
    ready_o   = dut.ready_o
    ready_i   = dut.ready_i
    valid_o   = dut.valid_o
    is_full_o = dut.is_full_o
    wr_data_i = dut.wr_data_i

    model  = ControlSramModel()
    runner = ControlSramModelRunner(dut, model)

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    runner.start()

    await FallingEdge(rst_i)
    valid_i.value = 1
    ready_i.value = 0

    # ---- Fill phase --------------------------------------------------------
    value = 0
    while True:
        wr_data_i.value = value & 0xFF
        await RisingEdge(clk_i)
        if ready_o.value == 1:
            # model runner already called model.write() via its monitor;
            # just advance local counter
            # cocotb.log.info(f"[FILL] wrote {value:#04x}")
            value += 1
        await FallingEdge(clk_i)

        if is_full_o.value == 1:
            cocotb.log.info(f"[FILL] FIFO full at depth {value}")
            break

    # Stop writing
    valid_i.value = 0
    await FallingEdge(clk_i)

    # ---- Drain phase -------------------------------------------------------
    depth = model.depth()
    ready_i.value = 1

    for _ in range(depth+1):
        await RisingEdge(clk_i)
        await FallingEdge(clk_i)

    ready_i.value = 0
    await FallingEdge(clk_i)

    assert model.depth() == 0, "Model queue not empty after drain"
    cocotb.log.info("[SIMPLE TEST] passed")


@cocotb.test()
async def read_priority_test(dut):
    """
    Write one entry, then assert both valid_i and ready_i simultaneously.
    The read-priority design must present the correct output *this* cycle
    and not double-consume the entry.
    """
    clk_i     = dut.clk_i
    rst_i     = dut.rst_i
    valid_i   = dut.valid_i
    ready_o   = dut.ready_o
    ready_i   = dut.ready_i
    valid_o   = dut.valid_o
    wr_data_i = dut.wr_data_i
    rd_data_o = dut.rd_data_o
 
    SENTINEL = 0xA5
 
    model  = ControlSramModel()
    runner = ControlSramModelRunner(dut, model)
 
    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    runner.start()
 
    await FallingEdge(rst_i)
 
    # Preload a few words so the FIFO is non-empty before the conflict cycle.
    # SENTINEL is the first word in; it must be the one that comes out.
    PRELOAD = [SENTINEL, 0x11, 0x22]
    valid_i.value = 1
    ready_i.value = 0
    for word in PRELOAD:
        wr_data_i.value = word
        await RisingEdge(clk_i)
        await FallingEdge(clk_i)
    valid_i.value = 0
 
    await FallingEdge(clk_i)
 
    # Conflict cycle: assert ready_i and valid_i simultaneously with a
    # different payload. Read must win — SENTINEL comes out, 0x5A is deferred.
    valid_i.value   = 1
    ready_i.value   = 1
    wr_data_i.value = 0x5A
 
    await FallingEdge(clk_i)
 
    got = rd_data_o.value.to_unsigned()
    assert got == SENTINEL, (
        f"read_priority_test: expected {SENTINEL:#04x}, got {got:#04x}"
    )
 
    valid_i.value = 0
    ready_i.value = 0
    await FallingEdge(clk_i)
    cocotb.log.info("[READ PRIORITY TEST] passed")


@cocotb.test()
async def is_full_deasserts_on_read_test(dut):
    """
    Once is_full_o is asserted, a successful read must deassert it within
    one cycle (or at most a couple of pipeline stages for registered outputs).
    """
    clk_i     = dut.clk_i
    rst_i     = dut.rst_i
    valid_i   = dut.valid_i
    ready_o   = dut.ready_o
    ready_i   = dut.ready_i
    valid_o   = dut.valid_o
    is_full_o = dut.is_full_o
    wr_data_i = dut.wr_data_i

    model  = ControlSramModel()
    runner = ControlSramModelRunner(dut, model)

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    runner.start()

    await FallingEdge(rst_i)
    valid_i.value = 1
    ready_i.value = 0

    # Fill until full
    value = 0
    while True:
        wr_data_i.value = value & 0xFF
        await RisingEdge(clk_i)
        if ready_o.value == 1:
            value += 1
        await FallingEdge(clk_i)
        if is_full_o.value == 1:
            break

    valid_i.value = 0
    cocotb.log.info("[IS_FULL TEST] FIFO full")

    # Issue one read
    ready_i.value = 1
    await RisingEdge(clk_i)
    await FallingEdge(clk_i)
    ready_i.value = 0

    # Allow a couple of cycles for registered is_full_o update
    for cycle in range(4):
        await RisingEdge(clk_i)
        if is_full_o.value == 0:
            cocotb.log.info(f"[IS_FULL TEST] deasserted after {cycle+1} cycle(s)")
            break
        await FallingEdge(clk_i)
    else:
        assert False, "is_full_o did not deassert after reading from a full FIFO"

    cocotb.log.info("[IS_FULL TEST] passed")

@cocotb.test()
async def no_spurious_output_when_empty_test(dut):
    """
    valid_o must remain 0 while the FIFO is empty and no writes are issued.
    """
    clk_i   = dut.clk_i
    rst_i   = dut.rst_i
    valid_i = dut.valid_i
    valid_o = dut.valid_o
    ready_i = dut.ready_i

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)
    valid_i.value = 0
    ready_i.value = 1   # downstream is always ready

    for cycle in range(16):
        await RisingEdge(clk_i)
        assert valid_o.value == 0, (
            f"valid_o spuriously asserted on cycle {cycle} with empty FIFO"
        )
        await FallingEdge(clk_i)

    cocotb.log.info("[SPURIOUS OUTPUT TEST] passed")

@cocotb.test()
async def alternating_single_read_write_test(dut):
    """
    Interleave individual writes and reads one at a time.

    Pattern: write X₀ → read X₀ → write X₁ → read X₁ → … for N rounds.

    Checks:
      - rd_data_o matches the written value on every read
      - valid_o is asserted exactly when the FIFO is non-empty
      - valid_o deasserts after draining the single entry
      - is_full_o never asserts (depth never exceeds 1)
    """
    clk_i     = dut.clk_i
    rst_i     = dut.rst_i
    valid_i   = dut.valid_i
    ready_o   = dut.ready_o
    ready_i   = dut.ready_i
    valid_o   = dut.valid_o
    is_full_o = dut.is_full_o
    wr_data_i = dut.wr_data_i
    rd_data_o = dut.rd_data_o

    N_ROUNDS = 16

    model  = ControlSramModel()
    runner = ControlSramModelRunner(dut, model)

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    runner.start()

    await FallingEdge(rst_i)
    valid_i.value = 0
    ready_i.value = 0
    await FallingEdge(clk_i)

    for i in range(N_ROUNDS):
        payload = (i * 0x13 + 0x55) & 0xFF   # deterministic but varied

        # ---- Write one entry -----------------------------------------------
        valid_i.value   = 1
        ready_i.value   = 0
        wr_data_i.value = payload

        # Spin until the write handshake fires (ready_o may not be immediate)
        while True:
            await RisingEdge(clk_i)
            fired = (ready_o.value == 1)
            await FallingEdge(clk_i)
            if fired:
                break

        valid_i.value = 0

        # is_full_o must not assert: depth is exactly 1
        assert is_full_o.value == 0, (
            f"Round {i}: is_full_o asserted after a single write — "
            "FIFO depth should be 1, not full"
        )
        # ---- Read that entry back ------------------------------------------
        ready_i.value = 1
        # Give the FIFO one idle cycle to update valid_o
        await RisingEdge(clk_i)
        await FallingEdge(clk_i)

        # valid_o must be high — there is one entry waiting
        assert valid_o.value == 1, (
            f"Round {i}: valid_o not asserted after writing {payload:#04x}"
        )

        # Spin until the read handshake fires
        while True:
            await RisingEdge(clk_i)
            rd_fired = (valid_o.value == 1 and ready_i.value == 1)
            if rd_fired:
                got = rd_data_o.value.to_unsigned()
                assert got == payload, (
                    f"Round {i}: rd_data_o mismatch — "
                    f"got {got:#04x}, expected {payload:#04x}"
                )
            await FallingEdge(clk_i)
            if rd_fired:
                break

        ready_i.value = 0

        # Allow one cycle for the FIFO to update its empty status
        await RisingEdge(clk_i)
        await FallingEdge(clk_i)

        # valid_o must be low again — FIFO is empty
        assert valid_o.value == 0, (
            f"Round {i}: valid_o still asserted after draining the FIFO"
        )

        cocotb.log.info(
            f"[ALT RW TEST] round {i:02d}  payload={payload:#04x}  OK"
        )

    # Model queue must be empty at the end
    assert model.depth() == 0, (
        f"Model not empty after {N_ROUNDS} alternating rounds — "
        f"depth={model.depth()}"
    )
    cocotb.log.info("[ALT RW TEST] passed")


# ---------------------------------------------------------------------------
# Pytest entry points
# ---------------------------------------------------------------------------

proj_path = Path("./src").resolve()
sources   = [proj_path / "control" / "control_sram.sv",
             proj_path / "sram/sram_1x256.sv",
             "./ip/gf180mcu_ocd_ip_sram.git/cells/gf180mcu_ocd_ip_sram__sram256x8m8wm1/gf180mcu_ocd_ip_sram__sram256x8m8wm1.v",]

TESTS = [
    "reset_test",
    "write_read_simple_test",
    "read_priority_test",
    "is_full_deasserts_on_read_test",
    "no_spurious_output_when_empty_test",
    "alternating_single_read_write_test",
]

@pytest.mark.parametrize("testcase", TESTS)
def test_control_sram_each(testcase):
    """Run each cocotb test independently; simulation continues on failure."""
    run_test(
        parameters={"sram_width_p": 8},
        sources=sources,
        module_name="test_control_sram",
        hdl_toplevel="control_sram",
        testcase=testcase,
        sims=['icarus'],
    )

def test_control_sram_all():
    """Run all cocotb tests in a single simulation."""
    run_test(
        parameters={"sram_width_p": 8},
        sources=sources,
        module_name="test_control_sram",
        hdl_toplevel="control_sram",
        sims=['icarus'],
    )