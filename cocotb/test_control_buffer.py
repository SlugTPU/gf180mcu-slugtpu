import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles, Event
from pathlib import Path
from shared import reset_sequence, clock_start, random_binary_driver, handshake
from runner import run_test
from collections import deque
import random

DRAM_WIDTH = 64
CONTROL_WIDTH = 8
DRAM_ADDR_WIDTH = 12

# Number of CONTROL_WIDTH-wide words packed into one DRAM_WIDTH word
PACK_FACTOR = DRAM_WIDTH // CONTROL_WIDTH  # 8


class ControlBufferModel:
    """
    Reference model for control_buffer.

    The DUT accepts 64-bit DRAM words and streams them out as 8-bit
    control words in order (LSB-first, i.e. byte 0 first).
    Tracks expected pc_out against the pc_in that was loaded most
    recently before each write burst.
    """

    def __init__(self):
        # Expected output bytes, in order
        self.expected_bytes: deque[int] = deque()
        # The pc that was latched when the last pc_valid_i pulse fired
        self._latched_pc: int | None = None
        # pc we expect to see at pc_out when the next output fires
        self._expected_pc: int | None = None

    def load_pc(self, pc: int):
        """Called when pc_valid_i is asserted."""
        self._latched_pc = pc
        cocotb.log.info(f"[Model] PC latched: 0x{pc:03X}")

    def consume(self, dut):
        """Called on every wr_valid_i / wr_ready_o handshake."""
        wr_data = dut.wr_data_i.value.to_unsigned()
        # Capture pc at the time of the write (the DUT should expose it)
        if self._latched_pc is not None and self._expected_pc is None:
            self._expected_pc = self._latched_pc

        for byte_idx in range(PACK_FACTOR):
            byte_val = (wr_data >> (byte_idx * CONTROL_WIDTH)) & 0xFF
            self.expected_bytes.append(byte_val)

        cocotb.log.info(
            f"[Model] Consumed DRAM word 0x{wr_data:016X} → "
            f"{PACK_FACTOR} bytes queued (total pending: {len(self.expected_bytes)})"
        )

    def produce(self, dut):
        """Called on every rd_valid_o / rd_ready_i handshake."""
        got = dut.rd_data_o.value.to_unsigned()
        expected = self.expected_bytes.popleft()
        cocotb.log.info(
            f"[Model] Output byte: got=0x{got:02X}  expected=0x{expected:02X}  "
            f"(remaining: {len(self.expected_bytes)})"
        )
        assert got == expected, (
            f"rd_data_o mismatch: got 0x{got:02X}, expected 0x{expected:02X}"
        )

        # Optionally check pc_out if we have an expectation
        if self._expected_pc is not None:
            pc_got = dut.pc_out.value.to_unsigned()
            assert pc_got == self._expected_pc, (
                f"pc_out mismatch: got 0x{pc_got:03X}, expected 0x{self._expected_pc:03X}"
            )


class ModelRunner:
    """
    Monitors handshakes on the DUT and drives the reference model.
    Mirrors the ModelRunner in test_fifo.py.
    """

    def __init__(self, dut, model: ControlBufferModel):
        self.model = model
        self.dut = dut

    def start(self):
        cocotb.start_soon(self._run_input())
        cocotb.start_soon(self._run_output())

    async def _run_input(self):
        """Watch wr_valid_i / wr_ready_o handshakes."""
        while True:
            await handshake(
                self.dut.clk_i,
                self.dut.rst_i,
                self.dut.wr_ready_o,
                self.dut.wr_valid_i,
            )
            self.model.consume(self.dut)

    async def _run_output(self):
        """Watch rd_valid_o / rd_ready_i handshakes."""
        while True:
            await handshake(
                self.dut.clk_i,
                self.dut.rst_i,
                self.dut.rd_ready_i,
                self.dut.rd_valid_o,
            )
            self.model.produce(self.dut)


class StreamIOModel:
    """
    Drives stimulus and checks completion, mirroring test_fifo StreamIOModel.

    Sends `num_dram_words` 64-bit words into the DUT and waits until
    num_dram_words * PACK_FACTOR 8-bit words come back out.
    """

    def __init__(self, dut, num_dram_words: int, in_pressure: bool, out_pressure: bool):
        self.dut = dut
        self.num_dram_words = num_dram_words
        self.num_output_bytes = num_dram_words * PACK_FACTOR
        self.in_pressure = in_pressure
        self.out_pressure = out_pressure
        self.n_wr = 0   # DRAM words written in
        self.n_rd = 0   # control bytes read out

    async def input_run(self):
        dut = self.dut
        stop_event = Event()

        await FallingEdge(dut.rst_i)
        dut.wr_valid_i.value = 0
        dut.wr_data_i.value = 0
        dut.pc_valid_i.value = 0
        dut.pc_in.value = 0
        await FallingEdge(dut.clk_i)

        if self.in_pressure:
            cocotb.start_soon(
                random_binary_driver(dut.clk_i, dut.wr_valid_i, prob=0.5, max_hold=10, stop_event=stop_event)
            )
        else:
            dut.wr_valid_i.value = 1

        word_idx = 0
        while word_idx < self.num_dram_words:
            # Build a deterministic DRAM word so the model can verify bytes
            dram_word = 0
            for b in range(PACK_FACTOR):
                byte_val = (word_idx * PACK_FACTOR + b) & 0xFF
                dram_word |= byte_val << (b * CONTROL_WIDTH)

            dut.wr_data_i.value = dram_word
            # Pulse pc_in alongside first word of each burst (one pc per word here)
            dut.pc_in.value = word_idx & ((1 << DRAM_ADDR_WIDTH) - 1)
            dut.pc_valid_i.value = 1

            await RisingEdge(dut.clk_i)
            if dut.wr_ready_o.value == 1 and dut.wr_valid_i.value == 1:
                self.n_wr += 1
                cocotb.log.info(f"[IO] DRAM word {self.n_wr}/{self.num_dram_words} accepted")
                word_idx += 1

            dut.pc_valid_i.value = 0
            await FallingEdge(dut.clk_i)

        stop_event.set()
        dut.wr_valid_i.value = 0
        dut.pc_valid_i.value = 0

    async def output_run(self):
        dut = self.dut
        stop_event = Event()
        
        await FallingEdge(dut.rst_i)
        dut.rd_ready_i.value = 0
        await FallingEdge(dut.clk_i)

        if self.out_pressure:
            cocotb.start_soon(
                random_binary_driver(dut.clk_i, dut.rd_ready_i, prob=0.5, max_hold=10, stop_event=stop_event)
            )
        else:
            dut.rd_ready_i.value = 1

        while self.n_rd < self.num_output_bytes:
            await RisingEdge(dut.clk_i)
            if dut.rd_valid_o.value == 1 and dut.rd_ready_i.value == 1:
                self.n_rd += 1
                cocotb.log.info(f"[IO] Control byte {self.n_rd}/{self.num_output_bytes} received")
            await FallingEdge(dut.clk_i)

        stop_event.set()
        dut.rd_ready_i.value = 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@cocotb.test()
async def reset_test(dut):
    """Check that outputs are inactive after reset."""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    dut.wr_valid_i.value = 0
    dut.wr_data_i.value = 0
    dut.pc_valid_i.value = 0
    dut.pc_in.value = 0
    dut.rd_ready_i.value = 0

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)

    # After reset: rd_valid_o should be de-asserted
    await FallingEdge(clk_i)
    assert dut.rd_valid_o.value == 0, "rd_valid_o should be 0 after reset"


@cocotb.test()
async def single_word_test(dut):
    """
    Write one 64-bit DRAM word and verify all 8 output bytes appear
    in the correct order.
    """
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    dut.wr_valid_i.value = 0
    dut.wr_data_i.value = 0
    dut.pc_valid_i.value = 0
    dut.pc_in.value = 0
    dut.rd_ready_i.value = 0

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    m = ControlBufferModel()
    r = ModelRunner(dut, m)
    r.start()

    await FallingEdge(rst_i)
    await FallingEdge(clk_i)

    # Set PC
    dut.pc_in.value = 0xABC
    dut.pc_valid_i.value = 1
    await RisingEdge(clk_i)
    await FallingEdge(clk_i)
    dut.pc_valid_i.value = 0

    # Write one DRAM word: bytes 0x00..0x07
    dram_word = 0x0706050403020100
    dut.wr_data_i.value = dram_word
    dut.wr_valid_i.value = 1
    dut.rd_ready_i.value = 0

    # Wait for acceptance
    while True:
        # await RisingEdge(clk_i)
        if dut.wr_ready_o.value == 1:
            await FallingEdge(clk_i)
            break
        await FallingEdge(clk_i)

    dut.wr_valid_i.value = 0

    # Now drain all 8 bytes
    dut.rd_ready_i.value = 1
    for _ in range(PACK_FACTOR + 4):   # a few extra cycles of margin
        # await RisingEdge(clk_i)
        await FallingEdge(clk_i)

    dut.rd_ready_i.value = 0
    await FallingEdge(clk_i)

    assert len(m.expected_bytes) == 0, (
        f"Model still has {len(m.expected_bytes)} unread bytes after drain"
    )


@cocotb.test()
async def fill_and_drain_test(dut):
    """
    Fill the internal FIFO to capacity (DEPTH_LOG2_P controls depth),
    then drain it completely.  Mirrors fifo_simple_test.
    """
    depth_log2_p = dut.DEPTH_LOG2_P.value.to_unsigned()
    fifo_depth = 2 ** depth_log2_p          # depth in 8-bit slots
    num_dram_words = fifo_depth // PACK_FACTOR  # DRAM words to fill it

    clk_i = dut.clk_i
    rst_i = dut.rst_i
    dut.wr_valid_i.value = 0
    dut.wr_data_i.value = 0
    dut.pc_valid_i.value = 0
    dut.pc_in.value = 0
    dut.rd_ready_i.value = 0

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    m = ControlBufferModel()
    r = ModelRunner(dut, m)
    r.start()

    await FallingEdge(rst_i)
    await FallingEdge(clk_i)

    dut.pc_in.value = 0x001
    dut.pc_valid_i.value = 1
    await RisingEdge(clk_i)
    await FallingEdge(clk_i)
    dut.pc_valid_i.value = 0

    # --- Fill phase ---
    dut.rd_ready_i.value = 0
    for i in range(num_dram_words):
        dram_word = 0
        for b in range(PACK_FACTOR):
            byte_val = (i * PACK_FACTOR + b) & 0xFF
            dram_word |= byte_val << (b * CONTROL_WIDTH)

        dut.wr_data_i.value = dram_word
        dut.wr_valid_i.value = 1

        while True:
            await RisingEdge(clk_i)
            if dut.wr_ready_o.value == 1:
                await FallingEdge(clk_i)
                break
            await FallingEdge(clk_i)

    dut.wr_valid_i.value = 0

    # --- Drain phase ---
    dut.rd_ready_i.value = 1
    for _ in range(fifo_depth + 4):
        await RisingEdge(clk_i)
        await FallingEdge(clk_i)

    dut.rd_ready_i.value = 0
    await FallingEdge(clk_i)

    assert len(m.expected_bytes) == 0, (
        f"Model still has {len(m.expected_bytes)} unread bytes after drain"
    )


@cocotb.test()
@cocotb.parametrize(with_pressure=[False, True])
async def random_stream_test(dut, with_pressure):
    """
    Stream 10 DRAM words through the DUT with optional backpressure on
    both input and output sides.  Mirrors fifo_random_stream_test.
    """
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    NUM_DRAM_WORDS = 10

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    m = ControlBufferModel()
    r = ModelRunner(dut, m)
    iom = StreamIOModel(dut, NUM_DRAM_WORDS, with_pressure, with_pressure)

    r.start()
    task_input = cocotb.start_soon(iom.input_run())
    task_output = cocotb.start_soon(iom.output_run())

    await task_output.complete
    await FallingEdge(clk_i)

    assert len(m.expected_bytes) == 0, (
        f"Model still has {len(m.expected_bytes)} unread bytes at end of stream test"
    )


@cocotb.test()
async def pc_tracking_test(dut):
    """
    Verify that pc_out reflects the pc_in value that was loaded before
    each DRAM write.
    """
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)
    await FallingEdge(clk_i)

    dut.rd_ready_i.value = 1
    dut.wr_valid_i.value = 0

    for trial in range(4):
        pc_val = (trial + 1) * 0x10
        dram_word = random.randint(0, (1 << DRAM_WIDTH) - 1)

        # Load PC
        dut.pc_in.value = pc_val
        dut.pc_valid_i.value = 1
        await RisingEdge(clk_i)
        await FallingEdge(clk_i)
        dut.pc_valid_i.value = 0

        # Write DRAM word
        dut.wr_data_i.value = dram_word
        dut.wr_valid_i.value = 1
        while True:
            await RisingEdge(clk_i)
            if dut.wr_ready_o.value == 1:
                await FallingEdge(clk_i)
                break
            await FallingEdge(clk_i)
        dut.wr_valid_i.value = 0

        # Drain the 8 bytes and check pc_out stays stable
        for _ in range(PACK_FACTOR + 2):
            await RisingEdge(clk_i)
            if dut.rd_valid_o.value == 1 and dut.rd_ready_i.value == 1:
                pc_got = dut.pc_out.value.to_unsigned()
                assert pc_got ==  pc_val +1, (
                    f"[Trial {trial}] pc_out=0x{pc_got:03X}, expected 0x{pc_val:03X}"
                )
            await FallingEdge(clk_i)

    dut.rd_ready_i.value = 0


# ---------------------------------------------------------------------------
# Pytest entry points (mirrors test_fifo.py style)
# ---------------------------------------------------------------------------

proj_path = Path("./src").resolve()
sources = [proj_path / "control" / "control_buffer.sv"]

tests = [
    "reset_test",
    "single_word_test",
    "fill_and_drain_test",
    "random_stream_test",
    # "pc_tracking_test",
]


@pytest.mark.parametrize("CONTROL_WIDTH", [8])
@pytest.mark.parametrize("testcase", tests)
def test_control_buffer_each(CONTROL_WIDTH, testcase):
    """Run each test independently so failures don't block others."""
    run_test(
        parameters={"CONTROL_WIDTH": CONTROL_WIDTH},
        sources=sources,
        module_name="test_control_buffer",
        hdl_toplevel="control_buffer",
        testcase=testcase,
        sims=['icarus'],
    )


@pytest.mark.parametrize("CONTROL_WIDTH", [8])
def test_control_buffer_all(CONTROL_WIDTH):
    """Run all tests sequentially as one simulation."""
    run_test(
        parameters={"CONTROL_WIDTH": CONTROL_WIDTH},
        sources=sources,
        module_name="test_control_buffer",
        hdl_toplevel="control_buffer",
        sims=['icarus'],
    )