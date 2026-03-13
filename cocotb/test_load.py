"""
test_load.py — cocotb testbench for the loader module (Icarus-compatible flat ports).

Timing discipline:
  - Driving (dut.x.value = y) is only done BEFORE ReadOnly() in the same time step,
    i.e. right after a RisingEdge or FallingEdge, never while parked at ReadOnly().
  - Checking (reading dut outputs) is always done at ReadOnly().
  - run_one_tile EXIT contract: returns immediately after the final ReadOnly() check,
    but the CALLER must not attempt to drive signals without first awaiting a new edge.
    To make this safe, every call site that drives signals after run_one_tile either:
      (a) calls run_one_tile again (which starts with a RisingEdge internally), or
      (b) awaits a RisingEdge before any driving.
  - reset_sequence EXIT: returns after RisingEdge (NOT ReadOnly), so callers can drive.
"""

from __future__ import annotations

import random
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly, with_timeout

from runner import run_test


# ──────────────────────────────────────────────────────────────────────────────
# Parameters — must match RTL defaults
# ──────────────────────────────────────────────────────────────────────────────
N          = 4
DATA_WIDTH = 8
K          = 4
CLK_PERIOD = 10  # ns


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def assert_resolvable(sig, name="signal"):
    val = sig.value
    if val.is_resolvable is False:
        raise AssertionError(f"{name} is not resolvable (X/Z): {val}")


async def clock_start(dut):
    cocotb.start_soon(Clock(dut.clk_i, CLK_PERIOD, unit="ns").start())


async def reset_sequence(dut, cycles=4):
    """
    Hold reset for `cycles` clocks, release, clock once more.
    EXIT: returns right after the final RisingEdge — safe to drive signals.
    """
    dut.rst_i.value     = 1
    dut.start_i.value   = 0
    dut.buf_sel_i.value = 0
    dut.A_flat.value    = 0
    dut.B_flat.value    = 0
    for _ in range(cycles):
        await RisingEdge(dut.clk_i)
    dut.rst_i.value = 0
    await RisingEdge(dut.clk_i)   # one clean IDLE cycle; returns here (drive-safe)


# ── Data generators ───────────────────────────────────────────────────────────

def make_A(n=N, k=K, seed=None):
    rng = random.Random(seed)
    return [[rng.randint(0, (1 << DATA_WIDTH) - 1) for _ in range(k)] for _ in range(n)]


def make_B(k=K, n=N, seed=None):
    rng = random.Random(seed)
    return [[rng.randint(0, (1 << DATA_WIDTH) - 1) for _ in range(n)] for _ in range(k)]


# ── Pack / unpack helpers ─────────────────────────────────────────────────────

def pack_A(A, n=N, k=K, dw=DATA_WIDTH):
    val = 0
    for row in range(n):
        for ki in range(k):
            base = (row * k + ki) * dw
            val |= (A[row][ki] & ((1 << dw) - 1)) << base
    return val


def pack_B(B, k=K, n=N, dw=DATA_WIDTH):
    val = 0
    for ki in range(k):
        for col in range(n):
            base = (ki * n + col) * dw
            val |= (B[ki][col] & ((1 << dw) - 1)) << base
    return val


def unpack_flat(sig, n=N, dw=DATA_WIDTH):
    raw  = int(sig.value)
    mask = (1 << dw) - 1
    return [(raw >> (i * dw)) & mask for i in range(n)]


def unpack_bits(sig, n=N):
    raw = int(sig.value)
    return [(raw >> i) & 1 for i in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# Reference Model
# ──────────────────────────────────────────────────────────────────────────────

class LoaderModel:
    IDLE     = "IDLE"
    LOAD_W   = "LOAD_W"
    STREAM_A = "STREAM_A"
    DONE     = "DONE"

    def __init__(self, n=N, k=K, dw=DATA_WIDTH):
        self.N  = n
        self.K  = k
        self.DW = dw
        self._state = self.IDLE
        self._k     = 0
        self._flush = 0
        self._A     = None
        self._B     = None
        self._buf   = 0

    def arm(self, A, B, buf_sel=0):
        assert self._state == self.IDLE, f"Model not IDLE at arm(): {self._state}"
        self._A   = A
        self._B   = B
        self._buf = buf_sel

    def step(self, start_i: int):
        if self._state == self.IDLE:
            if start_i:
                self._k     = 0
                self._flush = 0
                self._state = self.LOAD_W

        elif self._state == self.LOAD_W:
            if self._k == self.K - 1:
                self._k     = 0
                self._flush = 0
                self._state = self.STREAM_A
            else:
                self._k += 1

        elif self._state == self.STREAM_A:
            if self._k < self.K:
                self._k += 1
            else:
                if self._flush == (self.N - 2):
                    self._state = self.DONE
                else:
                    self._flush += 1

        elif self._state == self.DONE:
            self._state = self.IDLE

    def expected(self):
        exp = {
            "busy_o"     : int(self._state != self.IDLE),
            "done_o"     : int(self._state == self.DONE),
            "act_out"    : [0] * self.N,
            "weight_out" : [0] * self.N,
            "weight_we"  : [0] * self.N,
            "buf_sel_out": [self._buf] * self.N,
        }
        if self._state == self.LOAD_W:
            for col in range(self.N):
                exp["weight_out"][col] = self._B[self._k][col]
                exp["weight_we"][col]  = 1
        elif self._state == self.STREAM_A:
            if self._k < self.K:
                for row in range(self.N):
                    exp["act_out"][row] = self._A[row][self._k]
        return exp

    @property
    def state(self):
        return self._state


# ──────────────────────────────────────────────────────────────────────────────
# Output checker (call only from ReadOnly phase)
# ──────────────────────────────────────────────────────────────────────────────

def check_outputs(dut, model: LoaderModel, label=""):
    exp = model.expected()
    tag = f"[{label}] " if label else ""

    for sig_name in ("busy_o", "done_o"):
        sig = getattr(dut, sig_name)
        assert_resolvable(sig, sig_name)
        got = int(sig.value)
        assert got == exp[sig_name], (
            f"{tag}{sig_name}: expected {exp[sig_name]}, got {got} "
            f"(model state={model.state})"
        )

    act_got  = unpack_flat(dut.act_out,     n=N, dw=DATA_WIDTH)
    wgt_got  = unpack_flat(dut.weight_out,  n=N, dw=DATA_WIDTH)
    wwe_got  = unpack_bits(dut.weight_we,   n=N)
    bsel_got = unpack_bits(dut.buf_sel_out, n=N)

    assert act_got  == exp["act_out"],     f"{tag}act_out:     exp {exp['act_out']},     got {act_got}"
    assert wgt_got  == exp["weight_out"],  f"{tag}weight_out:  exp {exp['weight_out']},  got {wgt_got}"
    assert wwe_got  == exp["weight_we"],   f"{tag}weight_we:   exp {exp['weight_we']},   got {wwe_got}"
    assert bsel_got == exp["buf_sel_out"], f"{tag}buf_sel_out: exp {exp['buf_sel_out']}, got {bsel_got}"


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
#
# ENTRY contract: called right after a RisingEdge (drive-safe, not in ReadOnly).
# EXIT contract:  returns right after the final RisingEdge that clocks DONE→IDLE.
#                 Caller is again in drive-safe territory.
#
# Internal flow per cycle:
#   1. Drive inputs  (we just had a RisingEdge, so driving is legal)
#   2. await RisingEdge  (DUT clocks)
#   3. model.step()
#   4. await ReadOnly()  (outputs stable, read-only phase)
#   5. check_outputs()
# ──────────────────────────────────────────────────────────────────────────────

async def run_one_tile(dut, model: LoaderModel, A, B, buf_sel=0, label="tile"):
    assert model.state == LoaderModel.IDLE, \
        f"{label}: model must be IDLE at entry (got {model.state})"

    # ── Set up inputs (we are in drive-safe territory at entry) ───────────────
    dut.A_flat.value    = pack_A(A)
    dut.B_flat.value    = pack_B(B)
    dut.buf_sel_i.value = buf_sel
    dut.start_i.value   = 1          # assert start
    model.arm(A, B, buf_sel)

    # ── Clock edge 0: IDLE + start_i=1 → LOAD_W ──────────────────────────────
    await RisingEdge(dut.clk_i)
    model.step(start_i=1)
    dut.start_i.value = 0            # de-assert after the edge (still drive-safe)
    await ReadOnly()
    check_outputs(dut, model, label=f"{label} cyc0")

    # ── Subsequent clocks until done_o pulses ─────────────────────────────────
    max_cycles = (K - 1) + K + (N - 1) + 1 + 8

    done_seen = False
    for cyc in range(1, max_cycles + 1):
        # Drive phase (outputs from last ReadOnly still valid in registers)
        await RisingEdge(dut.clk_i)
        model.step(start_i=0)
        await ReadOnly()
        check_outputs(dut, model, label=f"{label} cyc{cyc}")

        if int(dut.done_o.value) == 1:
            assert model.state == LoaderModel.DONE, \
                f"{label} cyc{cyc}: DUT done_o=1 but model={model.state}"
            done_seen = True
            break

    assert done_seen, f"{label}: done_o never asserted within {max_cycles} cycles"

    # ── Post-DONE clock: DONE → IDLE ──────────────────────────────────────────
    # We are in ReadOnly phase right now; must get a new RisingEdge before driving.
    await RisingEdge(dut.clk_i)      # DUT clocks DONE → IDLE
    model.step(start_i=0)
    await ReadOnly()
    check_outputs(dut, model, label=f"{label} post-done")

    assert model.state == LoaderModel.IDLE, \
        f"{label}: model not IDLE after post-done (got {model.state})"
    assert int(dut.busy_o.value) == 0, \
        f"{label}: DUT busy_o still high after returning to IDLE"

    # Return here — we just finished ReadOnly().
    # Caller must await RisingEdge before driving (or just call run_one_tile again
    # which will drive on the NEXT RisingEdge via its own entry logic).
    # We consume one more edge so the caller is back in drive-safe territory:
    await RisingEdge(dut.clk_i)


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

@cocotb.test()
async def test_reset(dut):
    """All outputs clean after reset."""
    await clock_start(dut)
    await reset_sequence(dut)          # returns after RisingEdge — drive-safe
    await ReadOnly()                   # now safe to read

    assert_resolvable(dut.busy_o, "busy_o")
    assert_resolvable(dut.done_o, "done_o")
    assert int(dut.busy_o.value) == 0, "busy_o should be 0 after reset"
    assert int(dut.done_o.value) == 0, "done_o should be 0 after reset"

    wwe = unpack_bits(dut.weight_we,  n=N)
    act = unpack_flat(dut.act_out,    n=N, dw=DATA_WIDTH)
    wgt = unpack_flat(dut.weight_out, n=N, dw=DATA_WIDTH)
    assert wwe == [0]*N, f"weight_we not zero after reset: {wwe}"
    assert act == [0]*N, f"act_out not zero after reset: {act}"
    assert wgt == [0]*N, f"weight_out not zero after reset: {wgt}"


@cocotb.test()
async def test_single_tile(dut):
    """Deterministic tile — full cycle-by-cycle check."""
    await clock_start(dut)
    await reset_sequence(dut)          # drive-safe on return

    model = LoaderModel()
    await with_timeout(
        run_one_tile(dut, model, make_A(seed=42), make_B(seed=99), label="single"),
        timeout_time=2, timeout_unit="us",
    )


@cocotb.test()
async def test_buf_sel(dut):
    """buf_sel_out mirrors buf_sel_i on all columns throughout each run."""
    await clock_start(dut)
    await reset_sequence(dut)

    for bsel in (0, 1):
        model = LoaderModel()
        await with_timeout(
            run_one_tile(dut, model,
                         make_A(seed=7 + bsel), make_B(seed=13 + bsel),
                         buf_sel=bsel, label=f"bsel={bsel}"),
            timeout_time=2, timeout_unit="us",
        )
        # run_one_tile exits after a RisingEdge → safe to loop back in


@cocotb.test()
async def test_back_to_back(dut):
    """Three tiles with no idle gap between them."""
    await clock_start(dut)
    await reset_sequence(dut)

    for i in range(3):
        model = LoaderModel()
        await with_timeout(
            run_one_tile(dut, model,
                         make_A(seed=i * 17), make_B(seed=i * 31),
                         buf_sel=i & 1, label=f"b2b-{i}"),
            timeout_time=2, timeout_unit="us",
        )


@cocotb.test()
async def test_idle_gap(dut):
    """Random idle cycles between tiles; busy_o must stay low during gaps."""
    await clock_start(dut)
    await reset_sequence(dut)

    rng = random.Random(0xDEAD)
    for i in range(4):
        model = LoaderModel()
        await with_timeout(
            run_one_tile(dut, model,
                         make_A(seed=rng.randint(0, 0xFFFF)),
                         make_B(seed=rng.randint(0, 0xFFFF)),
                         buf_sel=rng.randint(0, 1), label=f"gap-{i}"),
            timeout_time=2, timeout_unit="us",
        )
        # run_one_tile already consumed one post-IDLE RisingEdge
        # add more idle clocks, then check at ReadOnly
        for _ in range(rng.randint(0, 5)):
            await RisingEdge(dut.clk_i)
        await ReadOnly()
        assert int(dut.busy_o.value) == 0, f"busy_o high during idle gap after tile {i}"
        assert int(dut.done_o.value) == 0, f"done_o high during idle gap after tile {i}"
        # next loop iteration: run_one_tile entry drives before its own RisingEdge
        # but we're currently at ReadOnly — need one more edge to be drive-safe
        await RisingEdge(dut.clk_i)


@cocotb.test()
async def test_fuzz(dut):
    """50 random tiles with occasional idle gaps."""
    await clock_start(dut)
    await reset_sequence(dut)

    rng = random.Random(0xC0FFEE)
    for i in range(50):
        model = LoaderModel()
        await with_timeout(
            run_one_tile(dut, model,
                         make_A(seed=rng.randint(0, 0xFFFF)),
                         make_B(seed=rng.randint(0, 0xFFFF)),
                         buf_sel=rng.randint(0, 1), label=f"fuzz-{i}"),
            timeout_time=2, timeout_unit="us",
        )
        # run_one_tile exits drive-safe; optional extra idle clocks are fine
        if rng.random() < 0.4:
            for _ in range(rng.randint(1, 4)):
                await RisingEdge(dut.clk_i)


@cocotb.test()
async def test_start_ignored_while_busy(dut):
    """done_o must pulse exactly once even if start_i is held high while busy."""
    await clock_start(dut)
    await reset_sequence(dut)       # drive-safe on return

    # Drive tile data and initial start pulse
    dut.A_flat.value    = pack_A(make_A(seed=5))
    dut.B_flat.value    = pack_B(make_B(seed=6))
    dut.buf_sel_i.value = 0
    dut.start_i.value   = 1
    await RisingEdge(dut.clk_i)
    dut.start_i.value = 0

    # Re-assert start_i mid-run (should be ignored)
    await RisingEdge(dut.clk_i)
    dut.start_i.value = 1
    await RisingEdge(dut.clk_i)
    await RisingEdge(dut.clk_i)
    dut.start_i.value = 0

    done_count = 0
    for _ in range(K + K + (N - 1) + 6):
        await RisingEdge(dut.clk_i)
        await ReadOnly()
        if int(dut.done_o.value) == 1:
            done_count += 1

    assert done_count == 1, f"done_o should pulse exactly once, got {done_count}"


# ──────────────────────────────────────────────────────────────────────────────
# Pytest wrappers
# ──────────────────────────────────────────────────────────────────────────────

tests = [
    "test_reset",
    "test_single_tile",
    "test_buf_sel",
    "test_back_to_back",
    "test_idle_gap",
    "test_fuzz",
    "test_start_ignored_while_busy",
]


@pytest.mark.parametrize("testcase", tests)
def test_load_each(testcase):
    sources = [Path("./rtl/loader.sv").resolve()]
    try:
        run_test(
            parameters={},
            sources=sources,
            module_name="test_load",
            hdl_toplevel="loader",
            testcase=testcase,
        )
    except SystemExit as exc:
        if exc.code != 0:
            pytest.fail(f"cocotb test '{testcase}' failed (exit code {exc.code})")


def test_load_all():
    sources = [Path("./rtl/loader.sv").resolve()]
    try:
        run_test(
            parameters={},
            sources=sources,
            module_name="test_load",
            hdl_toplevel="loader",
        )
    except SystemExit as exc:
        if exc.code != 0:
            pytest.fail(f"cocotb tests failed (exit code {exc.code})")