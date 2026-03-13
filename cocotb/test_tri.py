from __future__ import annotations

import random
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, ReadOnly

import pytest
from runner import run_test


async def drive(sig, val):
    sig.value = int(val)
    await Timer(1, unit="ps")


def u(sig) -> int:
    return int(sig.value)


async def settle_clk(dut, n=2):
    for _ in range(n):
        await RisingEdge(dut.clk)


async def init_dut(dut, clk_period_ns=10):
    cocotb.start_soon(Clock(dut.clk, clk_period_ns, unit="ns").start())

    n = len(dut.data_o)

    for i in range(n):
        await drive(dut.data_i[i], 0)

    await drive(dut.enable_i, 0)

    await drive(dut.rst, 1)
    await settle_clk(dut, 5)
    await drive(dut.rst, 0)
    await settle_clk(dut, 5)


class TriangleScoreboard:
    def __init__(self, n: int):
        self.n = n
        self.pipeline = [[0] * (i + 1) for i in range(n)]

    def step(self, lane_inputs: list[int], lane_en: list[int]) -> list[int]:
        exp = []
        for i in range(self.n):
            if lane_en[i]:
                self.pipeline[i].pop(0)
                self.pipeline[i].append(lane_inputs[i])
            exp.append(self.pipeline[i][0])
        return exp


async def drive_vec(sig_arr, vals):
    for i, v in enumerate(vals):
        await drive(sig_arr[i], v)


def pack_enables(ens: list[int]) -> int:
    result = 0
    for i, e in enumerate(ens):
        if e:
            result |= (1 << i)
    return result


async def sample_vec(sig_arr, n):
    return [u(sig_arr[i]) for i in range(n)]


async def check_outputs(dut, expected, tag=""):
    await ReadOnly()
    got = await sample_vec(dut.data_o, len(expected))
    for i, (g, e) in enumerate(zip(got, expected)):
        if g != e:
            raise AssertionError(
                f"[{tag}] lane {i}: expected 0x{e:04X}, got 0x{g:04X}"
            )
    await Timer(1, unit="ps")


@cocotb.test()
async def reset_test(dut):
    await init_dut(dut)
    n = len(dut.data_o)
    await ReadOnly()
    got = await sample_vec(dut.data_o, n)
    assert all(x == 0 for x in got), f"Reset outputs not zero: {got}"


@cocotb.test()
async def all_enabled_ramp_test(dut):
    await init_dut(dut)
    n = len(dut.data_o)
    sb = TriangleScoreboard(n)
    cycles = n + 12

    for cyc in range(cycles):
        vals = [(0x1000 + cyc * 0x10 + i) & 0xFFFF for i in range(n)]
        ens = [1] * n

        await drive_vec(dut.data_i, vals)
        await drive(dut.enable_i, pack_enables(ens))

        await RisingEdge(dut.clk)

        exp = sb.step(vals, ens)
        await check_outputs(dut, exp, tag=f"ramp cyc={cyc}")


@cocotb.test()
async def gated_bubbles_test(dut):
    await init_dut(dut)
    n = len(dut.data_o)
    sb = TriangleScoreboard(n)
    cycles = n + 20

    for cyc in range(cycles):
        bubble = (cyc % 3) == 0
        ens = [0] * n if bubble else [1] * n
        vals = [(0x2000 + cyc * 7 + i) & 0xFFFF for i in range(n)]

        await drive_vec(dut.data_i, vals)
        await drive(dut.enable_i, pack_enables(ens))

        await RisingEdge(dut.clk)

        exp = sb.step(vals, ens)
        await check_outputs(dut, exp, tag=f"bubble cyc={cyc}")


@cocotb.test()
async def random_enables_random_data_test(dut):
    await init_dut(dut)
    n = len(dut.data_o)
    sb = TriangleScoreboard(n)

    for cyc in range(120):
        ens = [random.randrange(0, 2) for _ in range(n)]
        vals = [random.randrange(0, 1 << 16) for _ in range(n)]

        await drive_vec(dut.data_i, vals)
        await drive(dut.enable_i, pack_enables(ens))

        await RisingEdge(dut.clk)

        exp = sb.step(vals, ens)
        await check_outputs(dut, exp, tag=f"rand cyc={cyc}")


tests = [
    "reset_test",
    "all_enabled_ramp_test",
    "gated_bubbles_test",
    "random_enables_random_data_test",
]


@pytest.mark.parametrize("testcase", tests)
def test_tri_each(testcase):
    sources = [
        Path("./rtl/tri_shift.sv").resolve(),
        Path("./rtl/utils/shift.sv").resolve(),
    ]
    try:
        run_test(
            parameters={"N": 8, "DATA_W": 16},
            sources=sources,
            module_name="test_tri",
            hdl_toplevel="tri_shift",
            testcase=testcase,
        )
    except SystemExit as exc:
        if exc.code != 0:
            pytest.fail(f"cocotb test '{testcase}' failed (exit code {exc.code})")


def test_tri_all():
    sources = [
        Path("./rtl/tri_shift.sv").resolve(),
        Path("./rtl/utils/shift.sv").resolve(),
    ]
    try:
        run_test(
            parameters={"N": 8, "DATA_W": 16},
            sources=sources,
            module_name="test_tri",
            hdl_toplevel="tri_shift",
        )
    except SystemExit as exc:
        if exc.code != 0:
            pytest.fail(f"cocotb tests failed (exit code {exc.code})")