import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
import random

N = 8
INT32_MIN = -(2**31)
INT32_MAX = (2**31) - 1


def to_uint(val):
    return val & 0xFFFFFFFF


def drive_array(signal, values):
    """Drive an unpacked array"""
    for i in range(N):
        signal[i].value = to_uint(values[i])


def read_array(signal):
    """Read"""
    return [signal[i].value.to_signed() for i in range(N)]


def relu_ref(val):
    return max(0, val)


@cocotb.test()
async def reset_test(dut):
    """should be zero and valid deasserted after reset."""
    await clock_start(dut.clk_i)

    dut.data_valid_i.value = 0
    dut.data_ready_i.value = 1
    drive_array(dut.data_i, [0] * N)

    await reset_sequence(dut.clk_i, dut.rst_i)
    await RisingEdge(dut.clk_i)

    assert dut.data_valid_o.value == 0, "valid_o should be 0 after reset"
    for i in range(N):
        assert dut.data_o[i].value.to_signed() == 0, f"data_o[{i}] should be 0 after reset"


@cocotb.test()
async def known_values_test(dut):
    """Drive known positive/negative/edge values through all N lanes."""
    await clock_start(dut.clk_i)

    dut.data_valid_i.value = 0
    dut.data_ready_i.value = 1
    drive_array(dut.data_i, [0] * N)

    await reset_sequence(dut.clk_i, dut.rst_i)

    test_vectors = [
        [1, -1, 0, 127, -128, INT32_MAX, INT32_MIN, 42],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -2, -3, -4, -5, -6, -7, -8],
        [10, 20, 30, 40, 50, 60, 70, 80],
        [INT32_MIN, INT32_MIN + 1, -1, 0, 1, INT32_MAX - 1, INT32_MAX, 999],
    ]

    errors = 0

    for row_idx, row in enumerate(test_vectors):
        await FallingEdge(dut.clk_i)
        drive_array(dut.data_i, row)
        dut.data_valid_i.value = 1

        await FallingEdge(dut.clk_i)

        if dut.data_valid_o.value != 1:
            dut._log.error(f"Row {row_idx}: data_valid_o not asserted")
            errors += 1
            continue

        got = read_array(dut.data_o)
        for i in range(N):
            exp = relu_ref(row[i])
            if got[i] != exp:
                dut._log.error(
                    f"Row {row_idx}[{i}]: in={row[i]}, exp={exp}, got={got[i]}"
                )
                errors += 1

    await FallingEdge(dut.clk_i)
    dut.data_valid_i.value = 0

    assert errors == 0, f"known_values_test failed with {errors} errors"
    dut._log.info("known_values_test passed")


@cocotb.test()
async def random_stress_test(dut):
    """checks N lanes over many beats"""
    await clock_start(dut.clk_i)

    dut.data_valid_i.value = 0
    dut.data_ready_i.value = 1
    drive_array(dut.data_i, [0] * N)

    await reset_sequence(dut.clk_i, dut.rst_i)

    NUM_BEATS = 64
    errors = 0
    pending = None

    for beat in range(NUM_BEATS):
        row = [random.randint(INT32_MIN, INT32_MAX) for _ in range(N)]

        await FallingEdge(dut.clk_i)
        drive_array(dut.data_i, row)
        dut.data_valid_i.value = 1

        if pending is not None:
            if dut.data_valid_o.value != 1:
                dut._log.error(f"Beat {beat}: valid_o not asserted")
                errors += 1
            else:
                got = read_array(dut.data_o)
                for i in range(N):
                    exp = relu_ref(pending[i])
                    if got[i] != exp:
                        dut._log.error(
                            f"Beat {beat}[{i}]: in={pending[i]}, exp={exp}, got={got[i]}"
                        )
                        errors += 1

        pending = row

    # Drain last beat
    await FallingEdge(dut.clk_i)
    dut.data_valid_i.value = 0
    if pending is not None:
        got = read_array(dut.data_o)
        for i in range(N):
            exp = relu_ref(pending[i])
            if got[i] != exp:
                dut._log.error(
                    f"Drain[{i}]: in={pending[i]}, exp={exp}, got={got[i]}"
                )
                errors += 1

    assert errors == 0, f"random_stress_test failed with {errors} errors"
    dut._log.info("random_stress_test passed")


@cocotb.test()
async def backpressure_test(dut):
    """verify data holds during backpressure."""
    await clock_start(dut.clk_i)

    dut.data_valid_i.value = 0
    dut.data_ready_i.value = 1
    drive_array(dut.data_i, [0] * N)

    await reset_sequence(dut.clk_i, dut.rst_i)

    test_rows = [
        [random.randint(INT32_MIN, INT32_MAX) for _ in range(N)]
        for _ in range(4)
    ]
    errors = 0

    for row_idx, row in enumerate(test_rows):
        # drive data w ready=1
        await FallingEdge(dut.clk_i)
        drive_array(dut.data_i, row)
        dut.data_valid_i.value = 1
        dut.data_ready_i.value = 1

        await FallingEdge(dut.clk_i)
        dut.data_valid_i.value = 0

        # output check
        if dut.data_valid_o.value != 1:
            dut._log.error(f"Row {row_idx}: valid_o not asserted")
            errors += 1
            continue

        got_before = read_array(dut.data_o)

        # stall downstream for 2 cycles
        dut.data_ready_i.value = 0

        await FallingEdge(dut.clk_i)
        assert dut.data_valid_o.value == 1, f"Row {row_idx}: valid_o dropped during stall"

        await FallingEdge(dut.clk_i)
        got_during = read_array(dut.data_o)

        # now output must be held
        for i in range(N):
            if got_during[i] != got_before[i]:
                dut._log.error(
                    f"Row {row_idx}[{i}]: data changed during stall: "
                    f"{got_before[i]} -> {got_during[i]}"
                )
                errors += 1

        dut.data_ready_i.value = 1

        # check correctness
        for i in range(N):
            exp = relu_ref(row[i])
            if got_before[i] != exp:
                dut._log.error(
                    f"Row {row_idx}[{i}]: in={row[i]}, exp={exp}, got={got_before[i]}"
                )
                errors += 1

    await FallingEdge(dut.clk_i)
    assert errors == 0, f"backpressure_test failed with {errors} errors"
    dut._log.info("backpressure_test passed")


tests = ["reset_test", "known_values_test", "random_stress_test", "backpressure_test"]
proj_path = Path("./src").resolve()
sources = [proj_path / "scalar_units" / "relu_n.sv"]


@pytest.mark.parametrize("testcase", tests)
def test_relu_n_each(testcase):
    run_test(
        parameters={},
        sources=sources,
        module_name="test_relu_n",
        hdl_toplevel="relu_n",
        testcase=testcase,
    )


def test_relu_n_all():
    run_test(
        parameters={},
        sources=sources,
        module_name="test_relu_n",
        hdl_toplevel="relu_n",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
