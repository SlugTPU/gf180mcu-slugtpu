from typing import Iterator
import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, FallingEdge, ReadWrite, ReadOnly
from pathlib import Path
from shared import reset_sequence, clock_start, random_binary_driver
from runner import run_test
from collections import deque
from cocotb.types import LogicArray, Logic, Array
from collections import deque
import random
import math
from shared import handshake

# TODO: 
# - test saturation behavior

def float_to_fixed(f_val, frac_bits):
    """Converts a float to a fixed-point integer with rounding."""
    scaling_factor = 1 << frac_bits 
    return int(f_val * scaling_factor + 0.5)

def fixed_to_float(fixed_val, frac_bits):
    """Converts a fixed-point integer back to a float."""
    scaling_factor = 1 << frac_bits
    return float(fixed_val) / scaling_factor

def quantize(x, m0):
    """Matches quantizer_mul.sv: multiply, round, shift, saturate."""
    return max(-128, 
           min(127, 
           math.floor(x * m0 + 0.5)))

class mul_n_model():
    def __init__(self, N, width=8):
        self.N = N
        # Assume width is 8 bits for quantization
        self.width = width
        self.q = deque()

    def consume(self, dut):
        data_i_n = [dut.data_i[i].value.to_signed() for i in range(self.N)]
        m0_i_n = [dut.m0_i[i].value.to_unsigned() for i in range(self.N)]
        self.q.append((data_i_n, m0_i_n))

    def produce(self, dut):
        data_o = dut.data_o
        inp_n, m0_n = self.q.popleft()
        FIXED_SHIFT_P = dut.FIXED_SHIFT_P.value.to_unsigned()
        # expected = quantized_mul_n(inp_n, m0_n, dut.ACC_WIDTH_P.value.to_unsigned(), dut.M0_WIDTH_P.value.to_unsigned(), dut.FIXED_SHIFT_P.value.to_unsigned(), self.N)

        for i in range(self.N):
            got = data_o[i].value.to_signed()
            expected = quantize(inp_n[i], fixed_to_float(m0_n[i], FIXED_SHIFT_P))

            cocotb.log.info(f"Producing with input {inp_n[i]} and m0 {fixed_to_float(m0_n[i], FIXED_SHIFT_P)} ({fixed_to_float(m0_n[i], FIXED_SHIFT_P)})): got {got}, expected {expected}")
            # check for accuracy
            # add small epsilon to expected to avoid division by zero in assertion
            err = abs(got - expected) / (abs(expected) + 1e-12)
            assert err < 0.5, f"Output mismatch at index {i}: got {got}, expected {expected}"

class InputModel():
    def __init__(self, dut, data_generator: Iterator[tuple[Array[int], Array[int]]], handshake_generator: Iterator[bool]):
        self.dut = dut
        self.N = dut.N.value.to_unsigned()
        self.data_generator = data_generator
        self.handshake_generator = handshake_generator
        self.nin = 0

    def start(self):
        return cocotb.start_soon(self._run())


    async def _run(self):
        clk_i = self.dut.clk_i
        rst_i = self.dut.rst_i
        valid_i = self.dut.data_valid_i
        ready_i = self.dut.data_ready_i
        ready_o = self.dut.data_ready_o
        data_i = self.dut.data_i
        m0_i = self.dut.m0_i

        FIXED_SHIFT_P = self.dut.FIXED_SHIFT_P.value

        await FallingEdge(rst_i)

        # stream random
        for (d, m0) in self.data_generator:
            tx = False

            print(f"InputModel: waiting to send data {d} with m0 {m0}")

            for i in range(self.N):
                data_i[i].value = d[i]
                m0_i[i].value = m0[i]

            while (not tx):
                valid_i.value = next(self.handshake_generator)

                await RisingEdge(clk_i)

                if valid_i.value and ready_o.value == 1:
                    cocotb.log.info(f"Tx input: data_i={[data_i[i].value.to_signed() for i in range(self.N)]}, m0_i={[fixed_to_float(m0_i[i].value.to_signed(), FIXED_SHIFT_P.to_unsigned()) for i in range(self.N)]}")
                    self.nin += 1
                    tx = True
            await FallingEdge(clk_i)

class OutputModel():
    def __init__(self, dut, handshake_generator, total_nin):
        self.dut = dut
        self.N = dut.N.value.to_unsigned()
        self.total_nin = total_nin
        self.nout = 0
        self.handshake_generator = handshake_generator

    def start(self):
        return cocotb.start_soon(self._run())

    async def _run(self):
        clk_i = self.dut.clk_i
        rst_i = self.dut.rst_i
        valid_o = self.dut.data_valid_o
        ready_i = self.dut.data_ready_i
        data_o = self.dut.data_o

        await FallingEdge(rst_i)

        while self.nout < self.total_nin:
            rx = False
            while (not rx):
                ready_i.value = next(self.handshake_generator)
                await RisingEdge(clk_i)
                if valid_o.value == 1:
                    self.nout += 1
                    rx = True
            await FallingEdge(clk_i)


class ModelRunner():
    def __init__(self, dut):
        self.dut = dut
        self.model = mul_n_model(dut.N.value.to_unsigned())

    def start(self):
        cocotb.start_soon(self.run_input())
        cocotb.start_soon(self.run_output())

    async def run_input(self):
        clk_i = self.dut.clk_i
        valid_i, valid_o = self.dut.data_valid_i, self.dut.data_valid_o
        ready_i, ready_o = self.dut.data_ready_i, self.dut.data_ready_o
        rst_i = self.dut.rst_i

        while True:
            await handshake(clk_i, rst_i, ready_o, valid_i)
            self.model.consume(self.dut)

    async def run_output(self):
        clk_i = self.dut.clk_i
        valid_i, valid_o = self.dut.data_valid_i, self.dut.data_valid_o
        ready_i, ready_o = self.dut.data_ready_i, self.dut.data_ready_o
        rst_i = self.dut.rst_i

        while True:
            cocotb.log.info("Waiting for output handshake...")
            await handshake(clk_i, rst_i, ready_i, valid_o)
            self.model.produce(self.dut)
            cocotb.log.info("Output handshake successful, producing output")

@cocotb.test()
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    data_i = dut.data_i

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)
    cocotb.log.info("Reset complete")


@cocotb.test()
async def scale_n_simple_test(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    data_i = dut.data_i
    m0_i = dut.m0_i
    data_valid_i = dut.data_valid_i
    data_ready_i = dut.data_ready_i
    N = dut.N
    FIXED_SHIFT_P = dut.FIXED_SHIFT_P
    m = ModelRunner(dut)

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    m.start()

    await FallingEdge(rst_i)

    data_ready_i.value = 1
    data_valid_i.value = 1

    for i in range(N.value):
        m0_i[i].value = random.randint(0, 10) << FIXED_SHIFT_P.value.to_unsigned()
        data_i[i].value = random.randint(-10, 10)

    await RisingEdge(dut.clk_i)

    await FallingEdge(dut.clk_i)
    data_ready_i.value = 1
    data_valid_i.value = 0
    await FallingEdge(dut.clk_i)

# test random data stream with no output backpressure, random input pressure
@cocotb.test()
async def scale_n_stream_test(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    N = dut.N.value.to_unsigned()
    FIXED_SHIFT_P = dut.FIXED_SHIFT_P.value.to_unsigned()
    total_nin = 10

    def data_generator() -> Iterator[tuple[Array[int], Array[int]]]:
        n = 0
        while n < total_nin:
            yield ([random.randint(-10, 10) for _ in range(N)], 
                   [float_to_fixed(random.random(), FIXED_SHIFT_P) for _ in range(N)])
            n += 1
    def in_handhsake_generator() -> Iterator[bool]:
        while True:
            yield random.choice([True, False])
    # emulate yes(1)
    def generate_yes() -> Iterator[bool]:
        while True:
            yield True

    input_model = InputModel(dut, data_generator(), in_handhsake_generator())
    output_model = OutputModel(dut, generate_yes(), total_nin)
    m = ModelRunner(dut)

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    m.start()
    task_im = input_model.start()
    task_om = output_model.start()

    await task_om.complete
    await FallingEdge(clk_i)
    dut.data_ready_i.value = 0
    dut.data_valid_i.value = 0
    await FallingEdge(clk_i)


tests = ["reset_test", "scale_n_simple_test"]
proj_path = Path("./src").resolve()
sources = [ proj_path/"common/elastic.sv", proj_path/"scalar_units"/"scale_n.sv", proj_path/"quantizer_mul.sv"]

@pytest.mark.parametrize("testcase", tests)
def test_scale_n_each(testcase):
    """Runs each test independently. Continues on test failure"""
    run_test(parameters={}, sources=sources, module_name="test_scale_n", hdl_toplevel="scale_n", testcase=testcase)

def test_scale_n_all():
    """Runs each test sequentially as one giant test."""

    # debug print parameters can be idea if a simulator fails silently without telling you why
    # print(f"DEBUG PARAMETERs: {depth_p}")

    run_test(parameters={}, sources=sources, module_name="test_scale_n", hdl_toplevel="scale_n")