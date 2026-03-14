from typing import Iterator
import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles
from pathlib import Path
from shared import clock_start, reset_sequence, handshake
from cocotb.types import LogicArray, Logic, Array
from runner import run_test
import random
from test_scale_n import float_to_fixed, fixed_to_float, quantize
from collections import deque

def relu(x):
    return max(0, x)

class SclarPipeModel():
    def __init__(self):
        self.q = deque()

    def consume(self, dut):
        N = dut.N.value.to_unsigned()
        data_n = [dut.data_i[i].value.to_signed() for i in range(N)]
        bias_n = [dut.bias_i[i].value.to_signed() for i in range(N)]
        zp_n = [dut.zero_point_i[i].value.to_signed() for i in range(N)]
        scale_n = [dut.scale_i[i].value.to_unsigned() for i in range(N)]
        self.q.append((data_n, bias_n, zp_n, scale_n))

    def produce(self, dut):
        data_o = dut.data_o
        data_n, bias_n, zp_n, scale_n = self.q.popleft()
        N = dut.N.value.to_unsigned()
        FIXED_SHIFT = dut.FIXED_SHIFT.value.to_unsigned()

        for i in range(N):
            data, bias, zp, m0 = data_n[i], bias_n[i], zp_n[i], scale_n[i]
            # bias -> relu -> zero point -> quantize
            # zp is signed so we implicitly subtracts
            got = data_o[i].value.to_signed()
            expected = quantize(relu(data + bias) + zp, fixed_to_float(m0, FIXED_SHIFT))

            # expected = max(0, inp[i].to_signed())
            cocotb.log.info(f"=== Producing output...")
            cocotb.log.info(f"Input data: {data}, bias: {bias}, zero point: {zp}, scale: {fixed_to_float(m0, FIXED_SHIFT)}")
            cocotb.log.info(f"Got {got}, expected {expected}")
            cocotb.log.info(f"=== Finished producing output")
            assert abs(got - expected) / (abs(expected) + 1e-12) < 0.10, f"Output mismatch at index {i}: got {got}, expected {expected}"

class ModelRunner():
    def __init__(self, dut):
        self.dut = dut
        self.model = SclarPipeModel()

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

    
class InputModel():
    def __init__(self, dut, data_generator: Iterator[tuple[Array[int], Array[int], Array[int], Array[int]]], handshake_generator: Iterator[bool]):
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
        bias_i = self.dut.bias_i
        zp_i = self.dut.zero_point_i
        scale_i = self.dut.scale_i

        await FallingEdge(rst_i)

        # stream random
        for (data, bias, zp, m0) in self.data_generator:
            tx = False

            for i in range(self.N):
                data_i[i].value = data[i]
                bias_i[i].value = bias[i]
                zp_i[i].value = zp[i]
                scale_i[i].value = m0[i]

            while (not tx):
                valid_i.value = next(self.handshake_generator)

                await RisingEdge(clk_i)

                # cocotb.log.info(f"Tx input: data_i={[data_i[i].value.to_signed() for i in range(self.N)]}, m0_i={[m0_i[i].value.to_signed() for i in range(self.N)]}")
                if valid_i.value and ready_o.value == 1:
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

@cocotb.test()
async def test_scalar_pipe_basic(dut):
    """Single vector through pipeline, verify each lane."""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    N = dut.N.value.to_unsigned()
    FIXED_SHIFT = dut.FIXED_SHIFT.value.to_unsigned()
    total_nin = 10

    # generate total_nin random transactions
    def data_generator() -> Iterator[tuple[Array[int], Array[int], Array[int], Array[int]]]:
        bias = [random.randint(-10, 10) for _ in range(N)]
        zp = [random.randint(-10, 10) for _ in range(N)]
        scale = [float_to_fixed(random.random(), FIXED_SHIFT) for _ in range(N)]
        # only data changes immediately after a transaction
        for _ in range(total_nin):
            data = [random.randint(-10, 10) for _ in range(N)]
            yield (data, bias, zp, scale)

    # emulate yes(1)
    def yes_generator() -> Iterator[bool]:
        while True:
            yield True

    input_model = InputModel(dut, data_generator(), yes_generator())
    output_model = OutputModel(dut, yes_generator(), total_nin)
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

@cocotb.test()
async def test_scalar_pipe_backpressure(dut):
    """Test backpressure handling."""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    N = dut.N.value.to_unsigned()
    FIXED_SHIFT = dut.FIXED_SHIFT.value.to_unsigned()
    total_nin = 10

    # generate total_nin random transactions
    def data_generator() -> Iterator[tuple[Array[int], Array[int], Array[int], Array[int]]]:
        bias = [random.randint(-10, 10) for _ in range(N)]
        zp = [random.randint(-10, 10) for _ in range(N)]
        scale = [float_to_fixed(random.random(), FIXED_SHIFT) for _ in range(N)]
        # only data changes immediately after a transaction
        for _ in range(total_nin):
            data = [random.randint(-10, 10) for _ in range(N)]
            yield (data, bias, zp, scale)

    # emulate yes(1)
    def yes_generator() -> Iterator[bool]:
        while True:
            yield True

    def backpressure_generator() -> Iterator[bool]:
        while True:
            # randomly apply backpressure with 20% probability
            yield random.random() > 0.2

    input_model = InputModel(dut, data_generator(), yes_generator())
    output_model = OutputModel(dut, backpressure_generator(), total_nin)
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

tests = [
    "test_scalar_pipe_basic",
    "test_scalar_pipe_backpressure"
]

SOURCES = [
    Path("./src/scalar_units/scalar_pipe.sv").resolve(),
    Path("./src/scalar_units/add_n.sv").resolve(),
    Path("./src/scalar_units/relu_n.sv").resolve(),
    Path("./src/scalar_units/scale_n.sv").resolve(),
    Path("./src/quantizer_mul.sv").resolve(),
    Path("./src/common/elastic.sv").resolve(),
]

@pytest.mark.parametrize("testcase", tests)
def test_scalar_pipe_each(testcase):
    run_test(
        sources=SOURCES,
        module_name="test_scalar_pipe",
        hdl_toplevel="scalar_pipe",
        parameters={},
        testcase=testcase,
    )

def test_scalar_pipe_all():
    run_test(
        sources=SOURCES,
        module_name="test_scalar_pipe",
        hdl_toplevel="scalar_pipe",
        parameters={},
    )