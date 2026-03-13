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
from shared import handshake, is_resetting

def logic_add(a: LogicArray, b: LogicArray, width: int) -> LogicArray:
    """
    Adds two LogicArrays, returning the logic array with the given width. 

    Assumes inputs are signed and returns a signed result.
    """
    res = (a.to_signed() + b.to_signed())
    return LogicArray.from_signed(res, width)

def add_n(data_i: Array[LogicArray], bias_i: Array[LogicArray], N: int, width: int) -> Array[LogicArray]:
    """Adds N elements of data_i to bias_i."""
    result = Array([], N)
    for i in range(N):
        result[i] = logic_add(data_i[i], bias_i[i], width)
    return result

class add_n_model():
    def __init__(self, N: int, width: int):
        self.N = N
        self.width = width
        self.q = deque()

    def consume(self, dut):
        cocotb.log.info(f"Debug: got data_i={[dut.data_i[i].value.to_signed() for i in range(self.N)]}, bias_i={[dut.bias_i[i].value.to_signed() for i in range(self.N)]}")
        cocotb.log.info(f"Consuming input: data_i={[dut.data_i[i].value.to_signed() for i in range(self.N)]}, bias_i={[dut.bias_i[i].value.to_signed() for i in range(self.N)]}")
        data_snapshot = [dut.data_i[i].value for i in range(self.N)]
        bias_snapshot = [dut.bias_i[i].value for i in range(self.N)]
        self.q.append((data_snapshot, bias_snapshot))

    def produce(self, dut):
        data_o = dut.data_o

        inp_n, bias_n = self.q.popleft()

        for i in range(self.N):
            res = logic_add(inp_n[i], bias_n[i], self.width)
            # cocotb.log.info(f"Produ {res}, got {got_n[i]}")
            cocotb.log.info(f"Producing with input {inp_n[i].to_signed()} and bias {bias_n[i].to_signed()}: got {data_o[i].value.to_signed()}, expected {res.to_signed()}")
            assert data_o[i].value == res, f"Expected {res.to_signed()}, got {data_o[i].value.to_signed()} at index {i}"

class InputModel():
    def __init__(self, dut, data_generator: Iterator[tuple[Array[int], Array[int]]], handshake_generator: Iterator[bool]):
        self.dut = dut
        self.N = dut.N.value.to_unsigned()
        self.width = dut.width_p.value.to_unsigned()
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

        await FallingEdge(rst_i)

        # stream random
        for (d, b) in self.data_generator:
            tx = False

            for i in range(self.N):
                data_i[i].value = d[i]
                bias_i[i].value = b[i]

            while (not tx):
                valid_i.value = next(self.handshake_generator)

                await RisingEdge(clk_i)

                cocotb.log.info(f"Tx input: data_i={[data_i[i].value.to_signed() for i in range(self.N)]}, bias_i={[bias_i[i].value.to_signed() for i in range(self.N)]}")
                if valid_i.value and ready_o.value == 1:
                    self.nin += 1
                    tx = True
            await FallingEdge(clk_i)

class OutputModel():
    def __init__(self, dut, handshake_generator, total_nin):
        self.dut = dut
        self.N = dut.N.value.to_unsigned()
        self.width = dut.width_p.value.to_unsigned()
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
        self.model = add_n_model(dut.N.value.to_unsigned(), dut.width_p.value.to_unsigned())

    def start(self):
        cocotb.start_soon(self.run_input())
        cocotb.start_soon(self.run_output())

    async def run_input(self):
        clk_i = self.dut.clk_i
        valid_i, valid_o = self.dut.data_valid_i, self.dut.data_valid_o
        ready_i, ready_o = self.dut.data_ready_i, self.dut.data_ready_o
        rst_i = self.dut.rst_i

        await FallingEdge(rst_i)

        while True:
            # await handshake(clk_i, rst_i, ready_o, valid_i)
            await RisingEdge(clk_i)
            if (valid_i.value == 1 and ready_o.value == 1):
                cocotb.log.info("Input handshake successful, consuming input")
                self.model.consume(self.dut)

    async def run_output(self):
        clk_i = self.dut.clk_i
        valid_i, valid_o = self.dut.data_valid_i, self.dut.data_valid_o
        ready_i, ready_o = self.dut.data_ready_i, self.dut.data_ready_o
        rst_i = self.dut.rst_i

        await FallingEdge(rst_i)

        while True:
            cocotb.log.info(f"Waiting for output handshake..., ready_i={ready_i.value}, valid_o={valid_o.value}, valid_i={valid_i.value}, ready_o={ready_o.value}")
            # await handshake(clk_i, rst_i, ready_i, valid_o)
            await RisingEdge(clk_i)
            if (ready_i.value == 1 and valid_o.value == 1):
                cocotb.log.info("Output handshake successful, producing output")
                self.model.produce(self.dut)

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
async def add_n_simple_test(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    bias_i = dut.bias_i
    data_i = dut.data_i
    data_valid_i = dut.data_valid_i
    data_ready_i = dut.data_ready_i
    N = dut.N
    width_p = dut.width_p
    m = ModelRunner(dut)

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    m.start()

    await FallingEdge(rst_i)

    data_ready_i.value = 1
    data_valid_i.value = 1

    for i in range(N.value):
        bias_i[i].value = random.randint(-10, 10)
        data_i[i].value = random.randint(-10, 10)

    await RisingEdge(dut.clk_i)

    await FallingEdge(dut.clk_i)
    data_ready_i.value = 1
    data_valid_i.value = 0
    await FallingEdge(dut.clk_i)

# stream input with no backpressure on output
@cocotb.test()
async def add_n_stream(dut):
    """Randomized test with backpressure."""

    width_p = dut.width_p    
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    N = dut.N
    total_nin = 10

    def generate_data() -> Iterator[tuple[Array[int], Array[int]]]:
        n = 0
        while n < total_nin:
            # yield random input and bias data from [-10, 10]
            yield ([random.randint(-10, 10) for _ in range(N.value.to_unsigned())], 
                   [random.randint(-10, 10) for _ in range(N.value.to_unsigned())])
            n += 1

    # emulate man(1)
    def generate_yes() -> Iterator[bool]:
        while True:
            yield True

    def generate_backpressure() -> Iterator[bool]:
        while True:
            yield random.choice([True, False])

    m = ModelRunner(dut)
    im = InputModel(dut, generate_data(), generate_backpressure())
    om = OutputModel(dut, generate_yes(), total_nin=total_nin)

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    m.start()
    task_im = im.start()
    task_om = om.start()

    await task_om.complete
    await FallingEdge(clk_i)
    dut.data_ready_i.value = 0
    dut.data_valid_i.value = 0

@cocotb.test()
async def add_n_random_backpressure(dut):
    """Randomized test with backpressure."""

    width_p = dut.width_p    
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    N = dut.N
    total_nin = 10

    def generate_data() -> Iterator[tuple[Array[int], Array[int]]]:
        n = 0
        while n < total_nin:
            yield ([random.randint(-10, 10) for _ in range(N.value.to_unsigned())], 
                   [random.randint(-10, 10) for _ in range(N.value.to_unsigned())])
            n += 1

    def generate_backpressure() -> Iterator[bool]:
        while True:
            yield random.choice([True, False])

    m = ModelRunner(dut)
    im = InputModel(dut, generate_data(), generate_backpressure())
    om = OutputModel(dut, generate_backpressure(), total_nin=total_nin)

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    m.start()
    task_im = im.start()
    task_om = om.start()

    await task_om.complete
    await FallingEdge(clk_i)
    dut.data_ready_i.value = 0
    dut.data_valid_i.value = 0


tests = ["reset_test", "add_n_simple_test", "add_n_random_backpressure"]
proj_path = Path("./rtl").resolve()
sources = [ proj_path/"utils/elastic.sv", proj_path/"scalar_units/add_n.sv"   ]

@pytest.mark.parametrize("testcase", tests)
def test_add_n_each(testcase):
    """Runs each test independently. Continues on test failure"""
    run_test(parameters={}, sources=sources, module_name="test_add_n", hdl_toplevel="add_n", testcase=testcase)

def test_add_all():
    """Runs each test sequentially as one giant test."""

    # debug print parameters can be idea if a simulator fails silently without telling you why
    # print(f"DEBUG PARAMETERs: {depth_p}")

    run_test(parameters={}, sources=sources, module_name="test_add_n", hdl_toplevel="add_n")