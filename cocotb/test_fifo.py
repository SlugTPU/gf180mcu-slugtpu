import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, FallingEdge, ReadWrite, ReadOnly, Event
from pathlib import Path
from shared import reset_sequence, clock_start, random_binary_driver, handshake
from runner import run_test
from collections import deque
import random

class FifoModel():
    def __init__(self):
        self.q = deque()

    def consume(self, dut):
        data_i = dut.data_i
        self.q.append(data_i.value.to_unsigned())
        cocotb.log.info(f"CONSUMING! Queue is now at {self.q}")

    def produce(self, dut):
        data_o = dut.data_o
        got = data_o.value.to_unsigned()
        expected = self.q.popleft()
        cocotb.log.info(f"PRODUCING! Queue is now at {self.q}")
        assert got == expected

class ModelRunner():
    def __init__(self, dut, model):
        self.model = model
        self.dut = dut
        self.clk_i = dut.clk_i
        self.rst_i = dut.rst_i
        self.ready_o = dut.ready_o
        self.ready_i = dut.ready_i
        self.valid_i = dut.valid_i
        self.valid_o = dut.valid_o
        self.data_i = dut.data_i
        self.data_o = dut.data_o

    def start(self):
        cocotb.start_soon(self.run_input())
        cocotb.start_soon(self.run_output())

    async def run_input(self):
        while True:
            await handshake(self.clk_i, self.rst_i, self.ready_o, self.valid_i)
            self.model.consume(self.dut)

    async def run_output(self):
        while True:
            await handshake(self.clk_i, self.rst_i, self.ready_i, self.valid_o)
            self.model.produce(self.dut)

# model for streaming n elements with or without backpressure
class StreamIOModel():
    def __init__(self, dut, length, in_pressure, out_pressure):
        self.dut = dut
        self.length = length
        self.in_pressure = in_pressure
        self.out_pressure = out_pressure
        self.n_rd = 0
        self.n_wr = 0

    async def input_run(self):
        data_i = self.dut.data_i
        rst_i = self.dut.rst_i
        clk_i = self.dut.clk_i
        valid_i = self.dut.valid_i
        valid_o = self.dut.valid_o
        ready_o = self.dut.ready_o

        stop_event = Event()

        cocotb.log.info("Running input model")

        await FallingEdge(rst_i)
        valid_i.value = 0
        await FallingEdge(clk_i)

        if (self.in_pressure):
            cocotb.start_soon(random_binary_driver(clk_i, valid_i, prob=0.5, max_hold=10, stop_event=stop_event))
        else:
            valid_i.value = 1

        while True:
            data_i.value = self.n_rd

            await RisingEdge(clk_i)
            if (ready_o.value == 1 and valid_i.value == 1):
                self.n_rd += 1
                cocotb.log.info(f"n_rd is now at {self.n_rd}")
            if (self.n_rd >= self.length):
                stop_event.set()
                await FallingEdge(clk_i)
                valid_i.value = 0
                break
            else:
                await FallingEdge(clk_i)


    async def output_run(self):
        data_i = self.dut.data_i
        rst_i = self.dut.rst_i
        clk_i = self.dut.clk_i
        valid_o = self.dut.valid_o
        ready_i = self.dut.ready_i

        stop_event = Event()

        cocotb.log.info("Running output model")

        await FallingEdge(rst_i)
        ready_i.value = 0
        await FallingEdge(clk_i)

        if (self.out_pressure):
            cocotb.start_soon(random_binary_driver(clk_i, ready_i, prob=0.5, max_hold=10, stop_event=stop_event))
        else:
            ready_i.value = 1

        while True:
            await RisingEdge(clk_i)

            if (valid_o.value == 1 and ready_i.value == 1):
                self.n_wr += 1
                cocotb.log.info(f"n_wr is now at {self.n_wr}")
            # read all expected inputs; can stop input now
            if (self.n_wr >= self.length):
                stop_event.set()
                await FallingEdge(clk_i)
                ready_i.value = 0
                break
            else:
                await FallingEdge(clk_i)

@cocotb.test()
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)

@cocotb.test()
async def fifo_simple_test(dut):
    """Fills up queue and then empties"""
    depth_log2_p = dut.DEPTH_LOG2_P
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i
    data_i = dut.data_i

    m = FifoModel()
    r = ModelRunner(dut, m)

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    r.start()

    await FallingEdge(rst_i)
    valid_i.value = 1
    ready_i.value = 0

    # send values
    for i in range(2 ** depth_log2_p.value.to_unsigned()):
        valid_i.value = 1
        data_i.value = i
        await RisingEdge(clk_i)
        await FallingEdge(clk_i)

    # stop sending values, ready to dequeue
    await FallingEdge(clk_i)
    valid_i.value = 0
    ready_i.value = 1

    for _ in range(2 ** depth_log2_p.value.to_unsigned() - 1):
        await RisingEdge(clk_i)
        await FallingEdge(clk_i)

    await FallingEdge(clk_i)
    ready_i.value = 0
    await FallingEdge(clk_i)

@cocotb.test()
@cocotb.parametrize(with_pressure=[False, True])
async def fifo_random_stream_test(dut, with_pressure):
    depth_log2_p = dut.DEPTH_LOG2_P
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    ready_i = dut.ready_i
    valid_i = dut.valid_i
    data_i = dut.data_i

    cocotb.log.info("Running random stream test")

    m = FifoModel()
    r = ModelRunner(dut, m)
    iom = StreamIOModel(dut, 10, with_pressure, with_pressure)


    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    task_input_model = cocotb.start_soon(iom.input_run())
    task_output_model = cocotb.start_soon(iom.output_run())
    r.start()

    await task_output_model.complete
    await FallingEdge(clk_i)

tests = ["reset_test", "fifo_simple_test", "fifo_stream_test"]

@pytest.mark.parametrize("depth_log2_p", [1, 3])
@pytest.mark.parametrize("testcase", tests)
def test_fifo_each(depth_log2_p, testcase):
    """Runs each test independently. Continues on test failure"""
    proj_path = Path("./rtl").resolve()
    sources = [ proj_path/"fifo.sv" ]

    run_test(parameters={"DEPTH_LOG2_P": depth_log2_p}, sources=sources, module_name="test_fifo", hdl_toplevel="fifo", testcase=testcase)

@pytest.mark.parametrize("depth_log2_p", [1, 3])
def test_fifo_all(depth_log2_p):
    """Runs each test sequentially as one giant test."""
    proj_path = Path("./rtl").resolve()
    sources = [ proj_path/"fifo.sv" ]

    # debug print parameters can be idea if a simulator fails silently without telling you why
    # print(f"DEBUG PARAMETERs: {depth_p}")

    run_test(parameters={"DEPTH_LOG2_P": depth_log2_p}, sources=sources, module_name="test_fifo", hdl_toplevel="fifo")
