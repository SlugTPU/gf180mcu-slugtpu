import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, FallingEdge, ReadWrite, ReadOnly
from pathlib import Path
from shared import reset_sequence, clock_start, random_binary_driver
from runner import run_test
from collections import deque
import random

@cocotb.test()
async def reset_test(dut):
    """Test for Initialization"""
    clk_i = dut.clk_i
    rst_i = dut.reset_i

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)


@cocotb.test()
async def load_simple_test(dut):
    test = [1,2,3,4,5,6,7,8]
    clk_i = dut.clk_i
    rst_i = dut.reset_i
    read_bus = dut.read_bus
    load_valid_i = dut.load_valid_i
    load_enable_i = dut.load_enable_i
    data_o = dut.scalar_values_o

    load_valid_i.value = 0
    load_enable_i.value = 0

    await clock_start(clk_i)
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)
    await FallingEdge(clk_i)

    for i in range(0, len(test)-1, 2):
        read_bus.value = test[i] | (test[i+1] << 32)
        load_valid_i.value = 1
        load_enable_i.value = 1
        print("loading: ", test[i:i+2])
        await FallingEdge(clk_i)
    print("done loading")
    print("data_o = ", data_o)
    load_valid_i.value = 0
    load_enable_i.value = 0
    await FallingEdge(clk_i)

    res = True
    for i in range (0, len(test)):
        print("correct ", test[i], "data ", data_o[i].value.to_unsigned())
        if test[i] != data_o[i].value.to_unsigned():
            res = False
    assert res

    await FallingEdge(clk_i)


tests = ["reset_test", "load_simple_test"]
proj_path = Path("./rtl").resolve()
sources = [ proj_path/"scalar_units/load_data.sv", proj_path/"utils/shift.sv" ]

@pytest.mark.parametrize("testcase", tests)
def test_bias_each(testcase):
    """Runs each test independently. Continues on test failure"""
    
    

    run_test(parameters={}, sources=sources, module_name="test_load_data", hdl_toplevel="load_scalar_data", testcase=testcase)

def test_bias_all():
    """Runs each test sequentially as one giant test."""

    # debug print parameters can be idea if a simulator fails silently without telling you why
    # print(f"DEBUG PARAMETERs: {depth_p}")

    run_test(parameters={}, sources=sources, module_name="test_load_data", hdl_toplevel="load_scalar_data")