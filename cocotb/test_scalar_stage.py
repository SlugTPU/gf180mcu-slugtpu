import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, ClockCycles
from pathlib import Path

from shared import clock_start, reset_sequence
from runner import run_test

N = 8
FIXED_SHIFT = 16

def to_uint(val, width=32):
    return val & ((1 << width) - 1)

def drive_array(signal, values, width=32):
    for i in range(len(values)):
        signal[i].value = to_uint(values[i], width)

def read_array_s8(signal, n):
    out =[]
    for i in range(n):
        raw = int(signal[i].value) & 0xFF
        out.append(raw - 256 if raw >= 128 else raw)
    return out

def quantize(psum, m0, shift=FIXED_SHIFT):
    product = psum * m0
    rounded = product + (1 << (shift - 1))
    shifted = rounded >> shift
    return max(-128, min(127, shifted))

def scalar_pipe_ref(data, bias, zp, scale):
    out =[]
    for i in range(len(data)):
        v = data[i] + bias[i]
        v = max(0, v)
        v = v - zp[i]
        v = quantize(v, scale[i])
        out.append(v)
    return out

async def init(dut):
    cocotb.start_soon(Clock(dut.clk_i, 10, unit="ns").start())
    dut.rst_i.value = 1
    
    # init mem bus inputs
    dut.read_bus_i.value = 0
    dut.load_valid_i.value = 0
    dut.load_bias_en_i.value = 0
    dut.load_zp_en_i.value = 0
    dut.load_scale_en_i.value = 0

    # init pipeline inputs
    dut.data_valid_i.value = 0
    dut.data_ready_i.value = 1
    drive_array(dut.data_i,[0] * N)

    await ClockCycles(dut.clk_i, 5)
    await FallingEdge(dut.clk_i)
    dut.rst_i.value = 0
    await ClockCycles(dut.clk_i, 2)

async def load_param(dut, param_vals, en_sig):
    """Shifts parameters sequentially via the 64-bit wide bus"""
    lanes = 2
    depth = 4
    
    for d in range(depth):
        await FallingEdge(dut.clk_i)
        bus_val = 0
        for l in range(lanes):
            idx = d * lanes + l
            val = param_vals[idx] & 0xFFFFFFFF
            bus_val |= (val << (l * 32))
        
        dut.read_bus_i.value = bus_val
        dut.load_valid_i.value = 1
        en_sig.value = 1
    
    await FallingEdge(dut.clk_i)
    dut.load_valid_i.value = 0
    en_sig.value = 0
    await ClockCycles(dut.clk_i, 2)


@cocotb.test()
async def test_basic(dut):
    """Load parameters over the bus, then send a single vector through."""
    await init(dut)

    data  =[100, -50, 200, 0, 50, 75, -10, 150]
    bias  =[10,   20, -30, 50, -5, 10,  15, -20]
    zp    =[5,     5,   5,  5,  5,  5,   5,   5]
    scale =[1 << FIXED_SHIFT] * N

    expected = scalar_pipe_ref(data, bias, zp, scale)

    # Emulate host loading variables across bus
    await load_param(dut, bias, dut.load_bias_en_i)
    await load_param(dut, zp, dut.load_zp_en_i)
    await load_param(dut, scale, dut.load_scale_en_i)

    # Emulate sys array firing
    await FallingEdge(dut.clk_i)
    drive_array(dut.data_i, data)
    dut.data_valid_i.value = 1

    await FallingEdge(dut.clk_i)
    dut.data_valid_i.value = 0

    # wait for results
    for _ in range(20):
        await RisingEdge(dut.clk_i)
        if dut.data_valid_o.value == 1:
            break
    else:
        assert False, "data_valid_o never asserted"

    got = read_array_s8(dut.data_o, N)
    for i in range(N):
        assert got[i] == expected[i], f"lane {i}: expected {expected[i]}, got {got[i]}"


@cocotb.test()
async def test_multi_vector(dut):
    """Load parameters over the bus, then push back-to-back vectors."""
    await init(dut)

    vectors = [[10, 20, 30, 40, 50, 60, 70, 80],[-5, -10, -15, -20, 100, 100, 100, 100],[0, 0, 0, 0, 0, 0, 0, 0],
    ]
    bias  =[1, 2, 3, 4, 5, 6, 7, 8]
    zp    =[2, 2, 2, 2, 2, 2, 2, 2]
    scale = [1 << FIXED_SHIFT] * N

    expected_all =[scalar_pipe_ref(v, bias, zp, scale) for v in vectors]

    # load everything sequentially across the 64 bit bus
    await load_param(dut, bias, dut.load_bias_en_i)
    await load_param(dut, zp, dut.load_zp_en_i)
    await load_param(dut, scale, dut.load_scale_en_i)

    # drive consecutive vectors
    for v in vectors:
        await FallingEdge(dut.clk_i)
        drive_array(dut.data_i, v)
        dut.data_valid_i.value = 1

    await FallingEdge(dut.clk_i)
    dut.data_valid_i.value = 0

    results =[]
    for _ in range(40):
        await RisingEdge(dut.clk_i)
        if dut.data_valid_o.value == 1:
            results.append(read_array_s8(dut.data_o, N))
            if len(results) == len(vectors):
                break

    assert len(results) == len(vectors), f"expected {len(vectors)} outputs, got {len(results)}"

    for idx, (got, exp) in enumerate(zip(results, expected_all)):
        for i in range(N):
            assert got[i] == exp[i], f"vec {idx} lane {i}: expected {exp[i]}, got {got[i]}"


# pytest
SOURCES =[
    Path("./src/scalar_units/scalar_stage.sv").resolve(),
    Path("./src/scalar_units/scalar_pipe.sv").resolve(),
    Path("./src/scalar_units/add_n.sv").resolve(),
    Path("./src/scalar_units/scale_n.sv").resolve(),
    Path("./src/scalar_units/relu_n.sv").resolve(),
    Path("./src/scalar_units/load_data.sv").resolve(),
    Path("./src/quantizer_mul.sv").resolve(),
    Path("./src/common/elastic.sv").resolve(),
    Path("./src/common/shift.sv").resolve(),
]

PARAMS = {"N": N, "PSUM_W": 32, "M0_W": 32, "FIXED_SHIFT": FIXED_SHIFT, "BUS_W": 64}

tests =[
    "test_basic",
    "test_multi_vector",
]

@pytest.mark.parametrize("testcase", tests)
def test_scalar_stage(testcase):
    try:
        run_test(
            parameters=PARAMS,
            sources=SOURCES,
            module_name="test_scalar_stage",
            hdl_toplevel="scalar_stage",
            testcase=testcase,
            sims=["icarus"],
        )
    except SystemExit as exc:
        if exc.code != 0:
            pytest.fail(f"cocotb test '{testcase}' failed (exit code {exc.code})")
