import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer, RisingEdge, ReadOnly
from pathlib import Path
from shared import reset_sequence, clock_start, handshake
from runner import run_test
import random, math

from test_compute_core import scalar_pipe_ref, tiled_matmul_saturating_ref

# ---------------------------------------------------------------------------
# Reset helper
# ---------------------------------------------------------------------------

async def do_reset(dut):
    # Instruction interface
    dut.instruction_data_i.value  = 0
    dut.instruction_valid_i.value = 0

    # DMA/DRAM control inputs
    dut.dma_busy.value = 0
    dut.dma_done.value = 0

    # DRAM->SRAM stream
    dut.dram2sram_valid_i.value = 0
    dut.dram2sram_data_i.value  = 0

    # SRAM->DRAM stream
    dut.sram2dram_ready_i.value = 0

    # TPU state
    dut.tpu_state_i.value = 0

    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)
    await FallingEdge(dut.clk_i)

async def instruction_load_helper(dut, instructions):
    for instruction in instructions:
        for word in instruction:
            while dut.instruction_ready_o.value != 1:
                await FallingEdge(dut.clk_i)
            await FallingEdge(dut.clk_i)
            dut.instruction_data_i.value = word
            dut.instruction_valid_i.value = 1
        await FallingEdge(dut.clk_i)
    dut.instruction_valid_i.value = 0

async def dram2sram_helper(dut, data):
    while dut.dma_busy.value != 0:
        await FallingEdge(dut.clk_i)
    dut.dma_busy.value = 1
    for word in data:

        dut.dram2sram_valid_i.value = 0
        delay = random.randint(3, 6)
        for _ in range(delay):
            await FallingEdge(dut.clk_i)

        dut.dram2sram_data_i.value = word
        dut.dram2sram_valid_i.value = 1
        while dut.dram2sram_ready_o.value != 1:
            await FallingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
    dut.dram2sram_valid_i.value = 0
    dut.dma_busy.value = 0

def flatten_matrix(matrix):
    matrix_packed = []
    for row in matrix:
        packed = 0
        for i, val in enumerate(row):
            packed |= (int(val) & 0xFF) << (i * 8)
        matrix_packed.append(packed)
    return matrix_packed

def init_act_data(scalar_params, matrix):
    bias  = scalar_params['bias']
    zp    = scalar_params['zp']
    mul   = scalar_params['mul']

    MASK32 = 0xFFFFFFFF

    bias_packed = [((b1 & MASK32) | ((b2 & MASK32) << 32)) for b1, b2 in zip(bias[0::2], bias[1::2])]
    zp_packed   = [((z1 & MASK32) | ((z2 & MASK32) << 32)) for z1, z2 in zip(zp[0::2],   zp[1::2])]
    mul_packed  = [((m1 & MASK32) | ((m2 & MASK32) << 32)) for m1, m2 in zip(mul[0::2],  mul[1::2])]

    words = bias_packed + zp_packed + mul_packed + flatten_matrix(matrix)
    return words


async def sram2dram_helper(dut, data):
    while dut.dma_busy.value != 0:
        await FallingEdge(dut.clk_i)
    dut.dma_busy.value = 1
    for word in data:
        dut.dma_busy.value = 1
        dut.sram2dram_ready_i.value = 0
        delay = random.randint(3, 10)
        for _ in range(delay):
            await FallingEdge(dut.clk_i)
        dut.sram2dram_ready_i.value = 1
        while dut.sram2dram_valid_o.value != 1:
            await FallingEdge(dut.clk_i)
        assert dut.sram2dram_data_o.value == word
        await FallingEdge(dut.clk_i)
    dut.sram2dram_ready_i.value = 0
    dut.dma_busy.value = 0

@cocotb.test()
async def test_reset(dut):
    """Verify the core comes out of reset without assertions firing."""
    await do_reset(dut)
    await FallingEdge(dut.clk_i)
    assert dut.instruction_ready_o.value == 1


@cocotb.test()
async def test_exit(dut):
    """Verify that we can exit"""
    await do_reset(dut)
    await FallingEdge(dut.clk_i)
    dut.dma_busy.value = 0
    dut.tpu_state_i.value = 3
    assert dut.instruction_ready_o.value == 1
    instructions = [[0b00000000, 0b00000000]]
    await instruction_load_helper(dut, instructions)
            
  #  await FallingEdge(dut.clk_i)
    assert dut.tpu_exit_o.value == 1
    dut.instruction_valid_i.value = 0
    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_load_basic(dut):
    """Verify that we can load variable length instructions"""
    await do_reset(dut)
    await FallingEdge(dut.clk_i)
    dut.dma_busy.value = 0
    dut.tpu_state_i.value = 3
    instructions = [[0b00011111, 0b00010000, 0b00000000, 0b11111111]]
    assert dut.instruction_ready_o.value == 1
    await instruction_load_helper(dut, instructions)
    await FallingEdge(dut.clk_i)
    assert dut.inst_q.value == 0xff00101f
    print("got here")
    instructions = [
        [0b00100101, 0b00010000, 0b00000000, 0b11111111], # DRAM to weight
        [0b00111000, 0b00000000]] # load bias, should get stuck in wait state
    await instruction_load_helper(dut, instructions)
    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_dram_inst(dut):
    """Test all 3 dram instructions"""
    await do_reset(dut)
    await FallingEdge(dut.clk_i)
    dut.dma_busy.value = 0
    dut.tpu_state_i.value = 3
    amount = 8
    data = [random.randint(0, math.exp2(64)-1) for _ in range(amount)]
    instructions = [[0b00011101, 0b00010000, 0b00000000, 0b00001000]]
    assert dut.instruction_ready_o.value == 1
    await instruction_load_helper(dut, instructions)
    await FallingEdge(dut.clk_i)
    cocotb.start_soon(dram2sram_helper(dut, data))
    instructions = [[0b00011111, 0b00010000, 0b00000000, 0b00001000]]
    await instruction_load_helper(dut, instructions)
    while(dut.dma_busy.value != 0):
        await FallingEdge(dut.clk_i)
    cocotb.start_soon(sram2dram_helper(dut, data))
    await FallingEdge(dut.clk_i)
    instructions = [
        [0b00100101, 0b00010000, 0b00000000, 0b11111111], # DRAM to weight
        [0b00111000, 0b00000000]]
    await instruction_load_helper(dut, instructions)
    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_single_layer(dut):
    await do_reset(dut)
    await FallingEdge(dut.clk_i)
    dut.dma_busy.value = 0  
    dut.tpu_state_i.value = 3

    bias = [1, 2, 3, 4, 5, 6, 7, 8]
    zp   = [-2, -2, -4, -4, -6, -6, -8, -8]
    # bias = [0,0,0,0,0,0,0,10]
    # zp   = [-1,0,-1,0,-1,0,-1,0]
    mul  = [1 << 31 for _ in range(8)]
    scalar_params = {'bias': bias, 'zp': zp, 'mul': mul} 
    act    = [[random.randint(0, 3) for _ in range(8)] for _ in range(8)]
    weight = [[random.randint(0, 3) for _ in range(8)] for _ in range(8)]
    weight_flipped = weight[::-1]
    act_data = init_act_data(scalar_params, act)
    weight_data = flatten_matrix(weight_flipped)

    expected_matmul = tiled_matmul_saturating_ref([act], [weight])
    expected_output = []
    for row in expected_matmul:
        expected_output.append(scalar_pipe_ref(row, bias, zp, mul))
    print("=============Activations=============\n", act)
    print("=============Weight=============\n", weight)
    print("=============Weight Flipped=============\n", weight_flipped)
    print("=============MATMUL_EXP=============\n", expected_matmul)
    print("=============OUTPUT_EXP=============\n", expected_output)
    print("length act data: ", len(act_data))
    print("length weight data: ", len(weight_data))
    expected_flat = flatten_matrix(expected_output)
    for data in expected_flat:
        print(f'{data:#018x}')

    instructions = [[0b00011101, 0b00010000, 0b00000000, 0b00010100]]
    assert dut.instruction_ready_o.value == 1
    await instruction_load_helper(dut, instructions)
    await FallingEdge(dut.clk_i)
    cocotb.start_soon(dram2sram_helper(dut, act_data))
    instructions = [[0b00010101, 0b00010000, 0b00000000, 0b00001000]]
    await instruction_load_helper(dut, instructions)
    await FallingEdge(dut.clk_i)
    cocotb.start_soon(dram2sram_helper(dut, weight_data))
    instructions = [
        [0b00011110, 0b00001000], # pipeline setup
        [0b00011000, 0b00000000], # bias
        [0b01011100, 0b00000000], # zp
        [0b10011010, 0b00000000], # scale
        [0b00010110, 0b00000000], # load weights
        [0b11010001, 0b00000000, 0b00001000], # do matmul
        [0b00001111, 0b00001000, 0b00000000, 0b00001000], # return results to dram
    ]
    await instruction_load_helper(dut, instructions)
    while dut.weight_load_ready.value != 1:
        await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    while dut.weight_load_ready.value != 1:
        await FallingEdge(dut.clk_i)
    
    while dut.act_load_ready.value != 1:
        await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    cocotb.start_soon(sram2dram_helper(dut, expected_flat))
    for _ in range(20):
        await FallingEdge(dut.clk_i)
    while dut.act_load_ready.value != 1:
        await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)


tests = [
    'test_reset',
    'test_exit',
    'test_load_basic',
    'test_dram_inst',
    'test_single_layer',
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "control/control_decoder.sv",
    proj_path / "compute_core.sv",
    proj_path / "scalar_units/scalar_stage_sram.sv",
    proj_path / "scalar_units/scalar_stage.sv",
    proj_path / "scalar_units/scalar_pipe.sv",
    proj_path / "scalar_units/add_n.sv",
    proj_path / "scalar_units/scale_n.sv",
    proj_path / "scalar_units/relu_n.sv",
    proj_path / "scalar_units/load_data.sv",
    proj_path / "scalar_units/quantizer_mul.sv",
    proj_path / "common/elastic.sv",
    proj_path / "common/shift.sv",
    proj_path / "sram/memory_transaction.sv",
    proj_path / "sram/sram_8x256_full.sv",
    proj_path / "sram/sram_8x256.sv",
    proj_path / "sysray/sysray_nxn.sv",
    proj_path / "sysray/pe.sv",
    proj_path / "sysray/mxu.sv",
    proj_path / "common/counter.sv",
    proj_path / "tri_shift.sv",
    "./ip/gf180mcu_ocd_ip_sram.git/cells/gf180mcu_ocd_ip_sram__sram256x8m8wm1/gf180mcu_ocd_ip_sram__sram256x8m8wm1.v",
]


@pytest.mark.parametrize("testcase", tests)
def test_compute_core_each(testcase):
    """Runs each test independently. Continues on test failure."""
    run_test(
        parameters={},
        sources=sources,
        module_name="test_control_decoder",
        hdl_toplevel="control_decoder",
        testcase=testcase,
        sims=['icarus'],
    )


def test_compute_core_all():
    """Runs all tests sequentially in one simulation."""
    run_test(
        parameters={},
        sources=sources,
        module_name="test_control_decoder",
        hdl_toplevel="control_decoder",
        sims=['icarus'],
    )