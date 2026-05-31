import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, Timer, RisingEdge, ReadOnly
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
import random

async def dram2sram_helper(dut, data):
    while dut.dma_busy_i.value != 0:
        await FallingEdge(dut.clk_i)
    dut.dma_busy_i.value = 1
    for word in data:

        dut.dram2sram_valid_i.value = 0
        delay = random.randint(9, 10)
        for _ in range(delay):
            await FallingEdge(dut.clk_i)

        dut.dram2sram_data_i.value = word
        dut.dram2sram_valid_i.value = 1
        while dut.dram2sram_ready_o.value != 1:
            await FallingEdge(dut.clk_i)
        await FallingEdge(dut.clk_i)
    dut.dram2sram_valid_i.value = 0
    dut.dma_busy_i.value = 0

# ---------------------------------------------------------------------------
# Reset helper
# ---------------------------------------------------------------------------

async def do_reset(dut):
    # DMA/DRAM control inputs
    dut.dma_busy_i.value = 0
    dut.dma_done_i.value = 0

    # DRAM->SRAM stream
    dut.dram2sram_valid_i.value = 0
    dut.dram2sram_data_i.value  = 0

    # SRAM->DRAM stream
    dut.sram2dram_ready_i.value = 0

    # SPI / PC input
    dut.pc_in.value      = 0
    dut.pc_valid_i.value = 0

    await clock_start(dut.clk_i)
    await reset_sequence(dut.clk_i, dut.rst_i)
    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_reset(dut):
    """Verify the core comes out of reset without assertions firing."""
    await do_reset(dut)
    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_pc_load(dut):
    inst = [0b0000100000011110000010000001111000001000000111100000100000011110] * 128 + [0]
    await do_reset(dut)
    await FallingEdge(dut.clk_i)
    assert dut.tpu_state_o.value == 0b01 # idle state
    dut.pc_in.value = 0b1
    dut.pc_valid_i.value = 1
    while (dut.pc_ready_o.value != 1):
        await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
    dut.pc_valid_i.value = 0
    assert dut.dram_start_o.value == 1
    assert dut.tpu_state_o.value == 0b10 # init pc
    await dram2sram_helper(dut, inst)
    for _ in range(40):
        await FallingEdge(dut.clk_i)
    # while dut.tpu_state_o.value != 0b01:
    #     await FallingEdge(dut.clk_i)
    await FallingEdge(dut.clk_i)
tests = [
    'test_reset',
    'test_pc_load',
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "control/control_top.sv",
    proj_path / "control/control_sram.sv",
    proj_path / "sram/sram_1x1024.sv",
    proj_path / "control/control_buffer.sv",
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
    proj_path / "sram/sram_8x1024_full.sv",
    proj_path / "sram/sram_8x1024.sv",
    proj_path / "sysray/sysray_nxn.sv",
    proj_path / "sysray/pe.sv",
    proj_path / "sysray/mxu.sv",
    proj_path / "common/counter.sv",
    proj_path / "tri_shift.sv",
    "./ip/gf180mcu_ocd_ip_sram.git/cells/gf180mcu_ocd_ip_sram__sram1024x8m8wm1/gf180mcu_ocd_ip_sram__sram1024x8m8wm1.v",
]


@pytest.mark.parametrize("testcase", tests)
def test_compute_core_each(testcase):
    """Runs each test independently. Continues on test failure."""
    run_test(
        parameters={},
        sources=sources,
        module_name="test_control_top",
        hdl_toplevel="control_top",
        testcase=testcase,
        sims=['icarus'],
    )


# def test_compute_core_all():
#     """Runs all tests sequentially in one simulation."""
#     run_test(
#         parameters={},
#         sources=sources,
#         module_name="test_control_top",
#         hdl_toplevel="control_top",
#         sims=['icarus'],
#     )