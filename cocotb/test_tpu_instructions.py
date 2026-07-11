import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, Timer, RisingEdge
from cocotbext.spi import SpiMaster, SpiBus, SpiConfig
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
from spibone_bfm import SpiboneBFM
import random
from test_control_decoder import flatten_matrix, init_act_data, scalar_pipe_ref, tiled_matmul_saturating_ref

def make_spi_config(clock_freq_mhz):
    return SpiConfig(
        word_width = 8,
        sclk_freq  = (clock_freq_mhz / 6) * 10**6,
        cpol       = False,
        cpha       = False,
        msb_first  = True,
        cs_active_low = True,
    )

def init_bfm(dut):
    clk_i = dut.clk_i
    spi_cs_ni = dut.spi_cs_ni

    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()

    spi_bus = SpiBus.from_entity(
        dut,
        sclk_name="spi_clk_i",
        mosi_name="spi_mosi_i",
        miso_name="spi_miso_o",
        cs_name=None, # cs is driven manually in SpiboneBFM
    )
    spi_master = SpiMaster(spi_bus, make_spi_config(sys_clk_mhz))
    bfm = SpiboneBFM(dut, clk_i, spi_cs_ni, spi_master)
    return bfm

def data_to_bytes(data):
    result = []
    for row in data:
        result.append(list(row.to_bytes(8, 'big')))
    return result

@cocotb.test()
async def test_tpu_mem_basic(dut):
    """
    Test SPI -> DRAM, SPI PC -> TPU, TPU <-> DRAM
    """
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    spi_cs_ni = dut.spi_cs_ni

    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()
    init_wait_cycles = 100 * sys_clk_mhz
    timeout = init_wait_cycles + 50

    spi_bus = SpiBus.from_entity(
        dut,
        sclk_name="spi_clk_i",
        mosi_name="spi_mosi_i",
        miso_name="spi_miso_o",
        cs_name=None, # cs is driven manually in SpiboneBFM
    )
    spi_master = SpiMaster(spi_bus, make_spi_config(sys_clk_mhz))
    bfm = SpiboneBFM(dut, clk_i, spi_cs_ni, spi_master)

    await clock_start(clk_i, period_ns=1000/sys_clk_mhz)
    await reset_sequence(clk_i, rst_i)

    base_addr = 0
    patterns  = [list((0xFFFF_FFFF_FFFF_FFFF - i).to_bytes(8, 'big')) for i in range (32)]

    await bfm.write(base_addr, patterns, timeout=timeout)

    await bfm.write(0x1000_0000, [list(0x0000_0000_0000_0001.to_bytes(8, 'big'))], timeout=timeout)

    await ClockCycles(dut.clk_i, 1000)

    await FallingEdge(dut.clk_i)

@cocotb.test()
async def test_tpu_instructions(dut):
    instruction_path = "../../../../../src/software/test.bin"

    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()
    init_wait_cycles = 100 * sys_clk_mhz
    timeout = init_wait_cycles + 50
    
    bfm = init_bfm(dut)

    await clock_start(dut.clk_i, period_ns=1000/sys_clk_mhz)
    await reset_sequence(dut.clk_i, dut.rst_i)

    bias = [1, 2, 3, 4, 5, 6, 7, 8]
    zp   = [-2, -2, -4, -4, -6, -6, -8, -8]
    mul  = [1 << 31 for _ in range(8)]
    scalar_params = {'bias': bias, 'zp': zp, 'mul': mul} 
    act    = [[random.randint(0, 3) for _ in range(8)] for _ in range(8)]
    weight = [[random.randint(0, 3) for _ in range(8)] for _ in range(8)]
    weight_flipped = weight[::-1]
    scalar_data = init_act_data(scalar_params, act)
    weight_matrix = flatten_matrix(weight_flipped)

    await bfm.write(0x300, data_to_bytes(scalar_data), timeout=timeout)
    await bfm.write(0x100, data_to_bytes(weight_matrix), timeout=timeout)

    base_addr = 0x0000_0001
    words = []
    with open(instruction_path, "rb") as f:
        i = 0
        while data := f.read(8):
            print(data)
            data = data.ljust(8, b'\x00')  # pad if final chunk < 8 bytes
            word = list(data)
            word.reverse()
            words.append(word)
            print(word)
            i += 1
    await bfm.write(base_addr, words, timeout=timeout)
    await bfm.write(0x1000_0000, [list(0x0000_0000_0000_0001.to_bytes(8, 'big'))], timeout=timeout)
    await ClockCycles(dut.clk_i, 2500)
    got = await bfm.read(0x400, 8)
    expected_matmul = tiled_matmul_saturating_ref([act], [weight])
    expected_output = []
    for row in expected_matmul:
        expected_output.append(scalar_pipe_ref(row, bias, zp, mul))
    assert len(got) == len(expected_output)
    print(got)
    print(expected_output)
    for i, got_row_bytes in enumerate(got):
        got_row_bytes.reverse()
        for j, got_elem_bytes in enumerate(got_row_bytes):
            expected_elem = expected_output[i][j]
            got_elem = int(got_elem_bytes)
            assert got_elem == expected_elem
    await FallingEdge(dut.clk_i)

tests = [
    # 'test_tpu_mem_basic',
    'test_tpu_instructions',
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "tpu_soc.sv",
    proj_path / "spi" / "spibone_wb.sv",
    proj_path / "spi" / "spi_slave.sv",
    proj_path / "spi" / "wb_decoder.sv",
    proj_path / "spi" / "tpu_regs.sv",
    proj_path / "spi" / "wb_demux_1to2.sv",
    proj_path / "spi" / "wb_test_ram.sv",
    proj_path / "dram" / "tpu_soc_with_dram.sv",
    proj_path / "dram" / "wb_dma_master.sv",
    proj_path / "dram" / "wb_mux_2to1.sv",
    proj_path / "dram" / "wb_sdr_mt48lc16m16a2_7e.sv",
    proj_path / "dram" / "sdram_model_mt48lc16m16a2.v",
    proj_path / "common" / "shift.sv",
    proj_path / "common" / "counter.sv",
    proj_path / "common" / "elastic.sv",
    proj_path / "control" / "control_top.sv",
    proj_path / "control" / "control_decoder.sv",
    proj_path / "control" / "control_buffer.sv",
    proj_path / "control" / "control_sram.sv",
    proj_path / "compute_core.sv",
    proj_path / "sram" / "sram_1x1024.sv",
    proj_path / "sram" / "sram_8x1024.sv",
    proj_path / "sram" / "sram_8x1024_full.sv",
    proj_path / "sram" / "memory_transaction.sv",
    proj_path / "sysray" / "mxu.sv",
    proj_path / "sysray" / "sysray_nxn.sv",
    proj_path / "sysray" / "pe.sv",
    proj_path / "scalar_units" / "add_n.sv",
    proj_path / "scalar_units" / "relu_n.sv",
    proj_path / "scalar_units" / "scale_n.sv",
    proj_path / "scalar_units" / "load_data.sv",
    proj_path / "scalar_units" / "scalar_pipe.sv",
    proj_path / "scalar_units" / "scalar_stage.sv",
    proj_path / "scalar_units" / "quantizer_mul.sv",
    proj_path / "scalar_units" / "scalar_stage_sram.sv",
    proj_path / "debug_mux.sv",
    proj_path / "tri_shift.sv",
    "ip/gf180mcu_ocd_ip_sram/cells/gf180mcu_ocd_ip_sram__sram1024x8m8wm1/gf180mcu_ocd_ip_sram__sram1024x8m8wm1.v"
]
parameters = {"sys_clk_mhz_p": 100}
module_name = "test_tpu_instructions"
hdl_toplevel = "tpu_soc_sdram_tb"
sims = ["icarus"]
# note: verilator doesn't like specify blocks in the SDRAM controller


@pytest.mark.parametrize("testcase", tests)
def test_tpu_soc_each(testcase):
    """Runs each test independently. Continues on test failure."""
    run_test(
        parameters=parameters,
        sources=sources,
        module_name=module_name,
        hdl_toplevel=hdl_toplevel,
        testcase=testcase,
        sims=sims,
        sim_flag=2,
    )


# def test_tpu_soc_all():
#     """Runs all tests sequentially in one simulation."""
#     run_test(
#         parameters=parameters,
#         sources=sources,
#         module_name=module_name,
#         hdl_toplevel=hdl_toplevel,
#         sims=sims,
#         sim_flag=2,
#     )