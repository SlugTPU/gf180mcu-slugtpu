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

def make_spi_config(clock_freq_mhz):
    return SpiConfig(
        word_width = 8,
        sclk_freq  = (clock_freq_mhz / 6) * 10**6,
        cpol       = False,
        cpha       = False,
        msb_first  = True,
        cs_active_low = True,
    )

@cocotb.test()
async def reset_test(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()

    await clock_start(clk_i, period_ns=1000/sys_clk_mhz)
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(clk_i)

@cocotb.test()
async def test_single_write_then_read(dut):
    """write one word, read back in a separate transaction"""
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    spi_cs_ni = dut.spi_cs_ni

    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()
    init_wait_cycles = (100+50) * sys_clk_mhz
    timeout = init_wait_cycles

    spi_bus = SpiBus.from_entity(
        dut,
        sclk_name="spi_clk_i",
        mosi_name="spi_mosi_i",
        miso_name="spi_miso_o",
        cs_name=None,
    )
    spi_master = SpiMaster(spi_bus, make_spi_config(sys_clk_mhz))
    bfm = SpiboneBFM(dut, clk_i, spi_cs_ni, spi_master)

    await clock_start(clk_i, period_ns=1000/sys_clk_mhz)
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)

    await ClockCycles(clk_i, init_wait_cycles)

    addr=0x04
    expected_val = [0x00, 0x00, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF]

    cocotb.log.info("writing!")
    await bfm.write(starting_address=addr, payloads=[expected_val])
    cocotb.log.info("reading!")
    recv = (await bfm.read(starting_address=addr))[0]

    print(f"recv is {recv}")

    for i in range(len(expected_val)):
        assert recv[i] == expected_val[i]


    await FallingEdge(clk_i)

@cocotb.test()
async def test_burst_write_then_burst_read(dut):
    """burst write 8 words, burst read them back. Address auto-increments."""
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
        cs_name=None,
    )
    spi_master = SpiMaster(spi_bus, make_spi_config(sys_clk_mhz))
    bfm = SpiboneBFM(dut, clk_i, spi_cs_ni, spi_master)

    await clock_start(clk_i, period_ns=1000/sys_clk_mhz)
    await reset_sequence(clk_i, rst_i)

    starting_addr = 0x4000
    expected_vals = [[0x00, 0x00, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF],
                     [0x00, 0x00, 0x00, 0x00, 0xCA, 0xFE, 0xBA, 0xBE],
                     [0xAF, 0xAB, 0xEF, 0xEA, 0x00, 0xFF, 0xBF, 0xFF],
                     [0x01, 0x02, 0x03, 0x04, 0x05, 0xAB, 0xCD, 0xEF],
                     [0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD],
                     [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11],
                     [0x00, 0x11, 0x22, 0x33, 0xAA, 0xBB, 0xCC, 0xDD],
                     [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xAC, 0xEF]]

    await bfm.write(starting_address=starting_addr,
                    payloads=expected_vals,
                    timeout=timeout)

    recv = await bfm.read(starting_address=starting_addr, count=8)

    for i in range(len(expected_vals)):
        for j in range(len(expected_vals[0])):
            print(f"Got value {hex(recv[i][j])} expecting value {hex(expected_vals[i][j])}")
            assert recv[i][j] == expected_vals[i][j]

    await FallingEdge(clk_i)


@cocotb.test()
async def test_individual_writes_burst_read(dut):
    """Each word written in its own SPI transaction. read back as one burst."""
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

    base = 0x200
    words = [[random.randint(0, 2**8- 1) for _ in range(8)] for _ in range(8)]

    for i, w in enumerate(words):
        await bfm.write(base+i, [w], timeout=timeout)

    got = await bfm.read(base, len(words))
    for i in range(len(got)):
        for j in range(len(words[0])):
            assert got[i][j] == words[i][j]

@cocotb.test()
async def test_overwrite(dut):
    """second write to the same address replaces the first."""
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
        cs_name=None,
    )
    spi_master = SpiMaster(spi_bus, make_spi_config(sys_clk_mhz))
    bfm = SpiboneBFM(dut, clk_i, spi_cs_ni, spi_master)

    await clock_start(clk_i, period_ns=1000/sys_clk_mhz)
    await reset_sequence(clk_i, rst_i)

    addr = 0x80
    expected_val = [0x00, 0x00, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55]

    await bfm.write(addr, [[0x00, 0x00, 0x00, 0x00, 0xAA, 0xAA, 0xAA, 0xAA]], timeout=timeout)
    await bfm.write(addr, [[0x00, 0x00, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55]])
    got = await bfm.read(addr, 1)

    for i in range(len(expected_val)):
        assert got[0][i] == expected_val[i]

@cocotb.test()
async def test_tpu_load_pc(dut):
    """
    Load in PC and see what happens
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

    await bfm.write(0x1000_0000, [list(0x0000_0000_0000_0003.to_bytes(8, 'big'))], timeout=timeout)

    await ClockCycles(dut.clk_i, 1000)

    await FallingEdge(dut.clk_i)


# @cocotb.test()
# async def test_tpu_mem_basic(dut):
    # """
    # Test SPI -> DRAM, SPI PC -> TPU, TPU <-> DRAM
    # """
    # spi = await init(dut)
    # n_words   = 4
    # base_addr = 0x0000_0000
    # patterns  = [0xFFFF_FFFF_FFFF_FFFF - i for i in range (33)]

    # for i, pat in enumerate(patterns):
        # await spibone_write(spi, base_addr + i * 8, pat)

    # await spibone_write(spi, 0x1000_0000, 0x0000_0000_0000_0001)

    # await ClockCycles(dut.clk_i, 1000)

    # await FallingEdge(dut.clk_i)

# @cocotb.test()
# async def test_tpu_instructions(dut):
    # instruction_path = "../../../../../src/software/test.bin"
    # spi = await init(dut)
    # base_addr = 0x0000_0000
    # with open(instruction_path, "rb") as f:
        # i = 0
        # while data := f.read(8):
            # data = data.ljust(8, b'\x00')  # pad if final chunk < 8 bytes
            # word = struct.unpack('<Q', data)[0]
            # await spibone_write(spi, base_addr + i * 8, word)
            # print(data)
            # i += 1
    # await spibone_write(spi, 0x1000_0000, 0x0000_0000_0000_0000)
    # await ClockCycles(dut.clk_i, 1500)
    # await FallingEdge(dut.clk_i)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

tests = [
    "test_reset",
    "test_single_write_then_read",
    "test_burst_write_then_burst_read",
    "test_individual_writes_burst_read",
    "test_overwrite",
    'test_tpu_load_pc',
    # 'test_tpu_mem_basic',
    # 'test_tpu_instructions',
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
    proj_path / "sram" / "sram_1x256.sv",
    proj_path / "sram" / "sram_8x256.sv",
    proj_path / "sram" / "sram_8x256_full.sv",
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
    "ip/gf180mcu_ocd_ip_sram/cells/gf180mcu_ocd_ip_sram__sram256x8m8wm1/gf180mcu_ocd_ip_sram__sram256x8m8wm1.v"
]
parameters = {"sys_clk_mhz_p": 100}
module_name = "test_tpu_soc_with_dram"
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


def test_tpu_soc_all():
    """Runs all tests sequentially in one simulation."""
    run_test(
        parameters=parameters,
        sources=sources,
        module_name=module_name,
        hdl_toplevel=hdl_toplevel,
        sims=sims,
        sim_flag=2,
    )
