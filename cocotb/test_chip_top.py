import os
import random
import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, Timer, RisingEdge
from cocotbext.spi import SpiMaster, SpiBus, SpiConfig
from pathlib import Path
from shared import reset_sequence, clock_start, stringify_dict
from spibone_bfm import SpiboneBFM
from cocotb_tools.runner import get_runner

sim      = os.getenv("SIM", "icarus")
pdk_root = Path(os.getenv("PDK_ROOT", Path("~/.ciel").expanduser()))
pdk      = os.getenv("PDK", "gf180mcuD")
scl      = os.getenv("SCL", "gf180mcu_as_sc_mcu7t3v3")
gl       = os.getenv("GL", False)
slot     = os.getenv("SLOT", "1x1")

hdl_toplevel = "chip_top_sdram_tb"
module_name  = "test_chip_top"
parameters   = {"sys_clk_mhz_p": 100}


def make_spi_config(clock_freq_mhz):
    return SpiConfig(
        word_width    = 8,
        sclk_freq     = (clock_freq_mhz / 6) * 10**6,
        cpol          = False,
        cpha          = False,
        msb_first     = True,
        cs_active_low = True,
    )


# ---------------------------------------------------------------------------
# Test cases (same interface as test_tpu_soc_with_dram.py)
# ---------------------------------------------------------------------------

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
    clk_i    = dut.clk_i
    rst_i    = dut.rst_i
    spi_cs_ni = dut.spi_cs_ni

    sys_clk_mhz      = dut.sys_clk_mhz_p.value.to_unsigned()
    init_wait_cycles = (100 + 50) * sys_clk_mhz
    timeout          = init_wait_cycles

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

    addr         = 0x04
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
    clk_i    = dut.clk_i
    rst_i    = dut.rst_i
    spi_cs_ni = dut.spi_cs_ni

    sys_clk_mhz      = dut.sys_clk_mhz_p.value.to_unsigned()
    init_wait_cycles = 100 * sys_clk_mhz
    timeout          = init_wait_cycles + 50

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
    expected_vals = [
        [0x00, 0x00, 0x00, 0x00, 0xDE, 0xAD, 0xBE, 0xEF],
        [0x00, 0x00, 0x00, 0x00, 0xCA, 0xFE, 0xBA, 0xBE],
        [0xAF, 0xAB, 0xEF, 0xEA, 0x00, 0xFF, 0xBF, 0xFF],
        [0x01, 0x02, 0x03, 0x04, 0x05, 0xAB, 0xCD, 0xEF],
        [0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD, 0xFD],
        [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00, 0x11],
        [0x00, 0x11, 0x22, 0x33, 0xAA, 0xBB, 0xCC, 0xDD],
        [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xAC, 0xEF],
    ]

    await bfm.write(starting_address=starting_addr, payloads=expected_vals, timeout=timeout)

    recv = await bfm.read(starting_address=starting_addr, count=8)

    for i in range(len(expected_vals)):
        for j in range(len(expected_vals[0])):
            print(f"Got value {hex(recv[i][j])} expecting value {hex(expected_vals[i][j])}")
            assert recv[i][j] == expected_vals[i][j]

    await FallingEdge(clk_i)


@cocotb.test()
async def test_individual_writes_burst_read(dut):
    """Each word written in its own SPI transaction. read back as one burst."""
    clk_i    = dut.clk_i
    rst_i    = dut.rst_i
    spi_cs_ni = dut.spi_cs_ni

    sys_clk_mhz      = dut.sys_clk_mhz_p.value.to_unsigned()
    init_wait_cycles = 100 * sys_clk_mhz
    timeout          = init_wait_cycles + 50

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

    base  = 0x200
    words = [[random.randint(0, 255) for _ in range(8)] for _ in range(8)]

    for i, w in enumerate(words):
        await bfm.write(base + i, [w], timeout=timeout)

    got = await bfm.read(base, len(words))
    for i in range(len(got)):
        for j in range(len(words[0])):
            assert got[i][j] == words[i][j]


@cocotb.test()
async def test_overwrite(dut):
    """second write to the same address replaces the first."""
    clk_i    = dut.clk_i
    rst_i    = dut.rst_i
    spi_cs_ni = dut.spi_cs_ni

    sys_clk_mhz      = dut.sys_clk_mhz_p.value.to_unsigned()
    init_wait_cycles = 100 * sys_clk_mhz
    timeout          = init_wait_cycles + 50

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

    addr         = 0x80
    expected_val = [0x00, 0x00, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55]

    await bfm.write(addr, [[0x00, 0x00, 0x00, 0x00, 0xAA, 0xAA, 0xAA, 0xAA]], timeout=timeout)
    await bfm.write(addr, [[0x00, 0x00, 0x00, 0x00, 0x55, 0x55, 0x55, 0x55]])
    got = await bfm.read(addr, 1)

    for i in range(len(expected_val)):
        assert got[0][i] == expected_val[i]


@cocotb.test()
async def test_tpu_load_pc(dut):
    """Load in PC and see what happens"""
    clk_i    = dut.clk_i
    rst_i    = dut.rst_i
    spi_cs_ni = dut.spi_cs_ni

    sys_clk_mhz      = dut.sys_clk_mhz_p.value.to_unsigned()
    init_wait_cycles = 100 * sys_clk_mhz
    timeout          = init_wait_cycles + 50

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

    await bfm.write(0x1000_0000, [list(0x0000_0000_0000_0003.to_bytes(8, 'big'))], timeout=timeout)

    await ClockCycles(dut.clk_i, 1000)

    await FallingEdge(dut.clk_i)


@cocotb.test()
async def test_tpu_mem_basic(dut):
    """Test SPI -> DRAM, SPI PC -> TPU, TPU <-> DRAM"""
    clk_i    = dut.clk_i
    rst_i    = dut.rst_i
    spi_cs_ni = dut.spi_cs_ni

    sys_clk_mhz      = dut.sys_clk_mhz_p.value.to_unsigned()
    init_wait_cycles = 100 * sys_clk_mhz

    spi_bus = SpiBus.from_entity(
        dut,
        sclk_name="spi_clk_i",
        mosi_name="spi_mosi_i",
        miso_name="spi_miso_o",
        cs_name=None,
    )
    spi_master = SpiMaster(spi_bus, make_spi_config(sys_clk_mhz))
    bfm = SpiboneBFM(dut, clk_i, spi_cs_ni, spi_master)
    timeout = init_wait_cycles + 50

    await clock_start(clk_i, period_ns=1000/sys_clk_mhz)
    await reset_sequence(clk_i, rst_i)

    patterns  = [list((0xFFFF_FFFF_FFFF_FFFF - i).to_bytes(8, 'big')) for i in range(33)]
    base_addr = 0

    await bfm.write(base_addr, patterns, timeout=timeout)
    await bfm.write(0x1000_0000, [list(0x0000_0000_0000_0001.to_bytes(8, 'big'))], timeout=timeout)

    await ClockCycles(clk_i, 100)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

tests = [
    "reset_test",
    "test_single_write_then_read",
    "test_burst_write_then_burst_read",
    "test_individual_writes_burst_read",
    "test_overwrite",
    "test_tpu_load_pc",
    "test_tpu_mem_basic",
]


def get_sources():
    proj_path = Path(__file__).resolve().parent.parent
    src_path  = proj_path / "src"
    ip_path   = proj_path / "ip"

    slot_define = f"SLOT_{slot.upper()}"
    defines = {slot_define: True}

    if gl:
        sources = [
            pdk_root / pdk / "libs.ref" / scl / "verilog" / f"{scl}.v",
            pdk_root / pdk / "libs.ref" / scl / "verilog" / "primitives.v",
            proj_path / "final" / "pnl" / "chip_top.pnl.v",
        ]
        defines.update({"functional": True, "USE_POWER_PINS": True})
    else:
        # FUNCTIONAL=True strips specify timing from IO pad models so SPI bit-
        # sampling is not affected by pad propagation skew during RTL sim.
        # (GL mode sets this for the same reason, plus USE_POWER_PINS.)
        defines.update({"functional": True})
        sources = [
            src_path / "chip_top.sv",
            src_path / "chip_core.sv",
            src_path / "tpu_soc.sv",
            src_path / "spi" / "spibone_wb.sv",
            src_path / "spi" / "spi_slave.sv",
            src_path / "spi" / "wb_decoder.sv",
            src_path / "spi" / "tpu_regs.sv",
            src_path / "spi" / "wb_demux_1to2.sv",
            src_path / "spi" / "wb_test_ram.sv",
            src_path / "dram" / "wb_dma_master.sv",
            src_path / "dram" / "wb_mux_2to1.sv",
            src_path / "dram" / "wb_sdr_mt48lc16m16a2_7e.sv",
            src_path / "common" / "shift.sv",
            src_path / "common" / "counter.sv",
            src_path / "common" / "elastic.sv",
            src_path / "control" / "control_top.sv",
            src_path / "control" / "control_decoder.sv",
            src_path / "control" / "control_buffer.sv",
            src_path / "control" / "control_sram.sv",
            src_path / "compute_core.sv",
            src_path / "sram" / "sram_1x256.sv",
            src_path / "sram" / "sram_8x256.sv",
            src_path / "sram" / "sram_8x256_full.sv",
            src_path / "sram" / "memory_transaction.sv",
            src_path / "sysray" / "mxu.sv",
            src_path / "sysray" / "sysray_nxn.sv",
            src_path / "sysray" / "pe.sv",
            src_path / "scalar_units" / "add_n.sv",
            src_path / "scalar_units" / "relu_n.sv",
            src_path / "scalar_units" / "scale_n.sv",
            src_path / "scalar_units" / "load_data.sv",
            src_path / "scalar_units" / "scalar_pipe.sv",
            src_path / "scalar_units" / "scalar_stage.sv",
            src_path / "scalar_units" / "quantizer_mul.sv",
            src_path / "scalar_units" / "scalar_stage_sram.sv",
            src_path / "debug_mux.sv",
            src_path / "tri_shift.sv",
        ]
        defines["SIM_TOP"] = 1

    sources += [
        pdk_root / pdk / "libs.ref" / "gf180mcu_fd_io" / "verilog" / "gf180mcu_fd_io.v",
        pdk_root / pdk / "libs.ref" / "gf180mcu_ocd_ip_sram" / "verilog" / "gf180mcu_ocd_ip_sram__sram256x8m8wm1.v",
        ip_path / "gf180mcu_ws_ip__id" / "vh" / "gf180mcu_ws_ip__id.v",
        ip_path / "gf180mcu_ws_ip__logo" / "vh" / "gf180mcu_ws_ip__logo.v",
        ip_path / "gf180mcu_ws_ip__qrcode_id" / "vh" / "gf180mcu_ws_ip__qrcode_id.v",
        ip_path / "gf180mcu_ws_ip__shuttle_id" / "vh" / "gf180mcu_ws_ip__shuttle_id.v",
        ip_path / "gf180mcu_ws_ip__project_id" / "vh" / "gf180mcu_ws_ip__project_id.v",
        ip_path / "gf180mcu_ws_ip__marker" / "vh" / "gf180mcu_ws_ip__marker.v",
        src_path / "dram" / "sdram_model_mt48lc16m16a2.v",
        src_path / "chip_top_sdram_tb.sv",
    ]

    return sources, defines


def run_gl_test(testcase=None):
    sources, defines = get_sources()
    proj_path = Path(__file__).resolve().parent.parent
    includes  = [proj_path / "src"]

    case_name = testcase if testcase else "all"
    build_dir = Path("./sim_build") / sim / module_name / case_name / stringify_dict(parameters)

    runner = get_runner(sim)
    runner.build(
        sources=sources,
        hdl_toplevel=hdl_toplevel,
        defines=defines,
        always=True,
        includes=includes,
        build_dir=build_dir,
        parameters=parameters,
        waves=True,
    )
    runner.test(
        hdl_toplevel=hdl_toplevel,
        test_module=module_name,
        testcase=testcase,
        waves=True,
    )


@pytest.mark.parametrize("testcase", tests)
def test_chip_top_each(testcase):
    """Runs each test independently."""
    run_gl_test(testcase=testcase)


def test_chip_top_all():
    """Runs all tests sequentially in one simulation."""
    run_gl_test()
