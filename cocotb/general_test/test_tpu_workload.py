"""
test_tpu_workload.py — Generic TPU SoC simulation testbench.

Configuration comes from pytest CLI arguments defined in conftest.py.
The pytest entry point resolves all paths to absolute, writes them to
environment variables, then launches the simulator.  The cocotb coroutine
reads those env vars and loads the TOML/bin files directly — no
serialization or shared-dict tricks needed.

Run with:
    pytest test_tpu_workload.py -s -v               \\
        --tpu-bin         sw/matmul.bin             \\
        --tpu-dram-init   data/dram_init.toml       \\
        --tpu-expected    data/expected.toml        \\
        [--tpu-pc         0x1]                      \\
        [--tpu-post-cycles 2000]                    \\
        [--tpu-timeout-mult 1.5]

TOML file formats are documented in example_dram_init.toml /
example_expected.toml.
"""

import os
import sys

try:
    import tomllib          # Python 3.11+ stdlib
except ModuleNotFoundError:
    import tomli as tomllib # pip install tomli  (Python <= 3.10)

import pytest
import cocotb
from cocotb.triggers import ClockCycles, FallingEdge
from cocotbext.spi import SpiMaster, SpiBus, SpiConfig
from pathlib import Path

# Allow this file to live in a subdirectory while shared helpers
# (shared.py, runner.py, spibone_bfm.py) remain in the parent directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cocotb_tools.runner import get_runner
from shared import reset_sequence, clock_start, stringify_dict
from spibone_bfm import SpiboneBFM


# ---------------------------------------------------------------------------
# Simulator / PDK configuration  (mirrors test_chip_top.py)
# ---------------------------------------------------------------------------
sim      = os.getenv("SIM", "icarus")
pdk_root = Path(os.getenv("PDK_ROOT", Path("~/.ciel").expanduser()))
pdk      = os.getenv("PDK", "gf180mcuD")
scl      = os.getenv("SCL", "gf180mcu_as_sc_mcu7t3v3")
gl       = os.getenv("GL", False)
slot     = os.getenv("SLOT", "1x1")

hdl_toplevel = "chip_top_sdram_tb"
module_name  = "test_tpu_workload"
parameters   = {"sys_clk_mhz_p": 100}


# ---------------------------------------------------------------------------
# Internal env-var keys (not part of the user-facing interface)
# ---------------------------------------------------------------------------
_E_BIN          = "_TPU_BIN"
_E_DRAM_INIT    = "_TPU_DRAM_INIT"
_E_EXPECTED     = "_TPU_EXPECTED"
_E_PC           = "_TPU_PC"
_E_POST_CYCLES  = "_TPU_POST_CYCLES"
_E_TIMEOUT      = "_TPU_TIMEOUT"


# ---------------------------------------------------------------------------
# Helpers shared between pytest side and cocotb side
# ---------------------------------------------------------------------------

def _parse_addr(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return int(value, 0)


def _make_spi_config(clock_freq_mhz: int) -> SpiConfig:
    return SpiConfig(
        word_width=8,
        sclk_freq=(clock_freq_mhz / 6) * 1e6,
        cpol=False,
        cpha=False,
        msb_first=True,
        cs_active_low=True,
    )


def _init_bfm(dut) -> SpiboneBFM:
    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()
    spi_bus = SpiBus.from_entity(
        dut,
        sclk_name="spi_clk_i",
        mosi_name="spi_mosi_i",
        miso_name="spi_miso_o",
        cs_name=None,
    )
    spi_master = SpiMaster(spi_bus, _make_spi_config(sys_clk_mhz))
    return SpiboneBFM(dut, dut.clk_i, dut.spi_cs_ni, spi_master)


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _load_bin_words(path: Path) -> list[list[int]]:
    """
    Read a flat binary and return a list of 8-byte words, each byte-reversed
    (little-endian on disk -> big-endian SPI), matching the original testbench.
    """
    words = []
    with open(path, "rb") as fh:
        while chunk := fh.read(8):
            word = list(chunk.ljust(8, b"\x00"))
            word.reverse()
            words.append(word)
    return words


def _validate_word(word: list, ctx: str = "") -> list[int]:
    if len(word) != 8:
        raise ValueError(f"{ctx}: expected 8 bytes per word, got {len(word)}: {word}")
    for b in word:
        if not (0 <= int(b) <= 255):
            raise ValueError(f"{ctx}: byte value out of [0,255] range: {b}")
    return [int(b) for b in word]


def _read_dram_init(path: Path) -> list[dict]:
    """
    Parse a DRAM init TOML file and return a list of region dicts.

    TOML structure:
        [[region]]
        address = 0x0300        # native hex integer
        words = [
          [0x00, 0x01, ...],    # 8 bytes, MSB first
        ]

    Returns: [{"address": int, "words": [[byte,...], ...]}, ...]
    """
    data = _load_toml(path)
    raw  = data.get("region")
    if not isinstance(raw, list):
        raise TypeError(f"{path}: must contain one or more [[region]] tables.")
    regions = []
    for i, region in enumerate(raw):
        addr  = int(region["address"])
        words = [_validate_word(w, f"region[{i}] word[{j}]")
                 for j, w in enumerate(region["words"])]
        regions.append({"address": addr, "words": words})
        print(f"[dram_init] region {i}: addr=0x{addr:08X}, words={len(words)}")
    return regions


def _read_expected(path: Path) -> list[dict]:
    """
    Parse an expected-data TOML file and return a list of check dicts.

    TOML structure:
        [[check]]
        address    = 0x0400     # native hex integer
        word_count = 8
        expected = [            # optional; omit to read-and-print without asserting
          [0x00, ..., 0x06],
        ]

    Returns: [{"address": int, "word_count": int,
               "expected": [[byte,...], ...] | None}, ...]
    """
    data = _load_toml(path)
    raw  = data.get("check")
    if not isinstance(raw, list):
        raise TypeError(f"{path}: must contain one or more [[check]] tables.")
    checks = []
    for i, check in enumerate(raw):
        addr       = int(check["address"])
        word_count = int(check["word_count"])
        expected   = check.get("expected")
        if expected is not None:
            if len(expected) != word_count:
                raise ValueError(
                    f"{path} check[{i}]: 'expected' has {len(expected)} rows "
                    f"but word_count is {word_count}."
                )
            expected = [_validate_word(w, f"check[{i}] word[{j}]")
                        for j, w in enumerate(expected)]
        checks.append({"address": addr, "word_count": word_count, "expected": expected})
        print(
            f"[expected] check {i}: addr=0x{addr:08X}, words={word_count}, "
            f"assert={'yes' if expected is not None else 'no (dump only)'}"
        )
    return checks


# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------

_PC_REG_ADDR  = 0x1000_0000   # PC register — fixed in hardware
_BIN_LOAD_ADDR = 0x0000_0001  # instruction binary always loaded here


# ---------------------------------------------------------------------------
# cocotb test coroutine
# ---------------------------------------------------------------------------

@cocotb.test()
async def test_tpu_workload(dut):
    """
    Generic TPU SoC end-to-end test.

    Reads file paths directly from env vars set by test_tpu_workload_run(),
    then loads and parses the TOML/bin files itself.

    Steps:
      1. Load and write DRAM init regions  (optional)
      2. Load and write instruction binary
      3. Write the PC register to start execution
      4. Wait post_cycles clock cycles
      5. Read back and assert expected memory regions  (optional)
    """
    # ---- Read config from env vars ----------------------------------------
    bin_path     = Path(os.environ[_E_BIN])
    dram_init    = os.environ.get(_E_DRAM_INIT)
    expected_env = os.environ.get(_E_EXPECTED)
    pc_value     = int(os.environ[_E_PC], 0)
    post_cycles  = int(os.environ[_E_POST_CYCLES])
    timeout      = int(os.environ[_E_TIMEOUT])

    print(f"\n[config] bin        = {bin_path}")
    print(f"[config] dram_init  = {dram_init}")
    print(f"[config] expected   = {expected_env}")
    print(f"[config] pc         = 0x{pc_value:08X}")
    print(f"[config] post_cycles= {post_cycles}")

    # ---- Hardware init -------------------------------------------------------
    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()
    bfm = _init_bfm(dut)
    await clock_start(dut.clk_i, period_ns=1000 / sys_clk_mhz)
    await reset_sequence(dut.clk_i, dut.rst_i)

    # ---- Step 1: optional DRAM pre-load -------------------------------------
    if dram_init:
        for region in _read_dram_init(Path(dram_init)):
            addr, words = region["address"], region["words"]
            print(f"[dram_init] writing {len(words)} words to 0x{addr:08X}")
            await bfm.write(addr, words, timeout=timeout)
    else:
        print("[dram_init] skipped")

    # ---- Step 2: write instruction binary ------------------------------------
    instruction_words = _load_bin_words(bin_path)
    print(f"[bin] {len(instruction_words)} words -> 0x{_BIN_LOAD_ADDR:08X}")
    await bfm.write(_BIN_LOAD_ADDR, instruction_words, timeout=timeout)

    # ---- Step 3: kick off execution ------------------------------------------
    pc_word = list(pc_value.to_bytes(8, "big"))
    print(f"[pc] 0x{pc_value:016X} -> 0x{_PC_REG_ADDR:08X}")
    await bfm.write(_PC_REG_ADDR, [pc_word], timeout=timeout)

    # ---- Step 4: wait --------------------------------------------------------
    print(f"[wait] {post_cycles} clock cycles")
    await ClockCycles(dut.clk_i, post_cycles)
    assert dut.tpu_active_o.value == 0

    # ---- Step 5: read back and verify ----------------------------------------
    if expected_env:
        all_passed = True
        for check_idx, check in enumerate(_read_expected(Path(expected_env))):
            addr       = check["address"]
            word_count = check["word_count"]
            exp_words  = check["expected"]

            print(f"\n[check {check_idx}] reading {word_count} words from 0x{addr:08X}")
            got = await bfm.read(addr, word_count)

            for word_idx, got_word_bytes in enumerate(got):
                got_be  = list(reversed(got_word_bytes))
                got_str = [f"0x{b:02X}" for b in got_be]

                if exp_words is not None:
                    exp_be  = exp_words[word_idx]
                    exp_str = [f"0x{b:02X}" for b in exp_be]
                    if got_be == exp_be:
                        print(f"  word[{word_idx:02d}] got={got_str} [PASS]")
                    else:
                        print(f"  word[{word_idx:02d}] got={got_str} exp={exp_str} [FAIL]")
                        all_passed = False
                else:
                    print(f"  word[{word_idx:02d}] got={got_str} (no assertion)")

        assert all_passed, "One or more expected-data checks failed — see output above."
    else:
        print("[expected] no expected file — skipping assertions")

    await FallingEdge(dut.clk_i)


# ---------------------------------------------------------------------------
# Source list
# ---------------------------------------------------------------------------

def get_sources():
    proj_path = Path(__file__).resolve().parent.parent.parent
    src_path  = proj_path / "src"
    ip_path   = proj_path / "ip"

    defines = {"functional": True}
    slot_define = f"SLOT_{slot.upper()}"
    defines = {slot_define: True}

    if gl:
        sources = [
            pdk_root / pdk / "libs.ref" / scl / "verilog" / f"{scl}.v",
            src_path / "gl_cell_stubs.v",
            proj_path / "final" / "pnl" / "chip_top.pnl.v",
        ]
        defines["USE_POWER_PINS"] = True
    else:
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
            src_path / "dram" / "tpu_soc_with_dram.sv",
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
        defines["SIM_TOP"] = 2

    sram_model = src_path / "sram" / "gf180mcu_ocd_ip_sram__sram256x8m8wm1_behavioural.v"
    sources += [
        pdk_root / pdk / "libs.ref" / "gf180mcu_fd_io" / "verilog" / "gf180mcu_fd_io.v",
        sram_model,
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


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------

def test_tpu_workload_run(request: pytest.FixtureRequest) -> None:
    """
    Reads --tpu-* CLI options, resolves all paths to absolute (so they are
    CWD-independent in the simulator subprocess), writes them to env vars,
    then launches the simulation.

    Set GL=1 in the environment to run against the gate-level netlist instead
    of RTL sources.  PDK_ROOT, PDK, SCL are read from the environment with
    the same defaults as test_chip_top.py.
    """
    bin_path_str  = request.config.getoption("--tpu-bin")
    dram_init_str = request.config.getoption("--tpu-dram-init")
    expected_str  = request.config.getoption("--tpu-expected")
    pc_str        = request.config.getoption("--tpu-pc")
    post_cycles   = request.config.getoption("--tpu-post-cycles")
    timeout_mult  = request.config.getoption("--tpu-timeout-mult")

    # Validate and resolve to absolute paths
    bin_path = Path(bin_path_str).resolve()
    if not bin_path.exists():
        pytest.fail(f"--tpu-bin not found: {bin_path}")

    if dram_init_str:
        dram_init_path = Path(dram_init_str).resolve()
        if not dram_init_path.exists():
            pytest.fail(f"--tpu-dram-init not found: {dram_init_path}")
    else:
        dram_init_path = None

    if expected_str:
        expected_path = Path(expected_str).resolve()
        if not expected_path.exists():
            pytest.fail(f"--tpu-expected not found: {expected_path}")
    else:
        expected_path = None

    sys_clk_mhz = parameters["sys_clk_mhz_p"]
    timeout      = int(100 * sys_clk_mhz * timeout_mult) + 50

    # Write resolved values to env vars so the cocotb subprocess can read them
    os.environ[_E_BIN]         = str(bin_path)
    os.environ[_E_PC]          = pc_str
    os.environ[_E_POST_CYCLES] = str(post_cycles)
    os.environ[_E_TIMEOUT]     = str(timeout)
    if dram_init_path:
        os.environ[_E_DRAM_INIT] = str(dram_init_path)
    elif _E_DRAM_INIT in os.environ:
        del os.environ[_E_DRAM_INIT]
    if expected_path:
        os.environ[_E_EXPECTED] = str(expected_path)
    elif _E_EXPECTED in os.environ:
        del os.environ[_E_EXPECTED]

    sources, defines = get_sources()
    proj_path = Path(__file__).resolve().parent.parent.parent
    includes  = [proj_path / "src"]

    build_dir = (
        Path("./sim_build") / sim / module_name / "workload"
        / stringify_dict(parameters)
    )

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
        testcase="test_tpu_workload",
        waves=True,
    )
