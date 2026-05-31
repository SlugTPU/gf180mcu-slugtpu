"""
conftest.py — pytest CLI options for the generic TPU SoC workload testbench.

Place this file alongside test_tpu_workload.py.  pytest picks it up
automatically and registers the --tpu-* options before any test runs.

Usage
-----
    pytest test_tpu_workload.py -v \
        --tpu-bin        sw/matmul.bin          \
        --tpu-dram-init  data/dram_init.json    \
        --tpu-expected   data/expected.json     \
        [--tpu-pc        0x1]                   \
        [--tpu-post-cycles 1000]                \
        [--tpu-timeout-mult 1.0]
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("tpu", "TPU SoC workload options")

    group.addoption(
        "--tpu-bin",
        metavar="PATH",
        required=True,
        help="Path to the instruction binary (.bin) file. Required.",
    )
    group.addoption(
        "--tpu-dram-init",
        metavar="PATH",
        default=None,
        help=(
            "Path to a JSON file describing DRAM regions to write before "
            "execution starts. Optional."
        ),
    )
    group.addoption(
        "--tpu-expected",
        metavar="PATH",
        default=None,
        help=(
            "Path to a JSON file describing memory regions to read back and "
            "compare after execution. Optional — omit for a smoke test."
        ),
    )
    group.addoption(
        "--tpu-pc",
        metavar="ADDR",
        default="0x1",
        help="Program-counter value written to the PC register (hex or dec). Default: 0x1.",
    )
    group.addoption(
        "--tpu-post-cycles",
        metavar="N",
        type=int,
        default=1000,
        help="Clock cycles to wait after writing PC before reading results. Default: 1000.",
    )
    group.addoption(
        "--tpu-timeout-mult",
        metavar="FLOAT",
        type=float,
        default=1.0,
        help="SPI timeout multiplier applied to the base 100-µs budget. Default: 1.0.",
    )