from __future__ import annotations
import os
import random
from pathlib import Path
import cocotb
from cocotb_tools.runner import get_runner
import pytest
import sys
import importlib
from shared import stringify_dict

LANGUAGE = os.getenv("HDL_TOPLEVEL_LANG", "verilog").lower().strip()

def run_test(parameters, sources, module_name, hdl_toplevel, testcase=None, sims = ["icarus", "verilator"]):
    timescale = ("1ps","1ps")
    case_name = "all"

    if testcase is not None:
        case_name = testcase

    for sim in sims:
        build_dir = Path("./sim_build", sim, module_name, case_name, stringify_dict(parameters))
        build_args = []
        test_args = []

        # extra stuff specifically for verilator
        if (sim == "verilator"):
            build_args.append("--trace")
            build_args.append("--trace-structs")
            build_args.append("--trace-fst")
            test_args = build_args.copy()

        runner = get_runner(sim)
        runner.build(
            sources=sources,
            hdl_toplevel=hdl_toplevel,
            always=True,
            timescale=timescale,
            build_dir=build_dir,
            parameters=parameters,
            build_args=build_args,
            verbose=True,
            waves=True
        )

        print(f"Running test '{case_name}' with {sim}...")
        print(f"Build command: {runner._build_command()}")

        try:
            runner.test(testcase=testcase, test_args=test_args, hdl_toplevel=hdl_toplevel, test_module=module_name, waves=True)
        except:
            print(f"Test '{case_name}' with {sim} failed")
