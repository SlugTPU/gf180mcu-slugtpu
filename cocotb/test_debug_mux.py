import pytest
import cocotb
from cocotb.triggers import Timer
from pathlib import Path
from runner import run_test


OOR_SENTINEL = 0xDE


def _packed(byte_values):
    """pakc list of byte values into one integer. LSB first"""
    v = 0
    for i, b in enumerate(byte_values):
        v |= (b & 0xFF) << (i * 8)
    return v


@cocotb.test()
async def in_range_test(dut):
    """every populated byte address returns the matching byte of data_i."""
    n_words = int(dut.N_WORDS.value)
    pattern = [(i + 1) & 0xFF for i in range(n_words)]  # nonzero to catch a zero default
    dut.data_i.value = _packed(pattern)

    for addr in range(n_words):
        dut.addr_i.value = addr
        await Timer(1, units="ns")
        got = int(dut.data_o.value)
        assert got == pattern[addr], f"addr={addr:#04x}: got {got:#04x}, want {pattern[addr]:#04x}"


@cocotb.test()
async def out_of_range_test(dut):
    """addresses past N_WORDS return the OOR sentinel"""
    n_words = int(dut.N_WORDS.value)
    dut.data_i.value = _packed([0x00] * n_words)
    for addr in range(n_words, 256):
        dut.addr_i.value = addr
        await Timer(1, units="ns")
        got = int(dut.data_o.value)
        assert got == OOR_SENTINEL, f"addr={addr:#04x}: got {got:#04x}, want {OOR_SENTINEL:#04x}"


@cocotb.test()
async def zero_pattern_test(dut):
    """should read back as zero, not as sentinel."""
    n_words = int(dut.N_WORDS.value)
    dut.data_i.value = _packed([0x00] * n_words)
    for addr in range(n_words):
        dut.addr_i.value = addr
        await Timer(1, units="ns")
        got = int(dut.data_o.value)
        assert got == 0x00, f"addr={addr:#04x}: zero data read back {got:#04x}"


proj_path = Path("./src").resolve()
sources = [proj_path / "debug_mux.sv"]


@pytest.mark.parametrize("n_words", [8, 32])
def test_debug_mux(n_words):
    run_test(
        parameters={"N_WORDS": n_words},
        sources=sources,
        module_name="test_debug_mux",
        hdl_toplevel="debug_mux",
    )