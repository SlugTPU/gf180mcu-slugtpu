import pytest
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, Timer, RisingEdge, First
from pathlib import Path
from shared import reset_sequence, clock_start
from runner import run_test
import random

@cocotb.test()
async def test_reset(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()

    print(f"DEBUG: Passed in clock is {1 // sys_clk_mhz}ns")
    Clock(clk_i, 1/sys_clk_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(clk_i)


@cocotb.test()
async def test_initialization(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    m_we_i = dut.m_we_i
    m_stb_i = dut.m_stb_i
    m_cyc_i = dut.m_cyc_i
    m_sel_i = dut.m_sel_i

    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()

    Clock(clk_i, 1/sys_clk_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)
    m_we_i.value = 0
    m_stb_i.value = 0
    m_cyc_i.value = 0
    m_sel_i.value = 0

    """
    since vanilla Wishbone does not really have a ready_o interface without
    sending a read/write request, waiting 200us will do
    """
    await ClockCycles(clk_i, 200 * sys_clk_mhz)

    await FallingEdge(clk_i)

@cocotb.test()
async def test_read(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    m_we_i = dut.m_we_i
    m_stb_i = dut.m_stb_i
    m_cyc_i = dut.m_cyc_i
    m_sel_i = dut.m_sel_i
    m_ack_o = dut.m_ack_o
    m_adr_i = dut.m_adr_i

    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()

    Clock(clk_i, 1/sys_clk_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)

    await FallingEdge(rst_i)
    m_we_i.value = 0
    m_stb_i.value = 1
    m_cyc_i.value = 1
    m_sel_i.value = 0
    m_adr_i.value = 1

    timeout = Timer(1000, unit="us")
    m_ack = RisingEdge(m_ack_o)

    r = await First(timeout, m_ack)
    if r is timeout:
        assert 1 == 0, "Timed out while waiting for m_ack_o to go high"

    await FallingEdge(clk_i)
    m_we_i.value = 0
    m_stb_i.value = 0
    m_cyc_i.value = 0
    m_sel_i.value = 0
    m_adr_i.value = 0
    await RisingEdge(clk_i)

    # go quiet for a bit for checking autorefresh
    await ClockCycles(clk_i, 20)

@cocotb.test()
async def test_write(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    m_we_i = dut.m_we_i
    m_stb_i = dut.m_stb_i
    m_cyc_i = dut.m_cyc_i
    m_sel_i = dut.m_sel_i
    m_ack_o = dut.m_ack_o
    m_adr_i = dut.m_adr_i
    m_dat_i = dut.m_dat_i
    m_dat_o = dut.m_dat_o

    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()

    Clock(clk_i, 1/sys_clk_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)

    expected_value = 0xCAFEDEADBEEF

    # WRITE 0xCAFEDEADBEEF
    await FallingEdge(rst_i)
    m_we_i.value = 1
    m_stb_i.value = 1
    m_cyc_i.value = 1
    m_adr_i.value = 1
    # ignored
    m_sel_i.value = 0
    m_dat_i.value = expected_value

    t0 = Timer(1000, unit="us")
    ma0 = RisingEdge(m_ack_o)

    r0 = await First(t0, ma0)
    if r0 is t0:
        assert 1 == 0, "Timed out while waiting for m_ack_o to go high"

    await RisingEdge(clk_i)

    # READ and confirm it was properly written to RAM
    await FallingEdge(clk_i)
    m_we_i.value = 0
    m_stb_i.value = 1
    m_cyc_i.value = 1
    m_sel_i.value = 1
    m_adr_i.value = 1
    m_dat_i.value = 0

    t1 = Timer(1000, unit="us")
    ma1 = RisingEdge(m_ack_o)

    r1 = await First(t1, ma1)
    if r1 is t1:
        assert 1 == 0, "Timed out while waiting for m_ack_o to go high"

    assert (m_dat_o.value.to_unsigned() == expected_value), f"Tried to retrieve expected value written ({expected_value}), but got {m_dat_o.value.to_unsigned()}"

    await FallingEdge(clk_i)
    m_we_i.value = 0
    m_stb_i.value = 0
    m_cyc_i.value = 0
    m_sel_i.value = 0
    m_adr_i.value = 0

    # go quiet for a bit for checking autorefresh
    await ClockCycles(clk_i, 1000)

async def wb_write(dut, addr, data, timeout_us=1000):
    """Wishbone write; returns True on ack, False on timeout."""
    clk_i   = dut.clk_i
    m_ack_o = dut.m_ack_o

    await FallingEdge(clk_i)
    dut.m_we_i.value  = 1
    dut.m_stb_i.value = 1
    dut.m_cyc_i.value = 1
    dut.m_sel_i.value = 0
    dut.m_adr_i.value = addr
    dut.m_dat_i.value = data

    t  = Timer(timeout_us, unit="us")
    ma = RisingEdge(m_ack_o)
    r  = await First(t, ma)

    await RisingEdge(clk_i)
    dut.m_we_i.value  = 0
    dut.m_stb_i.value = 0
    dut.m_cyc_i.value = 0
    dut.m_adr_i.value = 0
    dut.m_dat_i.value = 0

    return r is not t


async def wb_read(dut, addr, timeout_us=1000):
    """Wishbone read; returns (True, data) on ack, (False, None) on timeout."""
    clk_i   = dut.clk_i
    m_ack_o = dut.m_ack_o
    m_dat_o = dut.m_dat_o

    await FallingEdge(clk_i)
    dut.m_we_i.value  = 0
    dut.m_stb_i.value = 1
    dut.m_cyc_i.value = 1
    dut.m_sel_i.value = 1
    dut.m_adr_i.value = addr
    dut.m_dat_i.value = 0

    t  = Timer(timeout_us, unit="us")
    ma = RisingEdge(m_ack_o)
    r  = await First(t, ma)

    await FallingEdge(clk_i)
    data = m_dat_o.value.to_unsigned() if r is not t else None

    dut.m_we_i.value  = 0
    dut.m_stb_i.value = 0
    dut.m_cyc_i.value = 0
    dut.m_sel_i.value = 0
    dut.m_adr_i.value = 0

    await RisingEdge(clk_i)
    return (r is not t), data


# Build a user address matching the controller's bit-field layout:
#   addr_bank_w = m_adr_i[(log2(cols) - log2(burst)) + log2(banks) - 1 -: log2(banks)]
#   addr_row_w  = m_adr_i[usr_addr_bits - 1 -: log2(rows)]
#   addr_col_w  = m_adr_i[log2(cols) - 1 : 0]   (overlaps bank bits at the top)
def make_addr(row, bank, col=0, burst_p=4, cols_p=512, banks_p=4):
    import math
    bank_shift = int(math.log2(cols_p)) - int(math.log2(burst_p))
    row_shift  = bank_shift + int(math.log2(banks_p))
    col_mask   = (1 << bank_shift) - 1
    return (row << row_shift) | (bank << bank_shift) | (col & col_mask)


@cocotb.test()
async def test_bank_switch_same_row(dut):
    """Write distinct values to all 4 banks at the same row, then read back.

    After the first ACTIVE, subsequent writes to different banks on the same
    row take the IDLE -> ACTIVE path (no PRECHARGE needed).
    """
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()
    burst_p     = dut.dram_burst_p.value.to_unsigned()
    cols_p      = dut.cols_p.value.to_unsigned()
    banks_p     = dut.banks_p.value.to_unsigned()
    bus_mask    = (1 << (dut.data_bits_p.value.to_unsigned() * burst_p)) - 1

    Clock(clk_i, 1 / sys_clk_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)

    values = [random.getrandbits(32) & bus_mask for _ in range(banks_p)]
    addrs  = [make_addr(row=0, bank=b, burst_p=burst_p, cols_p=cols_p, banks_p=banks_p)
              for b in range(banks_p)]

    for addr, val in zip(addrs, values):
        ok = await wb_write(dut, addr, val)
        assert ok, f"Write to addr {hex(addr)} timed out"

    for addr, expected in zip(addrs, values):
        ok, got = await wb_read(dut, addr)
        assert ok, f"Read from addr {hex(addr)} timed out"
        assert got == expected, (
            f"addr {hex(addr)}: expected {hex(expected)}, got {hex(got)}"
        )

    await ClockCycles(clk_i, 20)


@cocotb.test()
async def test_bank_switch_diff_row(dut):
    """Write to each bank at two different rows, forcing a single-bank PRECHARGE on the row switch.

    Round 1 activates each bank at rows_a[b].  Round 2 writes the same banks
    at rows_b[b] != rows_a[b], which requires IDLE -> PRECHARGE_SINGLE -> ACTIVE
    for every bank (bank open, wrong row).  Only the target bank is closed;
    other banks remain open.  Read-back of round-2 values verifies the full
    PRECHARGE_SINGLE -> ACTIVE -> READ path as well.
    """
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()
    burst_p     = dut.dram_burst_p.value.to_unsigned()
    cols_p      = dut.cols_p.value.to_unsigned()
    banks_p     = dut.banks_p.value.to_unsigned()
    bus_mask    = (1 << (dut.data_bits_p.value.to_unsigned() * burst_p)) - 1

    Clock(clk_i, 1 / sys_clk_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)
    await FallingEdge(rst_i)

    rows_a = [0, 1, 5, 3]
    rows_b = [2, 4, 0, 7]

    def entries_for_rows(rows):
        return [
            (make_addr(row=rows[b], bank=b, burst_p=burst_p, cols_p=cols_p, banks_p=banks_p),
             random.getrandbits(32) & bus_mask)
            for b in range(banks_p)
        ]

    round1 = entries_for_rows(rows_a)
    round2 = entries_for_rows(rows_b)

    # Round 1: activate each bank at rows_a (no precharge needed yet)
    for addr, val in round1:
        ok = await wb_write(dut, addr, val)
        assert ok, f"Round-1 write to addr {hex(addr)} timed out"

    # Round 2: write same banks at different rows → forces PRECHARGE per bank
    for addr, val in round2:
        ok = await wb_write(dut, addr, val)
        assert ok, f"Round-2 write to addr {hex(addr)} timed out"

    # Read back round-2 values; each bank is still open at rows_b so these are
    # page hits, but the first read after the last write flushes via IDLE anyway.
    for addr, expected in round2:
        ok, got = await wb_read(dut, addr)
        assert ok, f"Round-2 read from addr {hex(addr)} timed out"
        assert got == expected, (
            f"addr {hex(addr)}: expected {hex(expected)}, got {hex(got)}"
        )

    await ClockCycles(clk_i, 20)


@cocotb.test()
async def test_random_access(dut):
    clk_i = dut.clk_i
    rst_i = dut.rst_i
    m_we_i = dut.m_we_i
    m_stb_i = dut.m_stb_i
    m_cyc_i = dut.m_cyc_i
    m_sel_i = dut.m_sel_i
    m_ack_o = dut.m_ack_o
    m_adr_i = dut.m_adr_i
    m_dat_i = dut.m_dat_i
    m_dat_o = dut.m_dat_o

    sys_clk_mhz = dut.sys_clk_mhz_p.value.to_unsigned()

    Clock(clk_i, 1/sys_clk_mhz, unit="us").start()
    await reset_sequence(clk_i, rst_i)

    expected_value0 = 0xCAFEDEADBEEF
    expected_at0 = 0x004D00
    expected_value1 = 0x6789ABCDEF
    expected_at1 = 0x000003

    # BEGIN writing values at different rows

    await FallingEdge(rst_i)
    m_we_i.value = 1
    m_stb_i.value = 1
    m_cyc_i.value = 1
    m_adr_i.value = expected_at0
    m_sel_i.value = 0 # ignored
    m_dat_i.value = expected_value0
    t = Timer(1000, unit="us")
    ma = RisingEdge(m_ack_o)
    r = await First(t, ma)
    if r is t:
        assert 1 == 0, "Timed out while waiting for m_ack_o to go high"

    await RisingEdge(clk_i)

    await FallingEdge(clk_i)
    m_we_i.value = 1
    m_stb_i.value = 1
    m_cyc_i.value = 1
    m_adr_i.value = expected_at1
    m_sel_i.value = 0 # ignored
    m_dat_i.value = expected_value1
    t = Timer(1000, unit="us")
    ma = RisingEdge(m_ack_o)
    r = await First(t, ma)
    if r is t:
        assert 1 == 0, "Timed out while waiting for m_ack_o to go high"

    await RisingEdge(clk_i)

    # END writing values at different rows

    # BEGIN reading values at different rows

    await FallingEdge(clk_i)
    m_we_i.value = 0
    m_stb_i.value = 1
    m_cyc_i.value = 1
    m_sel_i.value = 1
    m_adr_i.value = expected_at0
    m_dat_i.value = 0

    t = Timer(1000, unit="us")
    ma = RisingEdge(m_ack_o)

    r = await First(t, ma)
    if r is t:
        assert 1 == 0, "Timed out while waiting for m_ack_o to go high"

    await FallingEdge(clk_i)

    assert (m_dat_o.value.to_unsigned() == expected_value0), f"Tried to retrieve expected value written ({hex(expected_value0)}), but got {hex(m_dat_o.value.to_unsigned())}"

    await RisingEdge(clk_i)

    await FallingEdge(clk_i)
    m_we_i.value = 0
    m_stb_i.value = 1
    m_cyc_i.value = 1
    m_sel_i.value = 1
    m_adr_i.value = expected_at1
    m_dat_i.value = 0

    t = Timer(1000, unit="us")
    ma = RisingEdge(m_ack_o)

    r = await First(t, ma)
    if r is t:
        assert 1 == 0, "Timed out while waiting for m_ack_o to go high"

    await FallingEdge(clk_i)

    assert (m_dat_o.value.to_unsigned() == expected_value1), f"Tried to retrieve expected value written ({hex(expected_value1)}), but got {hex(m_dat_o.value.to_unsigned())}"

    await FallingEdge(clk_i)

    # END reading values at different rows

tests =[
    'test_reset',
    'test_initialization',
    'test_write',
    'test_read',
    'test_bank_switch_same_row',
    'test_bank_switch_diff_row',
]

proj_path = Path("./src").resolve()
sources = [
    proj_path / "dram" / "wb_sdr_mt48lc16m16a2_7e.sv",
    proj_path / "dram" / "sdram_model_mt48lc16m16a2.v",
    proj_path / "dram" / "tb_wb_sdr_mt48lc16m16a_7e.sv",
    proj_path / "common" / "shift.sv"
]
parameters = { "sys_clk_mhz_p": 100 }
module_name = "test_wb_sdr_mt48lc16m16a2_7e"
hdl_toplevel="tb_wb_sdr_mt48lc16m16a_7e"
sims = ['icarus']
# note: verilator doesn't like specify

@pytest.mark.parametrize("testcase", tests)
def test_sdr_ctrl_each(testcase):
    """Runs each test independently. Continues on test failure."""
    run_test(parameters=parameters, sources=sources, module_name=module_name, hdl_toplevel=hdl_toplevel, testcase=testcase, sims=sims)

def test_sdr_ctrl_all():
    """Runs all tests sequentially in one simulation."""
    run_test(parameters=parameters, sources=sources, module_name=module_name, hdl_toplevel=hdl_toplevel, sims=sims)
