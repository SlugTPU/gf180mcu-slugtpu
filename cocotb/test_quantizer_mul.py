import pytest
import cocotb
from cocotb.triggers import Timer
from pathlib import Path
from runner import run_test


def _to_signed32(val):
    """Convert 32-bit value to signed integer."""
    mask = (1 << 32) - 1
    v = val & mask
    if v & (1 << 31):
        return v - (1 << 32)
    return v


def quantizer_mul_model(psum, m0, fixed_shift=16, acc_width=32, m0_width=32):
    """
    Model of the quantizer_mul hardware.
    
    Args:
        psum: signed 32-bit partial sum
        m0: signed 32-bit multiplier
        fixed_shift: number of bits to shift right (default 16)
        acc_width: accumulator width (default 32)
        m0_width: multiplier width (default 32)
    
    Returns:
        Saturated 8-bit signed output
    """
    # Ensure inputs are converted to signed
    psum = _to_signed32(psum) if isinstance(psum, int) else int(psum)
    m0 = _to_signed32(m0) if isinstance(m0, int) else int(m0)
    
    # 1. Multiply (produces 64-bit result)
    product = psum * m0
    
    # 2. Add rounding constant (0.5 in fixed point)
    rounded = product + (1 << (fixed_shift - 1))
    
    # 3. Arithmetic right shift
    shifted = rounded >> fixed_shift
    
    # 4. Saturate to 8-bit signed range [-128, 127]
    if shifted > 127:
        result = 127
    elif shifted < -128:
        result = -128
    else:
        result = shifted & 0xFF
    
    return result & 0xFF  # Return as 8-bit unsigned representation


class QuantizerMulTest:
    def __init__(self, dut):
        self.dut = dut
        self.test_count = 0
        self.pass_count = 0

    async def test_case(self, psum, m0, expected):
        """Test a single case."""
        self.dut.psum.value = psum & ((1 << 32) - 1)
        self.dut.m0.value = m0 & ((1 << 32) - 1)
        
        # Wait a small time for combinational logic to settle
        await Timer(1, 'ns')
        
        got = int(self.dut.q_out.value)
        # Convert to signed for comparison
        if got & 0x80:
            got_signed = got - 256
        else:
            got_signed = got
        
        # Convert expected to signed for error messages
        if expected & 0x80:
            expected_signed = expected - 256
        else:
            expected_signed = expected
        
        self.test_count += 1
        
        if got_signed == expected_signed:
            self.pass_count += 1
            cocotb.log.info(f"✓ PASS: psum={_to_signed32(psum):6d}, m0={_to_signed32(m0):6d} -> {expected_signed:4d}")
        else:
            cocotb.log.error(f"✗ FAIL: psum={_to_signed32(psum):6d}, m0={_to_signed32(m0):6d} -> got {got_signed}, expected {expected_signed}")
            raise AssertionError(f"Test failed: psum={_to_signed32(psum)}, m0={_to_signed32(m0)}, got {got_signed}, expected {expected_signed}")


def _m0_one(dut):
    return 1 << int(dut.FIXED_SHIFT.value)


@cocotb.test()
async def test_quantizer_mul_basic(dut):
    """Test basic multiplication and quantization."""
    test = QuantizerMulTest(dut)
    fixed_shift = int(dut.FIXED_SHIFT.value)
    m0_one = _m0_one(dut)
    
    # Test case 1: Zero inputs
    cocotb.log.info("=== Testing zero inputs ===")
    await test.test_case(0, 0, quantizer_mul_model(0, 0))
    
    # Test case 2: One input zero
    await test.test_case(100, 0, quantizer_mul_model(100, 0))
    await test.test_case(0, 100, quantizer_mul_model(0, 100))
    
    # Test case 3: Small positive numbers (use 1.0 fixed-point multiplier)
    cocotb.log.info("=== Testing small positive numbers ===")
    await test.test_case(1, m0_one, quantizer_mul_model(1, m0_one, fixed_shift))
    await test.test_case(10, m0_one, quantizer_mul_model(10, m0_one, fixed_shift))
    await test.test_case(100, m0_one, quantizer_mul_model(100, m0_one, fixed_shift))
    
    # Test case 4: Small negative numbers (use 1.0 fixed-point multiplier)
    cocotb.log.info("=== Testing small negative numbers ===")
    psum_neg = ((-1) & ((1 << 32) - 1))
    await test.test_case(psum_neg, m0_one, quantizer_mul_model(psum_neg, m0_one, fixed_shift))
    
    psum_neg = ((-10) & ((1 << 32) - 1))
    await test.test_case(psum_neg, m0_one, quantizer_mul_model(psum_neg, m0_one, fixed_shift))
    
    # Test case 5: Mixed signs (use 1.0 fixed-point multiplier)
    cocotb.log.info("=== Testing mixed signs ===")
    psum_pos = 100
    m0_neg = ((-m0_one) & ((1 << 32) - 1))
    await test.test_case(psum_pos, m0_neg, quantizer_mul_model(psum_pos, m0_neg, fixed_shift))
    
    psum_neg = ((-100) & ((1 << 32) - 1))
    m0_pos = m0_one
    await test.test_case(psum_neg, m0_pos, quantizer_mul_model(psum_neg, m0_pos, fixed_shift))


@cocotb.test()
async def test_quantizer_mul_positive_saturation(dut):
    """Test positive saturation at 127."""
    test = QuantizerMulTest(dut)
    fixed_shift = int(dut.FIXED_SHIFT.value)
    
    cocotb.log.info("=== Testing positive saturation ===")
    
    # Large numbers that should saturate to 127
    # To get output > 127, we need: (psum * m0) >> 16 > 127
    # So psum * m0 > 127 * 65536 = 8323072
    
    # 10000 * 1000 = 10,000,000 >> 16 = 152 (should saturate to 127)
    await test.test_case(10000, 1000, quantizer_mul_model(10000, 1000))
    
    # 32767 * 32767 = very large, should saturate
    await test.test_case(32767, 32767, quantizer_mul_model(32767, 32767))
    
    # 256 * 256 = 65536 >> 16 = 1 (no saturation)
    await test.test_case(256, 256, quantizer_mul_model(256, 256))
    
    # 1000 * 1000 = 1000000 >> 16 = 15 (no saturation)
    await test.test_case(1000, 1000, quantizer_mul_model(1000, 1000))
    
    # 5000 * 1000 = 5000000 >> 16 = 76 (no saturation)
    await test.test_case(5000, 1000, quantizer_mul_model(5000, 1000))
    
    # Test values that cause saturation
    # 9000 * 1000 = 9000000, >> 16 = 137 (saturates to 127)
    await test.test_case(9000, 1000, quantizer_mul_model(9000, 1000))


@cocotb.test()
async def test_quantizer_mul_negative_saturation(dut):
    """Test negative saturation at -128."""
    test = QuantizerMulTest(dut)
    
    cocotb.log.info("=== Testing negative saturation ===")
    
    # To get output < -128, we need: (psum * m0) >> 16 < -128
    # So psum * m0 < -128 * 65536 = -8388608
    
    # Negative versions
    psum_neg = ((-10000) & ((1 << 32) - 1))
    m0_pos = 1000
    await test.test_case(psum_neg, m0_pos, quantizer_mul_model(psum_neg, m0_pos))
    
    psum_neg = ((-32768) & ((1 << 32) - 1))
    m0_neg = ((-32768) & ((1 << 32) - 1))
    await test.test_case(psum_neg, m0_neg, quantizer_mul_model(psum_neg, m0_neg))
    
    # -9000 * 1000 = -9000000 >> 16 = -137 (saturates to -128)
    psum_neg = ((-9000) & ((1 << 32) - 1))
    m0_pos = 1000
    await test.test_case(psum_neg, m0_pos, quantizer_mul_model(psum_neg, m0_pos))


@cocotb.test()
async def test_quantizer_mul_rounding(dut):
    """Test rounding behavior."""
    test = QuantizerMulTest(dut)
    fixed_shift = int(dut.FIXED_SHIFT.value)
    
    cocotb.log.info("=== Testing rounding ===")
    
    # Test rounding: the +0.5 before shift should round towards nearest integer
    # product = psum * m0
    # rounded = product + (1 << 15)  // add 32768 for rounding
    # shifted = rounded >> 16
    
    # For result with fractional part:
    # If we want (psum * m0) = X.Y in fixed point where shift is 16
    # Then psum * m0 (in integer) = X * 65536 + Y
    # After rounding: X * 65536 + Y + 32768
    # After shift: X + (Y + 32768) >> 16
    # If Y >= 32768, we round up; otherwise round down
    
    # Example: psum=3, m0=10922 -> 32766
    # 32766 >> 16 = 0.5 in fixed point
    # After rounding: 32766 + 32768 = 65534 >> 16 = 1
    await test.test_case(3, 10922, quantizer_mul_model(3, 10922, fixed_shift))
    
    # Another test: psum=1, m0=32768
    # 32768 >> 16 = 0.5 in fixed point
    # After rounding: 32768 + 32768 = 65536 >> 16 = 1
    await test.test_case(1, 32768, quantizer_mul_model(1, 32768, fixed_shift))
    
    # psum=1, m0=32767
    # 32767 >> 16 = 0.4999... in fixed point
    # After rounding: 32767 + 32768 = 65535 >> 16 = 0
    await test.test_case(1, 32767, quantizer_mul_model(1, 32767, fixed_shift))


@cocotb.test()
async def test_quantizer_mul_boundary(dut):
    """Test boundary values."""
    test = QuantizerMulTest(dut)
    m0_one = _m0_one(dut)
    fixed_shift = int(dut.FIXED_SHIFT.value)
    
    cocotb.log.info("=== Testing boundary values ===")
    
    # Test with maximum positive value for signed 32-bit
    max_32 = 0x7FFFFFFF
    
    # Test with -1 (all 1s when unsigned)
    min_32 = 0xFFFFFFFF
    
    # Small values (use fixed-point multipliers so shift is non-zero)
    await test.test_case(1, m0_one, quantizer_mul_model(1, m0_one, fixed_shift))  # 1.0 in fixed point
    await test.test_case(2, m0_one, quantizer_mul_model(2, m0_one, fixed_shift))
    
    # 127 * 2^fixed_shift >> fixed_shift = 127 (max output)
    await test.test_case(127, m0_one, quantizer_mul_model(127, m0_one, fixed_shift))
    
    # (-128) * 2^fixed_shift >> fixed_shift = -128 (min output)
    m0_signed = ((-128) & ((1 << 32) - 1))
    await test.test_case(m0_one & ((1 << 32) - 1), m0_signed, quantizer_mul_model(m0_one & ((1 << 32) - 1), m0_signed, fixed_shift))

    # Use boundary 32-bit inputs to exercise saturation
    await test.test_case(max_32, m0_one, quantizer_mul_model(max_32, m0_one, fixed_shift))
    await test.test_case(min_32, m0_one, quantizer_mul_model(min_32, m0_one, fixed_shift))


tests = [
    "test_quantizer_mul_basic",
    "test_quantizer_mul_positive_saturation",
    "test_quantizer_mul_negative_saturation",
    "test_quantizer_mul_rounding",
    "test_quantizer_mul_boundary",
]


@pytest.mark.parametrize("testcase", tests)
def test_quantizer_mul_each(testcase):
    """Runs each test independently. Continues on test failure."""
    proj_path = Path("./rtl").resolve()
    sources = [proj_path / "quantizer_mul.sv"]

    run_test(
        parameters={"ACC_WIDTH": 32, "FIXED_SHIFT": 16, "M0_WIDTH": 32},
        sources=sources,
        module_name="test_quantizer_mul",
        hdl_toplevel="quantizer_mul",
        testcase=testcase,
    )


def test_quantizer_mul_all():
    """Runs each test sequentially as one giant test."""
    proj_path = Path("./rtl").resolve()
    sources = [proj_path / "quantizer_mul.sv"]

    run_test(
        parameters={"ACC_WIDTH": 32, "FIXED_SHIFT": 16, "M0_WIDTH": 32},
        sources=sources,
        module_name="test_quantizer_mul",
        hdl_toplevel="quantizer_mul",
    )
