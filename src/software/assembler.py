import re
import struct

# Instruction encoders
def encode_gmem2smem(match):
    opcode = 0b0101 if match.group('dir') == 'weights' else 0b1111
    sram_addr = int(match.group('sram_addr'), 0)
    dram_addr = int(match.group('dram_addr'), 0)
    tx_amount = int(match.group('tx_amount'), 0)
    word = (opcode & 0xF)
    word |= (sram_addr & 0xFF) << 4
    word |= (sram_addr & 0xFF) << 12  # bits [11:8] mirror
    word |= (dram_addr & 0xFFF) << 12
    word |= (dram_addr & 0xFFF) << 20  # bits [23:12] mirror - actually spread
    word |= (tx_amount & 0xFF) << 24
    # Proper encoding:
    word = (opcode & 0xF)
    word |= ((sram_addr >> 4) & 0xFF) << 4   # sram [11:4] in bits [7:4]
    word |= ((sram_addr >> 4) & 0xFF) << 8   # sram [11:4] in bits [11:8]
    word |= ((dram_addr >> 12) & 0xFF) << 12 # dram [23:12] in bits [15:12]
    word |= ((dram_addr >> 12) & 0xFF) << 16 # dram [23:12] in bits [19:16]
    word |= ((dram_addr >> 12) & 0xFF) << 20 # dram [23:12] in bits [23:20]
    word |= (tx_amount & 0xFF) << 24
    return word, 32

def encode_smem2gmem(match):
    opcode = 0b0100
    sram_addr = int(match.group('sram_addr'), 0)
    dram_addr = int(match.group('dram_addr'), 0)
    tx_amount = int(match.group('tx_amount'), 0)
    word = (opcode & 0xF)
    word |= ((sram_addr >> 4) & 0xFF) << 4
    word |= ((sram_addr >> 4) & 0xFF) << 8
    word |= ((dram_addr >> 12) & 0xFF) << 12
    word |= ((dram_addr >> 12) & 0xFF) << 16
    word |= ((dram_addr >> 12) & 0xFF) << 20
    word |= (tx_amount & 0xFF) << 24
    return word, 32

def encode_load_bias(match):
    opcode = 0b1001
    sram_addr = int(match.group('sram_addr'), 0)
    word = (opcode & 0xF)
    word |= ((sram_addr >> 4) & 0xFF) << 4
    word |= ((sram_addr >> 4) & 0xFF) << 8
    return word, 16

def encode_load_zp(match):
    opcode = 0b1010
    sram_addr = int(match.group('sram_addr'), 0)
    word = (opcode & 0xF)
    word |= ((sram_addr >> 4) & 0xFF) << 4
    word |= ((sram_addr >> 4) & 0xFF) << 8
    return word, 16

def encode_load_scale(match):
    opcode = 0b1100
    sram_addr = int(match.group('sram_addr'), 0)
    word = (opcode & 0xF)
    word |= ((sram_addr >> 4) & 0xFF) << 4
    word |= ((sram_addr >> 4) & 0xFF) << 8
    return word, 16

def encode_pipeline_setup(match):
    opcode = 0b1000
    relu_mode_str = match.group('relu_mode').strip().lower()
    relu_map = {'enable': 0b1000, 'disable': 0b0100, 'clamped': 0b0010, 'leaky': 0b0001}
    relu_vec = relu_map.get(relu_mode_str, 0b1000)
    k_tile = int(match.group('k_tile'), 0)
    word = (opcode & 0xF)
    word |= (relu_vec & 0xF) << 4
    word |= (k_tile & 0xFF) << 8
    word |= (k_tile & 0xFF) << 12
    return word, 16

def encode_load_weights(match):
    opcode = 0b1110
    sram_addr = int(match.group('sram_addr'), 0)
    word = (opcode & 0xF)
    word |= ((sram_addr >> 4) & 0xFF) << 4
    word |= ((sram_addr >> 4) & 0xFF) << 8
    return word, 16

def encode_matmul(match):
    opcode = 0b0001
    act_addr  = int(match.group('act_addr'), 0)
    res_addr  = int(match.group('res_addr'), 0)
    word = (opcode & 0xF)
    word |= ((act_addr >> 4) & 0xFF) << 4
    word |= ((act_addr >> 4) & 0xFF) << 8
    word |= ((res_addr >> 12) & 0xFF) << 12
    word |= ((res_addr >> 12) & 0xFF) << 16
    return word, 24

def encode_exit(match):
    return 0b0000, 16

# Instruction patterns (case-insensitive)
INSTRUCTIONS = [
    (re.compile(
        r'^\s*dram2sram\s+(?P<dir>weights|activations)\s*,\s*sram=(?P<sram_addr>\w+)\s*,\s*dram=(?P<dram_addr>\w+)\s*,\s*n=(?P<tx_amount>\w+)\s*$',
        re.IGNORECASE), encode_gmem2smem),
    (re.compile(
        r'^\s*sram2dram\s+sram=(?P<sram_addr>\w+)\s*,\s*dram=(?P<dram_addr>\w+)\s*,\s*n=(?P<tx_amount>\w+)\s*$',
        re.IGNORECASE), encode_smem2gmem),
    (re.compile(
        r'^\s*load_bias\s+sram=(?P<sram_addr>\w+)\s*$',
        re.IGNORECASE), encode_load_bias),
    (re.compile(
        r'^\s*load_zp\s+sram=(?P<sram_addr>\w+)\s*$',
        re.IGNORECASE), encode_load_zp),
    (re.compile(
        r'^\s*load_scale\s+sram=(?P<sram_addr>\w+)\s*$',
        re.IGNORECASE), encode_load_scale),
    (re.compile(
        r'^\s*pipeline_setup\s+relu=(?P<relu_mode>\w+)\s*,\s*k_tile=(?P<k_tile>\w+)\s*$',
        re.IGNORECASE), encode_pipeline_setup),
    (re.compile(
        r'^\s*load_weights\s+sram=(?P<sram_addr>\w+)\s*$',
        re.IGNORECASE), encode_load_weights),
    (re.compile(
        r'^\s*matmul\s+act=(?P<act_addr>\w+)\s*,\s*res=(?P<res_addr>\w+)\s*$',
        re.IGNORECASE), encode_matmul),
    (re.compile(
        r'^\s*exit\s*$',
        re.IGNORECASE), encode_exit),
]

def assemble(source: str) -> list[dict]:
    """
    Assemble source text into a list of encoded instructions.
    Returns list of dicts with keys: line, mnemonic, bits, value
    """
    results = []
    for lineno, raw_line in enumerate(source.splitlines(), 1):
        line = raw_line.split('#')[0].strip()  # strip comments
        if not line:
            continue
        matched = False
        for pattern, encoder in INSTRUCTIONS:
            m = pattern.match(line)
            if m:
                value, bits = encoder(m)
                results.append({
                    'line':     lineno,
                    'mnemonic': line,
                    'bits':     bits,
                    'value':    value,
                })
                matched = True
                break
        if not matched:
            raise SyntaxError(f"Line {lineno}: unrecognized instruction: '{line}'")
    return results

def print_assembled(results):
    for r in results:
        nibbles = r['bits'] // 4
        fmt = f"0x{{:0{nibbles}X}}"
        print(f"[line {r['line']:3d}] {fmt.format(r['value'])}  ({r['bits']}b)  {r['mnemonic']}")

# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    source = """
    dram2sram weights,      sram=0x100, dram=0x1000, n=64
    dram2sram activations,  sram=0x200, dram=0x2000, n=32

    load_bias   sram=0x300
    load_zp     sram=0x310
    load_scale  sram=0x320

    pipeline_setup relu=enable, k_tile=8
    load_weights sram=0x400
    matmul act=0x200, res=0x500

    sram2dram sram=0x500, dram=0x3000, n=16

    exit
    """

    results = assemble(source)
    print_assembled(results)