import re
import struct
import sys
import argparse


# ── Instruction encoders ──────────────────────────────────────────────────────

def encode_gmem2smem(match):
    opcode = 0b0101 if match.group('dir').lower() == 'weights' else 0b1101
    sram_addr = int(match.group('sram_addr'), 0)
    dram_addr = int(match.group('dram_addr'), 0)
    tx_amount = int(match.group('tx_amount'), 0)
    word = (opcode & 0xF)
    word |= (sram_addr>>2 & 0xFF) << 4
    word |= (dram_addr>>2 & 0xFFF) << 12
    word |= (tx_amount>>2 & 0xFF) << 24
    return word, 32

def encode_smem2gmem(match):
    opcode = 0b1111
    sram_addr = int(match.group('sram_addr'), 0)
    dram_addr = int(match.group('dram_addr'), 0)
    tx_amount = int(match.group('tx_amount'), 0)
    word = (opcode & 0xF)
    word |= (sram_addr>>2 & 0xFF) << 4
    word |= (dram_addr>>2 & 0xFFF) << 12
    word |= (tx_amount>>2 & 0xFF) << 24
    return word, 32

def encode_load_bias(match):
    opcode = 0b1000
    sram_addr = int(match.group('sram_addr'), 0)
    word = (opcode & 0xF)
    word |= (sram_addr>>2 & 0xFF) << 4
    return word, 16

def encode_load_zp(match):
    opcode = 0b1100
    sram_addr = int(match.group('sram_addr'), 0)
    word = (opcode & 0xF)
    word |= (sram_addr>>2 & 0xFF) << 4
    return word, 16

def encode_load_scale(match):
    opcode = 0b1010
    sram_addr = int(match.group('sram_addr'), 0)
    word = (opcode & 0xF)
    word |= (sram_addr>>2 & 0xFF) << 4
    return word, 16

def encode_pipeline_setup(match):
    opcode = 0b1110
    relu_mode_str = match.group('relu_mode').strip().lower()
    relu_map = {'enable': 0b1000, 'disable': 0b0100, 'clamped': 0b0010, 'leaky': 0b0001}
    relu_vec = relu_map.get(relu_mode_str, 0b1000)
    k_tile = int(match.group('k_tile'), 0)
    word = (opcode & 0xF)
    word |= (relu_vec & 0xF) << 4
    word |= (k_tile>>2 & 0xFF) << 8
    return word, 16

def encode_load_weights(match):
    opcode = 0b0110
    sram_addr = int(match.group('sram_addr'), 0)
    word = (opcode & 0xF)
    word |= (sram_addr>>2 & 0xFF) << 4
    return word, 16

def encode_matmul(match):
    opcode = 0b0001
    act_addr = int(match.group('act_addr'), 0)
    res_addr = int(match.group('res_addr'), 0)
    word = (opcode & 0xF)
    word |= (act_addr>>2 & 0xFF) << 4
    word |= (res_addr>>2 & 0xFF) << 12
    return word, 24

def encode_exit(match):
    return 0b0000, 16


# ── Instruction patterns (case-insensitive) ───────────────────────────────────

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


# ── Core assembler ────────────────────────────────────────────────────────────

def assemble(source: str) -> list[dict]:
    """
    Assemble source text into a list of encoded instructions.
    Returns list of dicts with keys: line, mnemonic, bits, value.
    """
    results = []
    for lineno, raw_line in enumerate(source.splitlines(), 1):
        line = raw_line.split('#')[0].strip()   # strip comments
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


# ── Output helpers ────────────────────────────────────────────────────────────

def print_assembled(results, file=sys.stdout):
    """Print a human-readable listing of assembled instructions."""
    for r in results:
        nibbles = r['bits'] // 4
        hex_fmt = f"0x{{:0{nibbles}X}}"
        bin_str = f"{r['value']:0{r['bits']}b}"
        # Group binary digits into nibbles (4 bits) separated by underscores
        grouped = '_'.join(bin_str[i:i+4] for i in range(0, len(bin_str), 4))
        print(
            f"[line {r['line']:3d}]"
            f"  {hex_fmt.format(r['value'])}"
            f"  {grouped}"
            f"  ({r['bits']}b)"
            f"  {r['mnemonic']}",
            file=file
        )


def write_binary(results, file):
    """
    Write assembled instructions as packed binary.

    Each instruction is written in little-endian byte order using the exact
    number of bytes its bit-width requires (bits // 8).  The bit-widths
    produced by the encoders are all multiples of 8, so no padding is needed.
    """
    for r in results:
        n_bytes = r['bits'] // 8
        file.write(r['value'].to_bytes(n_bytes, byteorder='little'))


# ── Built-in test program ─────────────────────────────────────────────────────

BUILTIN_SOURCE = """\
dram2sram weights,      sram=0x10, dram=0x100, n=8
dram2sram activations,  sram=0x20, dram=0x300, n=20
pipeline_setup relu=enable, k_tile=8
load_weights sram=0x10

load_bias   sram=0x20
load_zp     sram=0x24 # comment
load_scale  sram=0x28

matmul act=0x2C, res=0x50

sram2dram sram=0x50, dram=0x400, n=8

exit
"""


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Assemble a custom ISA source file into binary.')
    parser.add_argument(
        'input', nargs='?', default=None,
        help='Assembly source file (omit to use the built-in test program)')
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output file path. '
             'Use -o <file>.bin for raw binary, '
             'or omit for a human-readable hex listing on stdout.')
    parser.add_argument(
        '-b', '--binary', action='store_true',
        help='Write raw binary even when sending to stdout '
             '(useful for piping: assembler.py --binary | xxd)')
    args = parser.parse_args()

    # ── Read source ───────────────────────────────────────────────────────────
    if args.input is None:
        source = BUILTIN_SOURCE
        print('(no input file given — using built-in test program)\n',
              file=sys.stderr)
    else:
        with open(args.input, 'r') as f:
            source = f.read()

    # ── Assemble ──────────────────────────────────────────────────────────────
    try:
        results = assemble(source)
    except SyntaxError as e:
        print(f'Assembler error: {e}', file=sys.stderr)
        sys.exit(1)

    # ── Emit output ───────────────────────────────────────────────────────────
    if args.output is not None:
        # Decide format from extension; --binary flag overrides to binary.
        binary_mode = args.binary or args.output.endswith('.bin')
        if binary_mode:
            with open(args.output, 'wb') as f:
                write_binary(results, f)
            print(f'Wrote {len(results)} instructions → {args.output}',
                  file=sys.stderr)
        else:
            with open(args.output, 'w') as f:
                print_assembled(results, file=f)
            print(f'Wrote {len(results)} instructions → {args.output}',
                  file=sys.stderr)
    else:
        if args.binary:
            write_binary(results, sys.stdout.buffer)
        else:
            print_assembled(results)


if __name__ == '__main__':
    main()