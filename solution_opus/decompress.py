#!/usr/bin/env python3
"""
Decompress stdin to stdout using neural prediction + arithmetic coding.

Pipeline:
    1. Read [4-byte length header] + [compressed bitstream] from stdin
    2. For each position (up to original length):
       a. model.predict() → probability distribution (same as compressor!)
       b. Convert probs to cumulative frequencies
       c. Arithmetic-decode to get the byte
       d. model.observe(byte) → update model (same as compressor!)
    3. Write restored bytes to stdout

CRITICAL: The model predict/observe sequence must be IDENTICAL to the
compressor. Any divergence will produce garbage output.
"""

import sys
import io

from arithmetic_coding import ArithmeticDecoder, BitInputStream
from utils import probs_to_cumfreqs, read_header
from model import Model


def decompress(compressed: bytes) -> bytes:
    """Decompress bytes using neural prediction + arithmetic coding."""
    model = Model()

    buf = io.BytesIO(compressed)

    # Read header
    original_length = read_header(buf)

    # Set up arithmetic decoder
    bit_in = BitInputStream(buf)
    decoder = ArithmeticDecoder(bit_in)

    # Decode byte by byte
    result = bytearray()
    for i in range(original_length):
        # Get prediction from model (MUST match compressor)
        probs = model.predict()

        # Convert to cumulative frequencies (MUST match compressor)
        cumfreqs = probs_to_cumfreqs(probs)

        # Decode the byte
        byte_val = decoder.read(cumfreqs)
        result.append(byte_val)

        # Update model (MUST match compressor)
        model.observe(byte_val)

        if (i + 1) % 10000 == 0:
            print(f"\rDecompressing: {i+1}/{original_length} bytes", end='', file=sys.stderr)

    if original_length >= 10000:
        print(file=sys.stderr)

    return bytes(result)


def main():
    compressed = sys.stdin.buffer.read()
    restored = decompress(compressed)
    sys.stdout.buffer.write(restored)


if __name__ == '__main__':
    main()
