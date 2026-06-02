#!/usr/bin/env python3
"""
Compress stdin to stdout using neural prediction + arithmetic coding.

Pipeline:
    1. Read raw bytes from stdin
    2. For each byte:
       a. model.predict() → probability distribution
       b. Convert probs to cumulative frequencies
       c. Arithmetic-encode the actual byte
       d. model.observe(byte) → update model
    3. Write [4-byte length header] + [compressed bitstream] to stdout

The decompressor mirrors this exactly (same model, same predict/observe order).
"""

import sys
import io
import struct

from arithmetic_coding import ArithmeticEncoder, BitOutputStream
from utils import probs_to_cumfreqs, write_header
from model import Model


def compress(data: bytes) -> bytes:
    """Compress raw bytes using neural prediction + arithmetic coding."""
    model = Model()

    # Write to in-memory buffer
    buf = io.BytesIO()

    # Reserve space for header (original length)
    write_header(buf, len(data))

    # Set up arithmetic encoder
    bit_out = BitOutputStream(buf)
    encoder = ArithmeticEncoder(bit_out)

    for i, byte_val in enumerate(data):
        # Get prediction from model
        probs = model.predict()

        # Convert to cumulative frequencies for arithmetic coder
        cumfreqs = probs_to_cumfreqs(probs)

        # Encode the actual byte
        encoder.write(cumfreqs, byte_val)

        # Update model with the actual byte (online learning)
        model.observe(byte_val)

        # Progress reporting (to stderr, not stdout)
        if (i + 1) % 10000 == 0:
            print(f"\rCompressing: {i+1}/{len(data)} bytes", end='', file=sys.stderr)

    encoder.finish()

    if len(data) >= 10000:
        print(file=sys.stderr)  # newline after progress

    return buf.getvalue()


def main():
    data = sys.stdin.buffer.read()
    compressed = compress(data)
    sys.stdout.buffer.write(compressed)


if __name__ == '__main__':
    main()
