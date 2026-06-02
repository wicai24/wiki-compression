"""Tests for the arithmetic coding library."""

import io
import sys
import os
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from lib.arithmetic_coding import (
    ArithmeticEncoder, ArithmeticDecoder,
    BitOutputStream, BitInputStream,
)
from lib.utils import probs_to_cumfreqs

import numpy as np


def _roundtrip(symbols, cumfreqs_per_step):
    """Encode symbols, decode them, verify they match."""
    # Encode
    buf = io.BytesIO()
    bit_out = BitOutputStream(buf)
    encoder = ArithmeticEncoder(bit_out)
    for sym, cf in zip(symbols, cumfreqs_per_step):
        encoder.write(cf, sym)
    encoder.finish()

    compressed = buf.getvalue()

    # Decode
    bit_in = BitInputStream(io.BytesIO(compressed))
    decoder = ArithmeticDecoder(bit_in)
    decoded = []
    for cf in cumfreqs_per_step:
        decoded.append(decoder.read(cf))

    assert decoded == symbols, f"Mismatch: {decoded[:10]}... vs {symbols[:10]}..."
    return compressed


def test_uniform_distribution():
    """Roundtrip with uniform distribution over 256 symbols."""
    n = 1000
    symbols = [random.randint(0, 255) for _ in range(n)]
    # Uniform: each symbol has freq=1, total=256
    cumfreqs = list(range(257))  # [0, 1, 2, ..., 256]
    cumfreqs_list = [cumfreqs] * n
    compressed = _roundtrip(symbols, cumfreqs_list)
    # With uniform dist over 256 symbols, should be ~8 bits/symbol = ~1000 bytes
    assert len(compressed) <= n + 100  # some overhead is ok


def test_skewed_distribution():
    """Roundtrip with heavily skewed distribution."""
    n = 2000
    # Symbol 0 has 90% probability, rest share 10%
    probs = np.zeros(256)
    probs[0] = 0.9
    probs[1:] = 0.1 / 255
    cumfreqs = probs_to_cumfreqs(probs, total=1 << 16)

    # Generate symbols biased toward 0
    symbols = [0 if random.random() < 0.9 else random.randint(1, 255) for _ in range(n)]
    cumfreqs_list = [cumfreqs] * n
    compressed = _roundtrip(symbols, cumfreqs_list)
    # Should compress much better than uniform
    assert len(compressed) < n // 2


def test_adaptive_frequencies():
    """Roundtrip where frequency table changes each step."""
    n = 500
    symbols = [random.randint(0, 9) for _ in range(n)]
    cumfreqs_list = []
    for i in range(n):
        # Build a different freq table each step
        probs = np.random.dirichlet(np.ones(10))
        cumfreqs = probs_to_cumfreqs(probs, total=1 << 14)
        cumfreqs_list.append(cumfreqs)
    _roundtrip(symbols, cumfreqs_list)


def test_single_symbol():
    """Edge case: encode and decode a single symbol."""
    cumfreqs = list(range(257))
    compressed = _roundtrip([42], [cumfreqs])
    assert len(compressed) > 0


def test_all_same_byte():
    """All symbols are the same — should compress well with right distribution."""
    n = 1000
    symbols = [7] * n
    probs = np.zeros(256)
    probs[7] = 0.99
    probs[:7] = 0.01 / 255
    probs[8:] = 0.01 / 255
    cumfreqs = probs_to_cumfreqs(probs, total=1 << 16)
    compressed = _roundtrip(symbols, [cumfreqs] * n)
    # 1000 symbols of something with 99% probability should be very small
    assert len(compressed) < 50


def test_determinism():
    """Same input + same freqs = same compressed output."""
    symbols = [1, 2, 3, 4, 5] * 100
    cumfreqs = list(range(257))
    c1 = _roundtrip(symbols, [cumfreqs] * len(symbols))
    c2 = _roundtrip(symbols, [cumfreqs] * len(symbols))
    assert c1 == c2


def test_probs_to_cumfreqs_basic():
    """Test probability to cumulative frequency conversion."""
    probs = np.array([0.5, 0.3, 0.2])
    cumfreqs = probs_to_cumfreqs(probs, total=1000)
    assert cumfreqs[0] == 0
    assert cumfreqs[-1] == 1000
    # Each symbol must have freq >= 1
    for i in range(len(probs)):
        assert cumfreqs[i + 1] > cumfreqs[i]


def test_probs_to_cumfreqs_uniform():
    """Uniform probabilities should give roughly equal frequencies."""
    probs = np.ones(256) / 256
    cumfreqs = probs_to_cumfreqs(probs, total=1 << 16)
    assert cumfreqs[0] == 0
    assert cumfreqs[-1] == 1 << 16
    # Each freq should be roughly 256
    for i in range(256):
        freq = cumfreqs[i + 1] - cumfreqs[i]
        assert freq >= 1
        assert 200 <= freq <= 300


def test_probs_to_cumfreqs_degenerate():
    """All-zero probabilities should fallback to uniform."""
    probs = np.zeros(10)
    cumfreqs = probs_to_cumfreqs(probs, total=1000)
    assert cumfreqs[0] == 0
    assert cumfreqs[-1] == 1000
    for i in range(10):
        assert cumfreqs[i + 1] > cumfreqs[i]


if __name__ == '__main__':
    test_uniform_distribution()
    test_skewed_distribution()
    test_adaptive_frequencies()
    test_single_symbol()
    test_all_same_byte()
    test_determinism()
    test_probs_to_cumfreqs_basic()
    test_probs_to_cumfreqs_uniform()
    test_probs_to_cumfreqs_degenerate()
    print("All arithmetic coding tests passed!")
