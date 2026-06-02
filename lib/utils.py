"""
Utility functions for the neural text compression task.
"""

import numpy as np
import os
import struct


def probs_to_cumfreqs(probs, total=1 << 16):
    """
    Convert a probability distribution (float array summing to ~1.0) into
    a cumulative frequency table suitable for arithmetic coding.

    Args:
        probs: array-like of shape (num_symbols,), non-negative, sums to ~1.0
        total: target total frequency count (must be <= 2^30 for the coder)

    Returns:
        list of (num_symbols + 1) integers: cumfreqs[0]=0, cumfreqs[-1]=total
        Every symbol is guaranteed freq >= 1.
    """
    probs = np.asarray(probs, dtype=np.float64)
    num_symbols = len(probs)

    # Ensure no negative or NaN probabilities
    probs = np.maximum(probs, 0.0)
    prob_sum = probs.sum()
    if prob_sum <= 0:
        # Fallback to uniform
        probs = np.ones(num_symbols, dtype=np.float64)
        prob_sum = float(num_symbols)

    # Scale to target total, ensuring each symbol gets at least 1
    freqs = np.maximum(np.floor(probs / prob_sum * total).astype(np.int64), 1)

    # Adjust to hit exact total
    diff = total - freqs.sum()
    if diff > 0:
        # Add surplus to the symbols with highest probability
        indices = np.argsort(-probs)
        for i in range(int(diff)):
            freqs[indices[i % num_symbols]] += 1
    elif diff < 0:
        # Remove from symbols with highest frequency (keeping min 1)
        indices = np.argsort(-freqs)
        for i in range(int(-diff)):
            idx = indices[i % num_symbols]
            if freqs[idx] > 1:
                freqs[idx] -= 1

    # Build cumulative frequency table
    cumfreqs = [0] * (num_symbols + 1)
    for i in range(num_symbols):
        cumfreqs[i + 1] = cumfreqs[i] + int(freqs[i])

    # Final adjustment to ensure exact total
    cumfreqs[-1] = total

    return cumfreqs


def compute_score(original_size, compressed_size, decompressor_size):
    """
    Compute the Hutter-Prize-style compression score.

    score = original_size / (compressed_size + decompressor_size)

    Higher is better. Score of 1.0 means no compression at all.
    """
    total = compressed_size + decompressor_size
    if total <= 0:
        return 0.0
    return original_size / total


def get_code_size(solution_dir):
    """
    Compute total size of all .py files in the solution directory.
    This counts toward the decompressor_size in the score.
    """
    total = 0
    for fname in os.listdir(solution_dir):
        if fname.endswith('.py'):
            fpath = os.path.join(solution_dir, fname)
            total += os.path.getsize(fpath)
    return total


def write_header(out_stream, original_length):
    """Write a 4-byte big-endian length header."""
    out_stream.write(struct.pack('>I', original_length))


def read_header(in_stream):
    """Read a 4-byte big-endian length header."""
    data = in_stream.read(4)
    if len(data) < 4:
        raise ValueError("Truncated header")
    return struct.unpack('>I', data)[0]
