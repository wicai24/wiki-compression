"""
No-op baseline: Uniform distribution over all bytes.
No learning, no state. The arithmetic coder cannot compress at all.

Score: ~0.98 on 200KB chunks (below 1.0 due to code size overhead).

Use this as the starting point for maximum exploration freedom — the agent
builds everything from scratch.
"""

import numpy as np


class BytePredictor:
    def __init__(self, vocab_size=256, **kwargs):
        self.vocab_size = vocab_size

    def predict(self):
        return np.ones(self.vocab_size) / self.vocab_size

    def observe(self, byte_val):
        return np.log(self.vocab_size)


Model = BytePredictor
