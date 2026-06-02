"""
PPM baseline: Adaptive Prediction by Partial Matching (order-0 through order-8).

Uses byte-level context statistics with confidence-weighted mixing and
Dirichlet smoothing. Higher-order contexts get more weight when they have
sufficient observations.

Score: ~2.3 on 200KB Wikipedia chunks (~3.5 bits per byte).

This is a strong classical baseline. To improve beyond it, the agent must
find approaches that capture patterns PPM cannot: long-range dependencies,
structural relationships, learned representations, or better model mixing.
"""

import numpy as np
from collections import defaultdict


class BytePredictor:
    def __init__(self, vocab_size=256, **kwargs):
        self.v = vocab_size
        self.max_order = 8

        # Smoothing factors per order
        self.smoothing = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]

        # Count tables: counts[order][context_tuple] -> array of 256 floats
        self.counts = [
            defaultdict(lambda s=s: np.ones(256, dtype=np.float64) * s)
            for s in self.smoothing
        ]

        # Base mixing weights per order
        self.base_weights = np.array([
            0.02, 0.04, 0.08, 0.14, 0.19, 0.20, 0.16, 0.10, 0.07
        ])

        self.history = []

    def predict(self):
        v = self.v
        prediction = np.zeros(v, dtype=np.float64)
        total_weight = 0.0

        for order in range(self.max_order + 1):
            if order == 0:
                ctx = ()
            elif len(self.history) < order:
                continue
            else:
                ctx = tuple(self.history[-order:])

            c = self.counts[order][ctx]
            total_count = c.sum()
            prob = c / total_count

            effective_count = total_count - self.smoothing[order] * v
            if order > 0:
                confidence = max(effective_count, 0) / (max(effective_count, 0) + 1.2) + 0.04
                weight = self.base_weights[order] * confidence
            else:
                weight = self.base_weights[0]

            prediction += weight * prob
            total_weight += weight

        if total_weight > 0:
            prediction /= total_weight
        else:
            prediction = np.ones(v, dtype=np.float64) / v

        prediction = np.maximum(prediction, 1e-7)
        prediction /= prediction.sum()
        return prediction

    def observe(self, byte_val):
        for order in range(self.max_order + 1):
            if order == 0:
                ctx = ()
            elif len(self.history) < order:
                continue
            else:
                ctx = tuple(self.history[-order:])
            self.counts[order][ctx][byte_val] += 1.0

        self.history.append(byte_val)
        if len(self.history) > 12:
            self.history = self.history[-12:]
        return 0.0


Model = BytePredictor
