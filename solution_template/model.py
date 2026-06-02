"""
Baseline: Adaptive PPM (Prediction by Partial Matching) compressor.

Uses order-0 through order-8 byte-level context models with confidence-weighted
mixing. Each context order maintains frequency counts with Dirichlet smoothing.
Higher-order contexts get more weight when they have sufficient statistics.

This scores ~2.3 on 200KB Wikipedia chunks. To improve further, consider:
what patterns does this model miss? Where does it waste bits?
"""

import numpy as np
from collections import defaultdict


class BytePredictor:
    def __init__(self, vocab_size=256, **kwargs):
        self.v = vocab_size
        self.max_order = 12

        # Smoothing factors per order (lower = sharper after more observations)
        self.smoothing = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

        # Count tables: counts[order][context_tuple] -> array of 256 floats
        self.counts = [
            defaultdict(lambda s=s: np.ones(256, dtype=np.float64) * s)
            for s in self.smoothing
        ]

        # Base mixing weights per order (log-space mixing)
        # Tuned: shift weight toward medium-high orders (5-9) for Wikipedia XML
        self.base_weights = np.array([
            0.01, 0.02, 0.04, 0.08, 0.12, 0.16, 0.18, 0.16, 0.12, 0.07, 0.03, 0.01, 0.01
        ])

        # Recent byte history (kept to max_order + some margin)
        self.history = []

    def predict(self):
        v = self.v
        h = self.history
        hn = len(h)

        # Mix in log-probability space (geometric mixing)
        log_pred = np.zeros(v, dtype=np.float64)
        total_weight = 0.0

        for order in range(self.max_order + 1):
            if order == 0:
                ctx = ()
            elif hn < order:
                continue
            else:
                ctx = tuple(h[-order:])

            c = self.counts[order][ctx]
            total_count = c.sum()

            effective_count = total_count - self.smoothing[order] * v
            eff = max(effective_count, 0.0)
            if order > 0:
                # Tighter confidence threshold for high orders
                thresh = 2.0 if order >= 6 else 3.0
                confidence = eff / (eff + thresh)
                weight = self.base_weights[order] * confidence
            else:
                weight = self.base_weights[0]

            if weight < 1e-10:
                continue

            prob = np.maximum(c / total_count, 1e-8)
            log_pred += weight * np.log(prob)
            total_weight += weight

        if total_weight > 0:
            log_pred /= total_weight
        else:
            log_pred = np.zeros(v, dtype=np.float64)

        log_pred -= log_pred.max()
        prediction = np.exp(log_pred)
        prediction /= prediction.sum()
        return prediction

    def observe(self, byte_val):
        # Update counts at every order
        for order in range(self.max_order + 1):
            if order == 0:
                ctx = ()
            elif len(self.history) < order:
                continue
            else:
                ctx = tuple(self.history[-order:])
            self.counts[order][ctx][byte_val] += 1.0

        h = self.history
        h.append(byte_val)
        if len(h) > self.max_order + 2:
            del h[0]

        return 0.0


Model = BytePredictor
