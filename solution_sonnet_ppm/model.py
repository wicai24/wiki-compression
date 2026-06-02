"""
Advanced Context Mixing Compressor for Wikipedia data.

Architecture:
1. PPM-D (Prediction by Partial Matching with Deterministic exclusion) orders 1-12
2. Match model: find the longest recent match and predict from what followed it
3. Run-length / repetition model
4. Logistic/linear mixing of all model predictions, weights adapt online

Score target: significantly better than 2.3
"""

import numpy as np
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _normalize(p):
    s = p.sum()
    if s <= 0:
        return np.full(len(p), 1.0 / len(p))
    return p / s


def _safe_log(p):
    return np.log(np.maximum(p, 1e-30))


# ──────────────────────────────────────────────────────────────────────────────
# PPM-D context model
# ──────────────────────────────────────────────────────────────────────────────

class PPM_D:
    """
    PPM with escape probability estimated by the method-D heuristic:
        escape_prob = n_unique / (total_count + n_unique)
    where n_unique = number of symbols seen in this context.

    Mixing: use a proper exclusion chain.  Start from highest order;
    if a symbol was NOT seen at that order it is "excluded" from lower orders.
    We implement a simplified version: compute a blended probability using
    escape-weighted mixture going from low to high order.
    """

    def __init__(self, max_order=12, vocab=256):
        self.max_order = max_order
        self.vocab = vocab
        # counts[o][ctx_tuple] -> np.array(256, float32)
        self.counts = [defaultdict(lambda: np.zeros(256, dtype=np.float32))
                       for _ in range(max_order + 1)]
        # unique symbol count per context (for escape estimation)
        self.unique = [defaultdict(int) for _ in range(max_order + 1)]
        self.history = []

    def predict(self):
        v = self.vocab
        history = self.history
        hlen = len(history)

        # Build context tuples for each order (from highest to lowest)
        # Blended prediction using escape-weighting (PPM-D style)
        blended = np.zeros(v, dtype=np.float64)
        carry = 1.0  # remaining probability mass to distribute

        # order 0 fallback
        c0 = self.counts[0][()]
        n0 = c0.sum()
        if n0 > 0:
            u0 = self.unique[0][()]
            esc0 = u0 / (n0 + u0)
            base_prob = c0 / n0
        else:
            esc0 = 1.0
            base_prob = np.ones(v, dtype=np.float64) / v

        # We'll iterate orders 1..max_order from lowest to highest, accumulating
        # from high to low to get proper exclusion, but do it bottom-up for speed
        # Simplified: go bottom-up, each order takes (1-esc)*carry
        orders = [0]
        for o in range(1, self.max_order + 1):
            if hlen >= o:
                orders.append(o)

        # Reset and do top-down: highest order first
        blended = np.zeros(v, dtype=np.float64)
        carry = 1.0

        for o in reversed(orders):
            if o == 0:
                ctx = ()
            else:
                ctx = tuple(history[-o:])

            c = self.counts[o][ctx]
            n = c.sum()

            if n == 0:
                # Context never seen; all mass passes down
                continue

            u = self.unique[o][ctx]
            # Method D: escape = n_unique / (total + n_unique)
            esc = u / (n + u)

            taken = carry * (1.0 - esc)
            blended += taken * (c / n)
            carry *= esc

            if carry < 1e-10:
                break

        if carry > 1e-10:
            # Distribute remaining mass uniformly
            blended += carry / v

        return blended

    def observe(self, byte_val):
        history = self.history
        hlen = len(history)
        for o in range(self.max_order + 1):
            if o == 0:
                ctx = ()
            elif hlen < o:
                continue
            else:
                ctx = tuple(history[-o:])

            cnt_arr = self.counts[o][ctx]
            if cnt_arr[byte_val] == 0:
                self.unique[o][ctx] += 1
            cnt_arr[byte_val] += 1.0

        self.history.append(byte_val)
        if len(self.history) > self.max_order + 2:
            self.history = self.history[-(self.max_order + 2):]


# ──────────────────────────────────────────────────────────────────────────────
# Match model: find repeated substrings and predict continuation
# ──────────────────────────────────────────────────────────────────────────────

class MatchModel:
    """
    Maintains a ring-buffer of recent bytes and searches for the longest
    match of the current context.  The byte that followed each past occurrence
    is used to build a probability estimate.

    Uses a simple hash table keyed on 4-byte hashes for fast lookup.
    """

    def __init__(self, buf_size=1 << 19, min_match=4, vocab=256):  # 512KB buffer
        self.buf_size = buf_size
        self.min_match = min_match
        self.vocab = vocab
        self.buf = np.zeros(buf_size + 64, dtype=np.uint8)
        self.pos = 0
        self.filled = 0

        # Hash table: key = 3-byte hash -> list of positions
        self.table_bits = 16
        self.table_size = 1 << self.table_bits
        self.table_mask = self.table_size - 1
        # Each slot stores up to 4 positions (ring, latest overwrites)
        self.ht = [[] for _ in range(self.table_size)]

    def _hash3(self, history):
        if len(history) < 3:
            return None
        h = (history[-3] * 2654435761 ^ history[-2] * 40503 ^ history[-1]) & self.table_mask
        return h

    def predict(self, history):
        if len(history) < self.min_match:
            return None

        ctx_len = min(len(history), 12)
        ctx = history[-ctx_len:]

        h = self._hash3(ctx)
        if h is None:
            return None

        positions = self.ht[h]
        if not positions:
            return None

        counts = np.zeros(self.vocab, dtype=np.float32)
        buf = self.buf
        buf_size = self.buf_size
        filled = self.filled
        ctx_arr = np.array(ctx, dtype=np.uint8)
        ctx_len2 = len(ctx_arr)

        best_match_len = 0
        for mpos in positions:
            # Check if history matches at mpos-ctx_len
            start = mpos - ctx_len2
            if start < 0:
                continue

            # Verify match
            match_len = 0
            for k in range(ctx_len2):
                bidx = (start + k) % buf_size
                if buf[bidx] == ctx_arr[k]:
                    match_len += 1
                else:
                    break

            if match_len >= self.min_match:
                # The byte that follows
                next_pos = mpos % buf_size
                if next_pos < filled or filled == buf_size:
                    w = match_len  # weight by match length
                    counts[buf[next_pos]] += w
                    if match_len > best_match_len:
                        best_match_len = match_len

        if counts.sum() == 0 or best_match_len < self.min_match:
            return None

        return counts / counts.sum()

    def observe(self, byte_val, history):
        pos = self.pos
        self.buf[pos] = byte_val
        self.pos = (pos + 1) % self.buf_size
        self.filled = min(self.filled + 1, self.buf_size)

        # Update hash table with current position
        if len(history) >= 2:
            h = self._hash3(history)
            if h is not None:
                lst = self.ht[h]
                lst.append(pos)
                if len(lst) > 6:
                    lst.pop(0)


# ──────────────────────────────────────────────────────────────────────────────
# Low-order context model (fast, smoothed, for robustness)
# ──────────────────────────────────────────────────────────────────────────────

class LowOrderModel:
    """Fast order-0 through order-4 model with Laplace smoothing."""

    def __init__(self, max_order=4, vocab=256):
        self.max_order = max_order
        self.vocab = vocab
        self.counts = [defaultdict(lambda: np.ones(256, dtype=np.float32))
                       for _ in range(max_order + 1)]
        self.history = []

    def predict(self):
        v = self.vocab
        history = self.history
        hlen = len(history)
        result = np.zeros(v, dtype=np.float64)
        total_w = 0.0

        weights = [0.05, 0.1, 0.2, 0.3, 0.35]
        for o in range(self.max_order + 1):
            if o > 0 and hlen < o:
                continue
            ctx = () if o == 0 else tuple(history[-o:])
            c = self.counts[o][ctx]
            p = c / c.sum()
            w = weights[o]
            result += w * p
            total_w += w

        return result / total_w

    def observe(self, byte_val):
        history = self.history
        hlen = len(history)
        for o in range(self.max_order + 1):
            if o > 0 and hlen < o:
                continue
            ctx = () if o == 0 else tuple(history[-o:])
            self.counts[o][ctx][byte_val] += 1.0

        self.history.append(byte_val)
        if len(self.history) > self.max_order + 2:
            self.history = self.history[-(self.max_order + 2):]


# ──────────────────────────────────────────────────────────────────────────────
# Adaptive linear mixer
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveMixer:
    """
    Combines N model log-probability vectors using adaptive weights.
    Weights are updated proportionally to each model's recent loss
    (online exponentiated gradient descent on log-loss).
    """

    def __init__(self, n_models, lr=0.05):
        self.n = n_models
        self.lr = lr
        self.weights = np.ones(n_models, dtype=np.float64) / n_models

    def mix(self, log_probs_list):
        """
        log_probs_list: list of length n, each element is shape (256,) log-prob array
        Returns: linear mixture probability array (256,)
        """
        w = self.weights
        result = np.zeros(256, dtype=np.float64)
        for i, lp in enumerate(log_probs_list):
            result += w[i] * np.exp(lp)
        return result

    def update(self, log_probs_list, byte_val):
        """Update weights based on how well each model predicted byte_val."""
        losses = np.array([lp[byte_val] for lp in log_probs_list])  # log-prob of truth
        # Exponentiated gradient: increase weight for better models
        w = self.weights * np.exp(self.lr * losses)
        w /= w.sum()
        self.weights = w


# ──────────────────────────────────────────────────────────────────────────────
# Main predictor
# ──────────────────────────────────────────────────────────────────────────────

class BytePredictor:
    def __init__(self, vocab_size=256, **kwargs):
        self.v = vocab_size
        self.history = []

        # Models
        self.ppm = PPM_D(max_order=12, vocab=vocab_size)
        self.low = LowOrderModel(max_order=4, vocab=vocab_size)
        self.match = MatchModel(buf_size=1 << 19, min_match=4, vocab=vocab_size)

        # Adaptive mixer: 3 models (ppm, low, match)
        self.mixer = AdaptiveMixer(n_models=3, lr=0.08)

        # Cache last prediction for weight update
        self._last_log_probs = None

    def predict(self):
        v = self.v
        history = self.history

        # PPM-D prediction
        p_ppm = self.ppm.predict()
        p_ppm = np.maximum(p_ppm, 1e-8)
        p_ppm /= p_ppm.sum()

        # Low-order prediction
        p_low = self.low.predict()
        p_low = np.maximum(p_low, 1e-8)
        p_low /= p_low.sum()

        # Match model prediction
        p_match_raw = self.match.predict(history)
        if p_match_raw is not None:
            p_match = np.maximum(p_match_raw, 1e-8)
            p_match /= p_match.sum()
        else:
            # Fall back to PPM when no match found
            p_match = p_ppm.copy()

        log_probs = [
            _safe_log(p_ppm),
            _safe_log(p_low),
            _safe_log(p_match),
        ]

        self._last_log_probs = log_probs

        # Mix
        mixed = self.mixer.mix(log_probs)
        mixed = np.maximum(mixed, 1e-8)
        mixed /= mixed.sum()
        return mixed

    def observe(self, byte_val):
        # Update mixer weights if we have a prediction cached
        if self._last_log_probs is not None:
            self.mixer.update(self._last_log_probs, byte_val)
            self._last_log_probs = None

        # Update all sub-models
        self.ppm.observe(byte_val)
        self.low.observe(byte_val)
        self.match.observe(byte_val, self.history)

        self.history.append(byte_val)
        if len(self.history) > 16:
            self.history = self.history[-16:]

        return 0.0


Model = BytePredictor
