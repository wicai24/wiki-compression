"""
Neural byte predictor for text compression.

CONSTRAINT: This task requires a genuine neural network. The model MUST:
- Inherit nn.Module
- Use learnable layers (nn.Linear, nn.GRUCell, nn.LSTMCell, nn.MultiheadAttention, etc.)
- Train via backpropagation (.backward())
Do NOT replace the neural network with classical counting — iterate on the
ARCHITECTURE. You may combine neural predictions with simple statistics,
but the core prediction must come from a trained neural network.
"""

import torch
import torch.nn as nn
import numpy as np
import math


class BytePredictor(nn.Module):
    def __init__(self, vocab_size=256, embed_size=32, hidden_size=128, lr=0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Deterministic initialization
        torch.manual_seed(42)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRUCell(embed_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

        self.h = torch.zeros(1, hidden_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.step_count = 0

        # Byte frequency counts for blending during warmup
        self.byte_counts = np.ones(vocab_size, dtype=np.float64)  # Laplace smoothing
        self.total_count = float(vocab_size)  # Start with pseudocounts

        self.eval()

    @torch.no_grad()
    def predict(self):
        logits = self.output(self.h)
        neural_probs = torch.softmax(logits, dim=-1).squeeze(0).numpy().astype(np.float64)

        # Frequency-based prior
        freq_probs = self.byte_counts / self.total_count

        # Blend: start mostly frequency-based, transition to neural
        # After ~500 bytes, neural dominates
        alpha = min(1.0, self.step_count / 500.0)
        probs = (1.0 - alpha) * freq_probs + alpha * neural_probs

        # Ensure valid probabilities
        probs = np.maximum(probs, 1e-8)
        probs /= probs.sum()
        return probs

    def observe(self, byte_val):
        self.train()
        target = torch.tensor([byte_val], dtype=torch.long)

        logits = self.output(self.h.detach())
        loss = nn.functional.cross_entropy(logits, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            x = self.embedding(target)
            self.h = self.gru(x, self.h.detach())

        # Update frequency counts
        self.byte_counts[byte_val] += 1.0
        self.total_count += 1.0
        self.step_count += 1

        self.eval()
        return loss.item()


Model = BytePredictor
