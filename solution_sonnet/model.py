"""
Neural byte predictor for text compression.

CONSTRAINT: This task requires a genuine neural network. The model MUST:
- Inherit nn.Module
- Use learnable layers (nn.Linear, nn.GRUCell, nn.LSTMCell, nn.MultiheadAttention, etc.)
- Train via backpropagation (.backward())

Round 3: Dev evaluation of optimized architecture.
- GRU (256 hidden) + Adam + freq blend
- Update every step (best quality) but use smaller model for speed
"""

import torch
import torch.nn as nn
import numpy as np


class BytePredictor(nn.Module):
    def __init__(self, vocab_size=256, embed_size=32, hidden_size=384, lr=0.002):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        torch.manual_seed(42)

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # GRU is faster than LSTM (one state vs two)
        self.gru = nn.GRUCell(embed_size, hidden_size)

        # Output projection
        self.output = nn.Linear(hidden_size, vocab_size)

        # Initialize weights
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

        # GRU hidden state
        self.h = torch.zeros(1, hidden_size)

        # Byte frequency counts for warm-up blend
        self.byte_counts = np.ones(vocab_size, dtype=np.float64)
        self.total_bytes = float(vocab_size)

        # Adam optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-6)

        # Step counter for blend scheduling
        self.step_count = 0

        self.eval()

    @torch.no_grad()
    def predict(self):
        # Neural network prediction
        logits = self.output(self.h)
        neural_probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        # Statistical prediction for warmup
        stat_probs = self.byte_counts / self.total_bytes

        # Blend: start with mostly statistics, transition to neural
        alpha = min(1.0, self.step_count / 500.0)
        probs = alpha * neural_probs + (1.0 - alpha) * stat_probs

        # Normalize
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs

    def observe(self, byte_val):
        self.train()

        # Update byte frequency counts
        self.byte_counts[byte_val] += 1
        self.total_bytes += 1.0

        target = torch.tensor([byte_val], dtype=torch.long)

        # Compute loss and update immediately (online learning)
        logits = self.output(self.h.detach())
        loss = nn.functional.cross_entropy(logits, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update hidden state
        with torch.no_grad():
            x = self.embedding(target)
            self.h = self.gru(x, self.h.detach())

        self.step_count += 1
        self.eval()
        return loss.item()


Model = BytePredictor
