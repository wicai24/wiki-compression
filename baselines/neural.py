"""
Neural baseline: GRU byte predictor with online learning.

Single-layer GRU with byte embedding, online SGD training. Adapts to the
data byte-by-byte during compression.

Score: ~1.5 on 200KB Wikipedia chunks (~5.3 bits per byte).

This is a starting point for neural architecture exploration. Key bottlenecks:

1. SLOW ADAPTATION - SGD takes thousands of bytes to learn. The first ~10KB
   is wasted on warmup. Ideas: faster optimizer, learned init, count warmup.

2. LIMITED CONTEXT - GRU hidden state is the only memory. Cannot directly
   attend to patterns 100+ bytes ago. Ideas: attention over recent history,
   explicit memory buffer, sliding window transformer.

3. SINGLE MODEL - One GRU handles markup, text, numbers, whitespace alike.
   Ideas: mixture of experts, context-dependent routing, separate sub-models.

4. NO STRUCTURAL AWARENESS - Treats bytes as flat sequence. Cannot exploit
   XML tags, wiki markup syntax, repeated templates. Ideas: learned groupings,
   positional features, bracket tracking.

5. CALIBRATION - Raw softmax may be overconfident on wrong predictions,
   wasting bits. Ideas: temperature scaling, label smoothing, entropy penalty.
"""

import torch
import torch.nn as nn
import numpy as np


class BytePredictor(nn.Module):
    def __init__(self, vocab_size=256, embed_size=48, hidden_size=256, lr=0.003):
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
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        self.eval()

    @torch.no_grad()
    def predict(self):
        logits = self.output(self.h)
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
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

        self.eval()
        return loss.item()


Model = BytePredictor
