"""
Neural byte predictor for text compression.

Scores ~1.5 on 200KB Wikipedia chunks. The PPM baseline scores ~2.3.

Look at the per-segment evaluator feedback: the first 10KB typically costs
~5.5 bpb (model knows nothing), while the last 10KB costs ~3.5 bpb (model
has learned). Most bits are wasted during warmup — a model that adapts
faster in the first few thousand bytes would significantly improve the score.

Also note: the hidden state is the model's only memory of past bytes. Once
a pattern scrolls past the effective memory horizon, it's forgotten. The
data contains patterns at multiple scales — individual byte frequencies,
common byte pairs, XML tag structures, repeated phrases across paragraphs.
A model that can represent and recall patterns at different scales would
predict much better than one that compresses everything into a fixed-size
hidden vector.

Finally: the model currently trains on each byte exactly once (online
learning). But the data is available in full — you could train the model
on the data first, serialize the trained weights into the compressed output
header, and then compress using the trained model. The decompressor would
load the weights and decode. The tradeoff: trained weights cost bytes in
the archive, but better predictions save more bytes than the weights cost.
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
