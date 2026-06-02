# Task: Neural Text Compression

## Objective

You are optimizing a **neural text compressor**. Your model predicts the probability distribution of each next byte in a text stream. A provided arithmetic coder converts these predictions into a compressed bitstream. Better predictions produce smaller compressed output.

Your goal: **maximize the compression ratio** while maintaining lossless correctness.

```
score = original_size / (compressed_size + decompressor_size)
```

Higher score is better.

- `compressed_size` = the output of compress.py (bitstream + any serialized model weights)
- `decompressor_size` = total bytes of all `.py` files in your solution directory

For online learning (model adapts from scratch), the compressed output is just the bitstream. For multi-pass strategies (train model first, then compress), trained weights must be serialized into the compressed output — they count as compressed_size.

## The Pipeline

```
Input bytes:   [b0, b1, b2, ..., bN]
                 │
                 ▼
For each byte bi:
  1. probs = model.predict()       ← your model predicts P(next byte | history)
  2. encode(bi, probs)             ← arithmetic coder uses prediction to encode
  3. model.observe(bi)             ← your model updates its state

Output: compressed bitstream (fewer bits when predictions are confident and correct)
```

The decompressor runs the **identical** sequence: `predict → decode → observe`. If your compressor and decompressor models diverge at any point, the output will be corrupted and the submission will be scored as infeasible.

## What You Can Modify

Files in `solution_template/`:
- **`model.py`** — your prediction model (this is the main optimization target)
- **`compress.py`** — the compression script
- **`decompress.py`** — the decompression script
- You may create additional `.py` files in `solution_template/`

## What You Cannot Modify

- `evaluator.py`, `run_evaluator.sh`
- `lib/arithmetic_coding.py`, `lib/utils.py`
- `instances/*` (data chunks)
- `baseline/*`

## Constraints

- **Correctness**: Decompressed output must be byte-identical to the original.
- **Time limit**: 600 seconds per chunk for compression; 600 seconds for decompression. Adjustable via `EVAL_TIMEOUT` env var for different hardware.
- **Allowed imports**: `torch`, `numpy`, standard library utilities. See README.md for the full list.
- **Banned imports**: Compression libraries, system access, networking, pre-trained model loaders. The evaluator performs AST-based import scanning.
- **No web search**: You do not have internet access during this task.

## Batch Evaluation

You can run the evaluator multiple times per iteration to test variants in parallel:

```bash
# Test multiple variants and compare
bash run_evaluator.sh --mode sample    # fast sanity check (~2s)
bash run_evaluator.sh --mode dev       # full evaluation (~30-300s depending on model)
bash run_evaluator.sh --seed 99        # test on random chunks (anti-overfitting)
```

Each invocation is independent. Use sample mode for fast iteration, dev mode for real scoring.

## Running the Evaluator

```bash
# Quick test on small samples
bash run_evaluator.sh --mode sample

# Evaluate on dev chunks (50KB each, requires setup_data.py)
bash run_evaluator.sh --mode dev
```

The evaluator outputs JSON:
```json
{
  "feasible": true,
  "score": 1.85,
  "per_chunk": [
    {"chunk": "dev_00.txt", "ratio": 1.92, "ok": true, "compress_time": 15.3, ...},
    ...
  ],
  "code_size_bytes": 3200,
  "errors": []
}
```

## The Baseline

The starting solution is an **adaptive PPM (Prediction by Partial Matching)** compressor that uses order-0 through order-8 byte-level context statistics with confidence-weighted mixing. It scores approximately **2.3** on 200KB chunks (~3.5 bits per byte).

This is a solid classical approach, but it has known limitations:
- It cannot capture patterns longer than its context window
- Its mixing weights are fixed, not learned
- It treats all bytes the same regardless of position or structure
- It has no mechanism for learning abstract patterns (e.g., that XML tags come in open/close pairs)

For reference, gzip achieves ~2.74 ratio (2.92 bpb) and neural SOTA achieves ~9.4 ratio (0.85 bpb) on full enwik8.

## Evaluator Feedback

The evaluator provides per-segment loss data showing bits-per-byte at each 10KB segment of the chunk. This reveals where your model predicts well and where it struggles. Early segments typically show higher loss (model warming up), and certain data patterns (tables, references, markup) may show loss spikes.

## Things to Consider

- **Prediction quality is everything.** The arithmetic coder is near-optimal given any distribution you provide. Every bit of improvement in your model's log-loss translates directly to fewer compressed bytes.

- **The data is 200KB of Wikipedia XML/markup.** It contains structured patterns (HTML tags, wiki markup, article headings, references) mixed with natural language. At this scale, there are long-range dependencies — repeated templates, cross-references, consistent formatting across sections — that simple local statistics cannot capture.

- **Compressor and decompressor must stay in perfect sync.** Both create a fresh `Model()` and call `predict`/`observe` in lockstep. Any source of non-determinism (random operations, thread ordering, floating-point instability) will cause decompression to fail. Design for reproducibility.

- **Three-way tradeoff.** There is a tension between model capacity (better predictions), speed (must finish within 600s per chunk), and code size (counts in the score denominator). You are free to redesign the architecture, training procedure, or even the compression pipeline itself.

- **This is a fully open-ended optimization problem.** The baseline uses a neural network, but you are not required to. Classical approaches, statistical models, hybrid methods, algorithmic techniques, or entirely novel strategies are all valid. The only constraint is the interface: `predict() → numpy array shape (256,)` and `observe(byte_val) → float`.

- **You are not limited to a single pass.** The baseline processes data in one sequential pass, but other strategies are possible. Any information the decompressor needs must either be derivable from the same process or stored in the compressed output.

- **Explore broadly, not just incrementally.** Tuning hyperparameters on a fixed architecture yields diminishing returns. Fundamentally different modeling approaches — different ways of representing context, different prediction strategies, different ways of combining multiple information sources — tend to produce larger gains. Consider what makes this data compressible and work backward from that.

- **Observe the data.** The evaluation chunks are in `instances/`. Reading and understanding the actual byte patterns, markup structure, and statistical properties of the data can inform better design choices than pure algorithmic intuition.

## Background

Text compression has been studied for decades across multiple paradigms: information theory, statistical modeling, neural sequence prediction, and program synthesis. The Hutter Prize (http://prize.hutter1.net/) demonstrates that compressing text is equivalent to building a good language model — the connection between prediction and compression is fundamental.

State-of-the-art compressors combine ideas from many fields. No single technique dominates; the best results come from understanding the problem deeply and combining approaches creatively. Consider: what information is available at each byte position? How can you represent and exploit that information efficiently? What is the right balance between the model's expressiveness and the cost of maintaining it?
