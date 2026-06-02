# Wiki Compression: Neural Text Compression Optimization

A [Frontier-Eng](https://github.com/EinsiaLab/Frontier-Engineering) style optimization task where an agent iteratively improves a text compressor to achieve better compression of Wikipedia data.

Based on the [Hutter Prize](http://prize.hutter1.net/) — compressing text is equivalent to building a good prediction model. An arithmetic coder converts predictions into compressed bits. Better predictions = fewer bits = higher score.

## Quick Start

```bash
# Install dependencies
pip install numpy torch pytest

# Download and prepare evaluation data (200KB enwik8 chunks)
python setup_data.py

# Run evaluator on sample data (included, no download needed)
bash run_evaluator.sh --mode sample

# Run on dev data
bash run_evaluator.sh --mode dev
```

## How It Works

```
Agent modifies solution_template/model.py
         │
         ▼
bash run_evaluator.sh --mode dev
         │
         ▼
Evaluator: import model → compress → decompress → verify → score
         │  (in-process evaluation — no subprocess determinism issues)
         ▼
JSON output with score, per-chunk ratios, and per-segment loss feedback
         │
         ▼
Agent reads feedback, plans next improvement
```

### Scoring

```
score = original_size / (compressed_size + decompressor_size)
```

- **Higher is better.** Score of 1.0 means no compression.
- `decompressor_size` = total bytes of all `.py` files in `solution_template/`
- Feasibility requires **byte-exact** lossless decompression.

### Evaluator Feedback

The evaluator provides per-segment (10KB) bits-per-byte data, showing where the model predicts well and where it wastes bits:

```json
{
  "feasible": true,
  "score": 2.45,
  "per_chunk": [{
    "chunk": "dev_00.txt",
    "ratio": 2.48,
    "segments": [
      {"start": 0, "end": 10000, "bpb": 4.7},
      {"start": 10000, "end": 20000, "bpb": 3.1},
      ...
    ]
  }]
}
```

## Baselines

Three starting points are provided in `baseline/`:

| Baseline | Score (200KB) | Description |
|----------|---------------|-------------|
| `noop.py` | ~0.98 | Uniform distribution — no learning. Maximum exploration freedom. |
| `ppm.py` | ~2.3 | Order-8 PPM with confidence-weighted mixing. Strong classical baseline. **Default.** |
| `neural.py` | ~1.5 | GRU with online SGD. Starting point for neural architecture exploration. |

To switch baseline:
```bash
cp baseline/neural.py solution_template/model.py
```

The default (`ppm.py`) is recommended — it forces the agent to go beyond simple n-gram counting.

## Directory Structure

```
├── README.md                  # This file
├── task.md                    # Agent-facing problem statement
├── metadata.json              # Task metadata
├── packaging_manifest.json    # File permissions manifest
├── requirements.txt           # Python dependencies
├── setup_data.py              # Downloads enwik8, extracts chunks
├── evaluator.py               # In-process evaluator (read-only)
├── run_evaluator.sh           # One-command entry point
├── lib/                       # Read-only infrastructure
│   ├── arithmetic_coding.py   # Integer arithmetic encoder/decoder
│   └── utils.py               # Scoring and frequency helpers
├── baseline/                 # Starting points (read-only)
│   ├── noop.py                # Uniform distribution
│   ├── ppm.py                 # Classical PPM (default)
│   └── neural.py              # GRU byte predictor
├── solution_template/         # Agent-editable
│   ├── model.py               # The prediction model (main target)
│   ├── compress.py            # Compression pipeline
│   └── decompress.py          # Decompression pipeline
├── instances/                 # Evaluation data
│   ├── sample/                # ~5KB each (in repo)
│   ├── dev/                   # 200KB each (setup_data.py)
│   ├── train/                 # 200KB each (setup_data.py)
│   └── hidden/                # 200KB each (setup_data.py)
├── tests/                     # Test suite (50 tests)
├── frontier_eval/             # Frontier-Eng framework integration
│   ├── run_eval.py            # Bridge script
│   └── *.txt                  # Metadata files
└── validation_report.md       # Empirical analysis and trajectory data
```

## Evaluation Modes

| Mode | Chunks | Size | Purpose |
|------|--------|------|---------|
| `sample` | 2 × ~5KB | In repo | Pipeline testing |
| `dev` | 2 × 200KB | setup_data.py | Fast iteration |
| `train` | 3 × 200KB | setup_data.py | Scoring |
| `hidden` | 3 × 200KB | setup_data.py | Final evaluation |

Dynamic instance generation (anti-memorization):
```bash
bash run_evaluator.sh --seed 42 --num-chunks 3 --chunk-size 200000
```

## Frontier-Eng Integration

This task plugs into the [Frontier-Engineering](https://github.com/EinsiaLab/Frontier-Engineering) benchmark framework:

```bash
# From the Frontier-Eng repo root (after symlinking this task into benchmarks/AIInfra/NeuralTextCompression/)
python -m frontier_eval task=unified task.benchmark=AIInfra/NeuralTextCompression algorithm=openevolve algorithm.iterations=20
```

Configure the LLM via `.env`:
```
OPENAI_API_KEY=your-key
OPENAI_API_BASE=https://api.anthropic.com/v1/
OPENAI_MODEL=claude-sonnet-4-6
```

## Constraints

- **Time**: 600 seconds per chunk (compress + decompress). Calibrated for Apple Silicon / modern x86. Adjust via `EVAL_TIMEOUT` env var.
- **Allowed imports**: `torch`, `numpy`, standard library utilities
- **Banned**: compression libraries (`zlib`, `lzma`), system access (`subprocess`, `os`), pre-trained model loading (`torch.hub`, `pickle`)
- **Determinism**: Compress and decompress create separate `Model()` instances that must produce identical prediction sequences

## Running Tests

```bash
python -m pytest tests/ -v
```

## Reference Scores

| Approach | Score | BPB | Notes |
|----------|-------|-----|-------|
| No-op | 0.98 | 8.2 | Uniform distribution |
| Neural GRU baseline | 1.5 | 5.3 | Single-layer GRU |
| PPM baseline (default) | 2.3 | 3.5 | Order-8 adaptive mixing |
| Best auto-discovered (PPM-D) | 2.7 | 2.9 | Matches gzip |
| gzip | 2.74 | 2.92 | Standard reference |
| LZMA | 4.02 | 1.99 | Strong reference |
| NNCP SOTA | 9.41 | 0.85 | Neural, on full enwik8 |
