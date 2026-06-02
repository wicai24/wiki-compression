#!/usr/bin/env bash
# Run the evaluator for the Neural Text Compression task.
#
# Usage:
#   bash run_evaluator.sh                    # evaluate on sample data
#   bash run_evaluator.sh --mode dev         # evaluate on dev chunks (need setup_data.py)
#   bash run_evaluator.sh --mode train       # evaluate on train chunks
#   bash run_evaluator.sh --mode hidden      # evaluate on hidden chunks (final eval)
#   bash run_evaluator.sh --seed 42          # random chunks from enwik8 (anti-memorization)
#   bash run_evaluator.sh --budget 50        # limit total evaluator invocations
#
# Output: JSON results to stdout, progress to stderr.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PYTHONPATH so solution can import from lib/
export PYTHONPATH="${SCRIPT_DIR}/lib:${SCRIPT_DIR}:${PYTHONPATH:-}"

# Force CPU for reproducibility
export CUDA_VISIBLE_DEVICES=""

# Detect Python — prefer the same python that has torch installed
PYTHON="${PYTHON:-$(command -v python 2>/dev/null || command -v python3 2>/dev/null)}"

# Run evaluator
"${PYTHON}" "${SCRIPT_DIR}/evaluator.py" "$@"
