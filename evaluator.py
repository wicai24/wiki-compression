#!/usr/bin/env python3
"""
Evaluator for the Neural Text Compression Optimization task.

This is a READ-ONLY component. The agent must not modify this file.

Usage:
    python evaluator.py --mode sample|dev|train|hidden [--solution-dir DIR]

Returns JSON to stdout:
    {
        "feasible": true/false,
        "score": float,
        "per_chunk": [...],
        "code_size_bytes": int,
        "errors": [...],
        "wall_time_sec": float
    }

Also appends results to eval_history.jsonl.
"""

import argparse
import ast
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time


# ─── Configuration ───────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TIMEOUT_PER_CHUNK = int(os.environ.get('EVAL_TIMEOUT', 600))
MEMORY_LIMIT_MB = 4096

# Modules that solutions are ALLOWED to import
ALLOWED_MODULES = {
    # Core
    'sys', 'io', 'struct', 'math', 'cmath',
    # Collections and functional
    'collections', 'functools', 'itertools', 'operator', 'bisect',
    'heapq', 'array', 'copy', 'enum',
    # Typing
    'typing', 'typing_extensions', 'dataclasses', 'abc',
    # Numeric
    'numpy', 'numpy.random', 'numpy.linalg',
    # ML framework
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim',
    'torch.nn.init', 'torch.nn.utils',
    # Time (for internal benchmarking, not cheating)
    'time', 'warnings',
    # Read-only lib modules (provided infrastructure)
    'arithmetic_coding', 'utils',
    'lib', 'lib.arithmetic_coding', 'lib.utils',
    # Local imports within solution_template/ (handled separately)
}

# Modules explicitly BANNED (overrides allowed)
BANNED_MODULES = {
    # Compression libraries
    'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
    'zstandard', 'snappy', 'lz4',
    # System access
    'subprocess', 'os', 'shutil', 'pathlib', 'signal',
    'ctypes', 'cffi', 'multiprocessing', 'threading',
    # Network
    'urllib', 'urllib.request', 'http', 'http.client',
    'requests', 'socket', 'ssl', 'ftplib', 'smtplib',
    # Dynamic code execution
    'importlib', 'importlib.util', 'pkgutil',
    # Pre-trained model loading
    'torch.hub', 'torchvision', 'torchaudio',
    'transformers', 'huggingface_hub',
    # Serialization (prevents smuggling pre-trained weights)
    'pickle', 'shelve', 'marshal',
    # File I/O (solution uses stdin/stdout only)
    'open',  # Note: this is a builtin, handled separately
}

# Banned builtins / attributes
BANNED_CALLS = {
    '__import__', 'exec', 'eval', 'compile',
    'os.system', 'os.popen', 'os.exec',
    'open',       # prevent reading test data, evaluator source, or writing files
    'io.open',    # alternative file open
    'builtins.open',
}


# ─── Import Scanner ──────────────────────────────────────────────────────────

class ImportViolation:
    def __init__(self, filename, line, module, reason):
        self.filename = filename
        self.line = line
        self.module = module
        self.reason = reason

    def __str__(self):
        return f"{self.filename}:{self.line}: banned import '{self.module}' ({self.reason})"


def scan_imports(solution_dir):
    """AST-scan all .py files in solution_dir for banned imports."""
    violations = []
    local_modules = set()

    # Discover local module names
    for fname in os.listdir(solution_dir):
        if fname.endswith('.py'):
            local_modules.add(fname[:-3])

    for fname in os.listdir(solution_dir):
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(solution_dir, fname)
        with open(fpath, 'r') as f:
            try:
                tree = ast.parse(f.read(), filename=fname)
            except SyntaxError as e:
                violations.append(ImportViolation(fname, e.lineno or 0, '<syntax>',
                                                  f"Syntax error: {e}"))
                continue

        for node in ast.walk(tree):
            # Check import statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _check_module(violations, fname, node.lineno, alias.name,
                                  local_modules)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    _check_module(violations, fname, node.lineno, node.module,
                                  local_modules)

            # Check for __import__, exec, eval calls
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in BANNED_CALLS:
                    violations.append(ImportViolation(
                        fname, node.lineno, func.id,
                        "Banned function call"))
                elif isinstance(func, ast.Attribute):
                    attr_str = _get_attr_string(func)
                    if attr_str and attr_str in BANNED_CALLS:
                        violations.append(ImportViolation(
                            fname, node.lineno, attr_str,
                            "Banned function call"))

    return violations


def _check_module(violations, fname, lineno, module_name, local_modules):
    """Check if a module import is allowed."""
    # Local imports are always OK
    if module_name in local_modules:
        return
    # lib.* imports are OK (read-only infrastructure)
    if module_name.startswith('lib.') or module_name == 'lib':
        return

    # Check against ban list (check prefixes too)
    for banned in BANNED_MODULES:
        if module_name == banned or module_name.startswith(banned + '.'):
            violations.append(ImportViolation(
                fname, lineno, module_name,
                f"Module '{banned}' is banned"))
            return

    # Check against allow list
    top_level = module_name.split('.')[0]
    allowed = False
    for allowed_mod in ALLOWED_MODULES:
        if module_name == allowed_mod or module_name.startswith(allowed_mod + '.'):
            allowed = True
            break
        if top_level == allowed_mod.split('.')[0]:
            allowed = True
            break

    if not allowed:
        violations.append(ImportViolation(
            fname, lineno, module_name,
            f"Module not in allowed list"))


def _get_attr_string(node):
    """Reconstruct attribute access string like 'os.system'."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return '.'.join(reversed(parts))
    return None


# ─── Evaluation ──────────────────────────────────────────────────────────────

def get_chunks(mode):
    """Get list of chunk file paths for the given mode."""
    chunk_dir = os.path.join(SCRIPT_DIR, 'instances', mode)
    if not os.path.isdir(chunk_dir):
        print(f"Error: directory {chunk_dir} does not exist.", file=sys.stderr)
        print(f"Run 'python setup_data.py' first to generate data.", file=sys.stderr)
        sys.exit(1)
    chunks = sorted(glob.glob(os.path.join(chunk_dir, '*.txt')))
    if not chunks:
        print(f"Error: no .txt files found in {chunk_dir}", file=sys.stderr)
        sys.exit(1)
    return chunks


# ─── Dynamic Instance Generation ────────────────────────────────────────────

ENWIK8_SIZE = 100_000_000  # 10^8 bytes

def generate_random_chunks(seed, num_chunks=3, chunk_size=100_000):
    """
    Generate random chunks from enwik8 at seeded offsets.

    This satisfies the Frontier-Eng requirement that the evaluator can
    "auto-generate a family of same-distribution problem instances."
    Different seeds produce different chunks, preventing memorization.

    Returns list of (chunk_name, chunk_data) tuples.
    """
    enwik8_path = os.path.join(SCRIPT_DIR, 'enwik8')
    if not os.path.exists(enwik8_path):
        print("Error: enwik8 not found. Run 'python setup_data.py' first.",
              file=sys.stderr)
        sys.exit(1)

    rng = random.Random(seed)
    max_offset = ENWIK8_SIZE - chunk_size

    with open(enwik8_path, 'rb') as f:
        data = f.read()

    chunks = []
    offsets_used = set()
    for i in range(num_chunks):
        # Generate non-overlapping offsets
        while True:
            offset = rng.randint(0, max_offset)
            # Ensure at least chunk_size separation from other chunks
            if all(abs(offset - o) >= chunk_size for o in offsets_used):
                break
        offsets_used.add(offset)

        chunk_data = data[offset:offset + chunk_size]
        chunk_name = f"random_s{seed}_c{i}.txt"
        chunks.append((chunk_name, chunk_data))

    return chunks


def get_chunks_for_mode(mode, seed=None, num_chunks=3, chunk_size=100_000):
    """
    Get chunks either from fixed files (mode) or generated randomly (seed).

    If seed is provided, generates random chunks from enwik8.
    Otherwise, loads fixed chunks from instances/{mode}/.
    """
    if seed is not None:
        return generate_random_chunks(seed, num_chunks, chunk_size)
    else:
        chunk_paths = get_chunks(mode)
        result = []
        for path in chunk_paths:
            with open(path, 'rb') as f:
                result.append((os.path.basename(path), f.read()))
        return result


# ─── Budget Enforcement ──────────────────────────────────────────────────────

DEFAULT_BUDGET = 100  # maximum evaluator invocations

def check_budget(budget):
    """
    Count past evaluator runs and check if budget is exhausted.

    Returns (runs_used, budget_remaining). Exits if budget is exhausted.
    """
    history_path = os.path.join(SCRIPT_DIR, 'eval_history.jsonl')
    runs_used = 0
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    runs_used += 1

    remaining = budget - runs_used
    if remaining <= 0:
        result = {
            'feasible': False,
            'score': 0.0,
            'per_chunk': [],
            'code_size_bytes': 0,
            'errors': [f"Budget exhausted: {runs_used}/{budget} runs used. "
                       f"Use --budget to increase or delete eval_history.jsonl to reset."],
            'wall_time_sec': 0,
        }
        print(json.dumps(result, indent=2))
        sys.exit(0)

    return runs_used, remaining


def get_code_size(solution_dir):
    """Total size of all .py files in the solution directory."""
    total = 0
    for fname in os.listdir(solution_dir):
        if fname.endswith('.py'):
            total += os.path.getsize(os.path.join(solution_dir, fname))
    return total


def evaluate_chunk(chunk_name, original_data, solution_dir):
    """
    Run compress + decompress on a single chunk IN-PROCESS.

    Runs both compress and decompress in the same Python process, eliminating
    subprocess-level non-determinism (different random states, thread scheduling,
    float accumulation order). This makes neural approaches viable — the model
    only needs to be deterministic given the same seed, not across processes.

    Also collects per-segment loss data to help the agent identify weak spots.

    Args:
        chunk_name: display name for the chunk
        original_data: raw bytes of the chunk
        solution_dir: path to the solution directory

    Returns dict with keys: chunk, original_size, compressed_size, ok, error,
                            compress_time, decompress_time, segments
    """
    import io as _io
    import importlib.util
    import math

    SEGMENT_SIZE = 10000  # 10KB segments for per-segment feedback

    result = {
        'chunk': chunk_name,
        'original_size': len(original_data),
        'compressed_size': 0,
        'ok': False,
        'error': None,
        'compress_time': 0,
        'decompress_time': 0,
        'segments': [],
    }

    model_path = os.path.join(solution_dir, 'model.py')
    if not os.path.exists(model_path):
        result['error'] = f"model.py not found in {solution_dir}"
        return result

    # Add solution dir and lib to path for imports
    lib_dir = os.path.join(SCRIPT_DIR, 'lib')
    old_path = sys.path[:]
    sys.path.insert(0, solution_dir)
    sys.path.insert(0, lib_dir)
    sys.path.insert(0, SCRIPT_DIR)

    # Force CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    try:
        # Import the arithmetic coding library
        from arithmetic_coding import ArithmeticEncoder, ArithmeticDecoder
        from arithmetic_coding import BitOutputStream, BitInputStream
        from utils import probs_to_cumfreqs, write_header, read_header

        # Load the model module fresh each time
        spec = importlib.util.spec_from_file_location("_eval_model", model_path)
        model_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_mod)
        ModelClass = model_mod.Model

        # --- Compress ---
        t0 = time.time()
        model_c = ModelClass()
        buf = _io.BytesIO()
        write_header(buf, len(original_data))
        bit_out = BitOutputStream(buf)
        encoder = ArithmeticEncoder(bit_out)

        # Per-segment loss tracking
        segment_losses = []
        current_segment_loss = 0.0
        current_segment_count = 0

        for i, byte_val in enumerate(original_data):
            probs = model_c.predict()
            cumfreqs = probs_to_cumfreqs(probs)
            encoder.write(cumfreqs, byte_val)

            # Track per-byte loss for segment feedback
            p = max(float(probs[byte_val]), 1e-10)
            current_segment_loss += -math.log2(p)
            current_segment_count += 1

            if current_segment_count >= SEGMENT_SIZE:
                avg_bpb = current_segment_loss / current_segment_count
                segment_losses.append({
                    'start': i - current_segment_count + 1,
                    'end': i + 1,
                    'bpb': round(avg_bpb, 3),
                })
                current_segment_loss = 0.0
                current_segment_count = 0

            model_c.observe(byte_val)

            # Timeout check every 10K bytes
            if i % 10000 == 0 and (time.time() - t0) > TIMEOUT_PER_CHUNK:
                result['error'] = f"Compression timed out after {TIMEOUT_PER_CHUNK}s at byte {i}/{len(original_data)}"
                return result

        # Final partial segment
        if current_segment_count > 0:
            avg_bpb = current_segment_loss / current_segment_count
            segment_losses.append({
                'start': len(original_data) - current_segment_count,
                'end': len(original_data),
                'bpb': round(avg_bpb, 3),
            })

        encoder.finish()
        compressed_data = buf.getvalue()
        result['compress_time'] = time.time() - t0
        result['compressed_size'] = len(compressed_data)
        result['segments'] = segment_losses

        # --- Decompress (fresh model instance, same process) ---
        t0 = time.time()
        model_d = ModelClass()
        buf2 = _io.BytesIO(compressed_data)
        original_length = read_header(buf2)
        bit_in = BitInputStream(buf2)
        decoder = ArithmeticDecoder(bit_in)

        restored = bytearray()
        for i in range(original_length):
            probs = model_d.predict()
            cumfreqs = probs_to_cumfreqs(probs)
            byte_val = decoder.read(cumfreqs)
            restored.append(byte_val)
            model_d.observe(byte_val)

            if i % 10000 == 0 and (time.time() - t0) > TIMEOUT_PER_CHUNK:
                result['error'] = f"Decompression timed out after {TIMEOUT_PER_CHUNK}s at byte {i}/{original_length}"
                return result

        result['decompress_time'] = time.time() - t0

        # --- Verify byte-exact match ---
        restored_data = bytes(restored)
        if restored_data != original_data:
            if len(restored_data) != len(original_data):
                result['error'] = (
                    f"Length mismatch: original={len(original_data)}, "
                    f"restored={len(restored_data)}"
                )
            else:
                for i in range(len(original_data)):
                    if restored_data[i] != original_data[i]:
                        result['error'] = (
                            f"Content mismatch at byte {i}: "
                            f"expected {original_data[i]}, got {restored_data[i]}"
                        )
                        break
            return result

        result['ok'] = True
        return result

    except Exception as e:
        import traceback
        result['error'] = f"Evaluation error: {str(e)}\n{traceback.format_exc()[-500:]}"
        return result
    finally:
        sys.path[:] = old_path
        # Clean up imported module to avoid state leaks between chunks
        if '_eval_model' in sys.modules:
            del sys.modules['_eval_model']


def run_evaluation(mode, solution_dir, seed=None, num_chunks=3, chunk_size=100_000, require_neural=False):
    """Run full evaluation and return results dict."""
    t_start = time.time()
    errors = []

    # Step 1: Import scanning
    print(f"[1/3] Scanning imports in {solution_dir} ...", file=sys.stderr)
    violations = scan_imports(solution_dir)
    if violations:
        error_msgs = [str(v) for v in violations]
        return {
            'feasible': False,
            'score': 0.0,
            'per_chunk': [],
            'code_size_bytes': get_code_size(solution_dir),
            'errors': error_msgs,
            'wall_time_sec': time.time() - t_start,
        }

    # Step 1b: Neural enforcement (optional)
    if require_neural:
        model_path = os.path.join(solution_dir, 'model.py')
        if os.path.exists(model_path):
            with open(model_path, 'r') as f:
                source = f.read()

            # Must use torch AND have a genuine nn.Module with learnable parameters
            has_torch = 'import torch' in source or 'from torch' in source
            has_nn_module = 'nn.Module' in source
            has_parameters = 'nn.Linear' in source or 'nn.Embedding' in source or \
                             'nn.GRUCell' in source or 'nn.LSTMCell' in source or \
                             'nn.RNNCell' in source or 'nn.GRU(' in source or \
                             'nn.LSTM(' in source or 'nn.Conv' in source or \
                             'nn.TransformerEncoder' in source or \
                             'nn.MultiheadAttention' in source
            has_backward = '.backward()' in source or 'loss.backward' in source

            if not (has_torch and has_nn_module and has_parameters and has_backward):
                missing = []
                if not has_torch: missing.append('import torch')
                if not has_nn_module: missing.append('nn.Module subclass')
                if not has_parameters: missing.append('learnable layers (nn.Linear, nn.GRUCell, etc.)')
                if not has_backward: missing.append('gradient-based training (.backward())')
                return {
                    'feasible': False,
                    'score': 0.0,
                    'per_chunk': [],
                    'code_size_bytes': get_code_size(solution_dir),
                    'errors': [f'Neural enforcement: model.py must use a genuine neural network. '
                               f'Missing: {", ".join(missing)}. '
                               f'The model must inherit nn.Module, use learnable layers, '
                               f'and train via backpropagation.'],
                    'wall_time_sec': time.time() - t_start,
                }

    # Step 2: Get chunks and code size
    chunks = get_chunks_for_mode(mode, seed=seed,
                                 num_chunks=num_chunks, chunk_size=chunk_size)
    code_size = get_code_size(solution_dir)
    source = f"seed={seed}" if seed is not None else f"mode={mode}"
    print(f"[2/3] Code size: {code_size} bytes, evaluating {len(chunks)} chunks "
          f"({source}) ...", file=sys.stderr)

    # Step 3: Evaluate each chunk
    chunk_results = []
    all_ok = True
    for i, (chunk_name, chunk_data) in enumerate(chunks):
        print(f"  [{i+1}/{len(chunks)}] {chunk_name} ...",
              end='', file=sys.stderr, flush=True)
        result = evaluate_chunk(chunk_name, chunk_data, solution_dir)
        chunk_results.append(result)

        if result['ok']:
            ratio = result['original_size'] / (result['compressed_size'] + code_size)
            result['ratio'] = ratio
            print(f" ratio={ratio:.3f} ({result['compress_time']:.1f}s + "
                  f"{result['decompress_time']:.1f}s)", file=sys.stderr)
        else:
            all_ok = False
            result['ratio'] = 0.0
            errors.append(result['error'])
            print(f" FAILED: {result['error'][:80]}", file=sys.stderr)

    # Compute aggregate score
    if all_ok and chunk_results:
        score = sum(r['ratio'] for r in chunk_results) / len(chunk_results)
    else:
        score = 0.0

    wall_time = time.time() - t_start
    print(f"[3/3] Done in {wall_time:.1f}s. "
          f"{'FEASIBLE' if all_ok else 'INFEASIBLE'}, score={score:.4f}",
          file=sys.stderr)

    return {
        'feasible': all_ok,
        'score': round(score, 6),
        'per_chunk': [
            {
                'chunk': r['chunk'],
                'original_size': r['original_size'],
                'compressed_size': r['compressed_size'],
                'ratio': round(r.get('ratio', 0), 6),
                'ok': r['ok'],
                'error': r.get('error'),
                'compress_time': round(r['compress_time'], 2),
                'decompress_time': round(r['decompress_time'], 2),
                'segments': r.get('segments', []),
            }
            for r in chunk_results
        ],
        'code_size_bytes': code_size,
        'errors': errors,
        'wall_time_sec': round(wall_time, 2),
    }


def main():
    parser = argparse.ArgumentParser(description='Neural Text Compression Evaluator')
    parser.add_argument('--mode', choices=['sample', 'dev', 'train', 'hidden'],
                        default='sample',
                        help='Which data split to evaluate on (default: sample)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for dynamic instance generation. '
                             'When set, generates random chunks from enwik8 instead '
                             'of using fixed files from instances/.')
    parser.add_argument('--num-chunks', type=int, default=3,
                        help='Number of random chunks to generate (with --seed)')
    parser.add_argument('--chunk-size', type=int, default=100_000,
                        help='Size of each random chunk in bytes (with --seed)')
    parser.add_argument('--budget', type=int, default=DEFAULT_BUDGET,
                        help=f'Max evaluator invocations (default: {DEFAULT_BUDGET}). '
                             f'Set to 0 for unlimited.')
    parser.add_argument('--solution-dir', default=None,
                        help='Path to solution directory (default: solution_template/)')
    parser.add_argument('--require-neural', action='store_true',
                        help='Require model.py to import torch (enforce neural approach)')
    args = parser.parse_args()

    solution_dir = args.solution_dir or os.path.join(SCRIPT_DIR, 'solution_template')
    solution_dir = os.path.abspath(solution_dir)

    if not os.path.isdir(solution_dir):
        print(f"Error: solution directory {solution_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Budget enforcement
    if args.budget > 0:
        runs_used, remaining = check_budget(args.budget)
        print(f"Budget: {runs_used}/{args.budget} used, {remaining} remaining",
              file=sys.stderr)

    results = run_evaluation(args.mode, solution_dir,
                             seed=args.seed,
                             num_chunks=args.num_chunks,
                             chunk_size=args.chunk_size,
                             require_neural=args.require_neural)

    # Output JSON to stdout (this is what the agent/harness parses)
    print(json.dumps(results, indent=2))

    # Append to eval history
    history_path = os.path.join(SCRIPT_DIR, 'eval_history.jsonl')
    with open(history_path, 'a') as f:
        record = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'mode': args.mode,
            'seed': args.seed,
            **results,
        }
        f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    main()
