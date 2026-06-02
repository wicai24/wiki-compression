#!/usr/bin/env python3
"""
Bridge script: Frontier-Eng framework -> existing evaluator.

Called by the framework via eval_command.txt. Translates between
Frontier-Eng conventions (metrics.json with combined_score/valid)
and our evaluator (JSON stdout with feasible/score).

Usage:
    python run_eval.py --candidate PATH --benchmark PATH [--mode sample]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True,
                        help='Path to the evolved model.py')
    parser.add_argument('--benchmark', required=True,
                        help='Path to the benchmark directory')
    parser.add_argument('--metrics-out', default='metrics.json',
                        help='Path to write metrics JSON')
    parser.add_argument('--mode', default='sample',
                        help='Evaluation mode: sample|dev|train|hidden')
    args = parser.parse_args()

    benchmark_dir = os.path.abspath(args.benchmark)
    template_dir = os.path.join(benchmark_dir, 'solution_template')
    evaluator_py = os.path.join(benchmark_dir, 'evaluator.py')

    # Create an isolated temp copy of solution_template so parallel runs
    # don't stomp on each other. Only model.py is swapped out.
    with tempfile.TemporaryDirectory(prefix='fe_eval_') as sandbox:
        # Copy all solution template files to sandbox
        for fname in os.listdir(template_dir):
            src = os.path.join(template_dir, fname)
            dst = os.path.join(sandbox, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)

        # Place the candidate model.py in the sandbox
        candidate_path = os.path.join(sandbox, 'model.py')
        shutil.copy2(os.path.abspath(args.candidate), candidate_path)

        # Fast pre-flight: syntax check + import check before expensive eval.
        # Catches truncated LLM output (e.g., "self.x = np." cut mid-line)
        # and missing Model class without running the full compressor.
        try:
            with open(candidate_path) as f:
                source = f.read()
            compile(source, 'model.py', 'exec')
            if 'Model' not in source and 'BytePredictor' not in source:
                raise SyntaxError("No Model or BytePredictor class found")
        except SyntaxError as e:
            metrics = {
                'combined_score': -1e18,
                'valid': 0.0,
                'runtime_s': 0.1,
                'score': 0.0,
                'feasible': 0.0,
                'preflight_error': f'Syntax error in candidate: {e}',
            }
            with open(args.metrics_out, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"PREFLIGHT FAIL: {e}", file=sys.stderr)
            sys.exit(0)

        # Build environment pointing to the sandbox
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join([
            sandbox,
            os.path.join(benchmark_dir, 'lib'),
            benchmark_dir,
        ])
        env['CUDA_VISIBLE_DEVICES'] = ''

        # Run existing evaluator against the sandbox
        cmd = [
            sys.executable, evaluator_py,
            '--mode', args.mode,
            '--solution-dir', sandbox,
            '--budget', '0',
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=1800,
                                env=env, cwd=benchmark_dir)

        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')

    # Translate to Frontier-Eng metrics format
    metrics = {
        'combined_score': -1e18,
        'valid': 0.0,
        'runtime_s': 0.0,
    }

    try:
        eval_result = json.loads(stdout)

        if eval_result.get('feasible', False):
            metrics['valid'] = 1.0
            metrics['combined_score'] = eval_result.get('score', 0.0)
        else:
            metrics['valid'] = 0.0
            metrics['combined_score'] = -1e18

        # Pass through detailed metrics
        metrics['score'] = eval_result.get('score', 0.0)
        metrics['feasible'] = 1.0 if eval_result.get('feasible') else 0.0
        metrics['code_size_bytes'] = eval_result.get('code_size_bytes', 0)
        metrics['wall_time_sec'] = eval_result.get('wall_time_sec', 0.0)
        metrics['runtime_s'] = eval_result.get('wall_time_sec', 0.0)
        metrics['num_chunks'] = len(eval_result.get('per_chunk', []))

        for i, chunk in enumerate(eval_result.get('per_chunk', [])):
            metrics[f'chunk_{i}_ratio'] = chunk.get('ratio', 0.0)
            metrics[f'chunk_{i}_ok'] = 1.0 if chunk.get('ok') else 0.0

    except (json.JSONDecodeError, KeyError) as e:
        metrics['parse_error'] = str(e)

    # Write metrics.json
    with open(args.metrics_out, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print to stderr for debugging
    print(f"Score: {metrics.get('score', 0):.4f}, "
          f"Valid: {metrics['valid']}, "
          f"Combined: {metrics['combined_score']}", file=sys.stderr)

    sys.exit(0)


if __name__ == '__main__':
    main()
