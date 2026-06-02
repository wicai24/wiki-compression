"""Tests for the evaluator."""

import json
import os
import sys
import subprocess
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_evaluator(mode='sample', solution_dir=None):
    """Run evaluator and return parsed JSON result."""
    cmd = [sys.executable, os.path.join(PROJECT_ROOT, 'evaluator.py'), '--mode', mode]
    if solution_dir:
        cmd.extend(['--solution-dir', solution_dir])

    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([
        os.path.join(PROJECT_ROOT, 'lib'),
        PROJECT_ROOT,
    ])

    result = subprocess.run(
        cmd, capture_output=True, timeout=300, env=env,
        cwd=PROJECT_ROOT,
    )
    stdout = result.stdout.decode('utf-8')
    # Parse JSON from stdout (skip any non-JSON prefix)
    return json.loads(stdout)


def test_evaluator_runs():
    """Evaluator produces valid JSON output."""
    result = _run_evaluator('sample')
    assert 'feasible' in result
    assert 'score' in result
    assert 'per_chunk' in result
    assert 'code_size_bytes' in result
    assert 'errors' in result
    assert 'wall_time_sec' in result


def test_evaluator_baseline_feasible():
    """Baseline solution should be feasible on sample data."""
    result = _run_evaluator('sample')
    assert result['feasible'] is True
    assert result['score'] > 0


def test_evaluator_per_chunk_format():
    """Each chunk result should have required fields."""
    result = _run_evaluator('sample')
    for chunk in result['per_chunk']:
        assert 'chunk' in chunk
        assert 'original_size' in chunk
        assert 'compressed_size' in chunk
        assert 'ratio' in chunk
        assert 'ok' in chunk
        assert 'compress_time' in chunk
        assert 'decompress_time' in chunk


def test_evaluator_catches_banned_import():
    """Evaluator should reject solution with banned imports."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a solution that imports zlib
        with open(os.path.join(tmpdir, 'compress.py'), 'w') as f:
            f.write("import zlib\nimport sys\ndata=sys.stdin.buffer.read()\nsys.stdout.buffer.write(zlib.compress(data))")
        with open(os.path.join(tmpdir, 'decompress.py'), 'w') as f:
            f.write("import zlib\nimport sys\ndata=sys.stdin.buffer.read()\nsys.stdout.buffer.write(zlib.decompress(data))")

        result = _run_evaluator('sample', solution_dir=tmpdir)
        assert result['feasible'] is False
        assert any('zlib' in e for e in result['errors'])


def test_evaluator_catches_crash():
    """Evaluator should handle crashing solutions gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'compress.py'), 'w') as f:
            f.write("raise RuntimeError('intentional crash')")
        with open(os.path.join(tmpdir, 'decompress.py'), 'w') as f:
            f.write("pass")

        result = _run_evaluator('sample', solution_dir=tmpdir)
        assert result['feasible'] is False


def test_evaluator_catches_wrong_output():
    """Evaluator should detect when decompression produces wrong output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Compress correctly but decompress outputs garbage
        with open(os.path.join(tmpdir, 'compress.py'), 'w') as f:
            f.write("import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())")
        with open(os.path.join(tmpdir, 'decompress.py'), 'w') as f:
            f.write("import sys; sys.stdout.buffer.write(b'wrong output')")

        result = _run_evaluator('sample', solution_dir=tmpdir)
        assert result['feasible'] is False
