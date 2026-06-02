"""Tests for the evaluator's anti-hack import scanner."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluator import scan_imports


def _write_and_scan(code_snippets):
    """Write code to a temp dir and run the import scanner."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, code in code_snippets.items():
            with open(os.path.join(tmpdir, fname), 'w') as f:
                f.write(code)
        return scan_imports(tmpdir)


# --- Should be BANNED ---

def test_ban_import_zlib():
    violations = _write_and_scan({'test.py': 'import zlib'})
    assert len(violations) >= 1
    assert any('zlib' in str(v) for v in violations)


def test_ban_from_zlib():
    violations = _write_and_scan({'test.py': 'from zlib import compress'})
    assert len(violations) >= 1


def test_ban_import_subprocess():
    violations = _write_and_scan({'test.py': 'import subprocess'})
    assert len(violations) >= 1


def test_ban_from_subprocess():
    violations = _write_and_scan({'test.py': 'from subprocess import run'})
    assert len(violations) >= 1


def test_ban_import_os():
    violations = _write_and_scan({'test.py': 'import os'})
    assert len(violations) >= 1


def test_ban_import_lzma():
    violations = _write_and_scan({'test.py': 'import lzma'})
    assert len(violations) >= 1


def test_ban_import_pickle():
    violations = _write_and_scan({'test.py': 'import pickle'})
    assert len(violations) >= 1


def test_ban_importlib():
    violations = _write_and_scan({'test.py': 'import importlib'})
    assert len(violations) >= 1


def test_ban_torch_hub():
    violations = _write_and_scan({'test.py': 'import torch.hub'})
    assert len(violations) >= 1


def test_ban_transformers():
    violations = _write_and_scan({'test.py': 'import transformers'})
    assert len(violations) >= 1


def test_ban_dunder_import():
    violations = _write_and_scan({'test.py': "__import__('os')"})
    assert len(violations) >= 1


def test_ban_exec():
    violations = _write_and_scan({'test.py': "exec('import os')"})
    assert len(violations) >= 1


def test_ban_eval():
    violations = _write_and_scan({'test.py': "eval('1+1')"})
    assert len(violations) >= 1


def test_ban_gzip():
    violations = _write_and_scan({'test.py': 'import gzip'})
    assert len(violations) >= 1


def test_ban_bz2():
    violations = _write_and_scan({'test.py': 'import bz2'})
    assert len(violations) >= 1


def test_ban_ctypes():
    violations = _write_and_scan({'test.py': 'import ctypes'})
    assert len(violations) >= 1


def test_ban_urllib():
    violations = _write_and_scan({'test.py': 'from urllib.request import urlopen'})
    assert len(violations) >= 1


# --- Should be ALLOWED ---

def test_allow_numpy():
    violations = _write_and_scan({'test.py': 'import numpy as np'})
    assert len(violations) == 0


def test_allow_torch():
    violations = _write_and_scan({'test.py': 'import torch'})
    assert len(violations) == 0


def test_allow_torch_nn():
    violations = _write_and_scan({'test.py': 'import torch.nn as nn'})
    assert len(violations) == 0


def test_allow_torch_optim():
    violations = _write_and_scan({'test.py': 'from torch.optim import Adam'})
    assert len(violations) == 0


def test_allow_sys():
    violations = _write_and_scan({'test.py': 'import sys'})
    assert len(violations) == 0


def test_allow_io():
    violations = _write_and_scan({'test.py': 'import io'})
    assert len(violations) == 0


def test_allow_math():
    violations = _write_and_scan({'test.py': 'import math'})
    assert len(violations) == 0


def test_allow_collections():
    violations = _write_and_scan({'test.py': 'from collections import defaultdict'})
    assert len(violations) == 0


def test_allow_struct():
    violations = _write_and_scan({'test.py': 'import struct'})
    assert len(violations) == 0


def test_allow_arithmetic_coding():
    violations = _write_and_scan({'test.py': 'from arithmetic_coding import ArithmeticEncoder'})
    assert len(violations) == 0


def test_allow_local_imports():
    violations = _write_and_scan({
        'model.py': 'import torch',
        'compress.py': 'from model import Model',
    })
    assert len(violations) == 0


def test_allow_lib_prefix():
    violations = _write_and_scan({'test.py': 'from lib.arithmetic_coding import ArithmeticEncoder'})
    assert len(violations) == 0
