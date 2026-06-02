"""Shared test fixtures."""

import os
import sys
import pytest

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib'))

SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'instances', 'sample')
BASELINE_DIR = os.path.join(PROJECT_ROOT, 'baseline')
SOLUTION_DIR = os.path.join(PROJECT_ROOT, 'solution_template')


@pytest.fixture
def sample_data():
    """Load sample_00.txt as bytes."""
    path = os.path.join(SAMPLE_DIR, 'sample_00.txt')
    with open(path, 'rb') as f:
        return f.read()


@pytest.fixture
def small_data():
    """A small test string for quick tests."""
    return b"The quick brown fox jumps over the lazy dog. " * 10
