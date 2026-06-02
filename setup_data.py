#!/usr/bin/env python3
"""
Download enwik8 and extract chunks for the compression task.

Usage:
    python setup_data.py

Downloads enwik8 (~36MB compressed, 100MB uncompressed) from Matt Mahoney's
Large Text Compression Benchmark, then extracts deterministic chunks into
instances/dev/, instances/train/, and instances/hidden/.
"""

import hashlib
import os
import sys
import urllib.request
import zipfile

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"
ENWIK8_SHA256 = "3fa39b759ef1e5fac4a230f75a1b3cf1dd61e4e2ad2f0e80218cc8a8d2576da3"
ENWIK8_SIZE = 100_000_000  # exactly 10^8 bytes

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "instances")

# Chunk extraction offsets — chosen to sample diverse parts of enwik8
# (different Wikipedia articles, markup styles, etc.)
CHUNK_SPECS = {
    "dev": [
        # 2 × 200KB chunks — same size as train for representative iteration
        ("dev_00.txt", 1_000_000, 200_000),
        ("dev_01.txt", 50_000_000, 200_000),
    ],
    "train": [
        # 3 × 200KB chunks for scoring
        # At 200KB, long-range patterns matter and simple n-grams become sparse.
        ("train_00.txt", 5_000_000, 200_000),
        ("train_01.txt", 35_000_000, 200_000),
        ("train_02.txt", 70_000_000, 200_000),
    ],
    "hidden": [
        # 3 × 200KB chunks for final evaluation
        ("hidden_00.txt", 15_000_000, 200_000),
        ("hidden_01.txt", 45_000_000, 200_000),
        ("hidden_02.txt", 85_000_000, 200_000),
    ],
}


def download_enwik8(dest_path):
    """Download enwik8.zip and extract enwik8."""
    zip_path = dest_path + ".zip"

    if os.path.exists(dest_path):
        print(f"  enwik8 already exists at {dest_path}")
        return

    print(f"  Downloading {ENWIK8_URL} ...")
    urllib.request.urlretrieve(ENWIK8_URL, zip_path)

    print(f"  Extracting enwik8 ...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extract("enwik8", os.path.dirname(dest_path))

    os.remove(zip_path)
    print(f"  Saved to {dest_path}")


def verify_enwik8(path):
    """Verify enwik8 file integrity."""
    print(f"  Verifying SHA256 ...")
    sha = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            sha.update(chunk)
    digest = sha.hexdigest()
    if digest != ENWIK8_SHA256:
        print(f"  WARNING: SHA256 mismatch!")
        print(f"  Expected: {ENWIK8_SHA256}")
        print(f"  Got:      {digest}")
        print(f"  The file may be corrupted or a different version.")
        # Don't abort — the file might still work
    else:
        print(f"  SHA256 OK")


def extract_chunks(enwik8_path):
    """Extract chunks from enwik8 at specified offsets."""
    with open(enwik8_path, 'rb') as f:
        data = f.read()

    assert len(data) == ENWIK8_SIZE, f"Expected {ENWIK8_SIZE} bytes, got {len(data)}"

    for split_name, specs in CHUNK_SPECS.items():
        split_dir = os.path.join(DATA_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for fname, offset, size in specs:
            chunk = data[offset:offset + size]
            chunk_path = os.path.join(split_dir, fname)
            with open(chunk_path, 'wb') as f:
                f.write(chunk)
            print(f"  {split_name}/{fname}: {len(chunk)} bytes (offset {offset})")


def main():
    enwik8_path = os.path.join(SCRIPT_DIR, "enwik8")

    print("Step 1: Download enwik8")
    download_enwik8(enwik8_path)

    print("\nStep 2: Verify integrity")
    verify_enwik8(enwik8_path)

    print("\nStep 3: Extract chunks")
    extract_chunks(enwik8_path)

    print("\nDone! Data is ready in instances/dev/, instances/train/, instances/hidden/")
    print("You can now run: bash run_evaluator.sh --mode dev")


if __name__ == "__main__":
    main()
