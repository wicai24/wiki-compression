"""Tests for the default baseline (PPM) compressor."""

import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'solution_template'))

from arithmetic_coding import ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream
from utils import probs_to_cumfreqs, write_header, read_header


def _compress_decompress(data):
    """Full roundtrip: compress then decompress."""
    from model import Model

    # Compress
    model_c = Model()
    buf = io.BytesIO()
    write_header(buf, len(data))
    bit_out = BitOutputStream(buf)
    encoder = ArithmeticEncoder(bit_out)
    for byte_val in data:
        probs = model_c.predict()
        cumfreqs = probs_to_cumfreqs(probs)
        encoder.write(cumfreqs, byte_val)
        model_c.observe(byte_val)
    encoder.finish()
    compressed = buf.getvalue()

    # Decompress
    model_d = Model()
    buf2 = io.BytesIO(compressed)
    orig_len = read_header(buf2)
    bit_in = BitInputStream(buf2)
    decoder = ArithmeticDecoder(bit_in)
    result = bytearray()
    for i in range(orig_len):
        probs = model_d.predict()
        cumfreqs = probs_to_cumfreqs(probs)
        byte_val = decoder.read(cumfreqs)
        result.append(byte_val)
        model_d.observe(byte_val)

    return compressed, bytes(result)


def test_roundtrip_small():
    """Roundtrip on a small string."""
    data = b"Hello, World!"
    compressed, restored = _compress_decompress(data)
    assert restored == data


def test_roundtrip_repeated():
    """Roundtrip on repeated text — should compress well eventually."""
    data = b"abcabc" * 100
    compressed, restored = _compress_decompress(data)
    assert restored == data


def test_roundtrip_sample(sample_data):
    """Roundtrip on sample_00.txt."""
    compressed, restored = _compress_decompress(sample_data)
    assert restored == sample_data


def test_compression_ratio(sample_data):
    """Baseline should produce SOME compression on sufficient data."""
    compressed, restored = _compress_decompress(sample_data)
    # Archive itself should be smaller than original (even if code+archive > original)
    assert len(compressed) < len(sample_data) * 1.1  # at most 10% expansion


def test_model_determinism():
    """Two fresh models should produce identical predictions."""
    from model import Model
    import numpy as np

    model1 = Model()
    model2 = Model()

    # Same predictions from fresh state
    p1 = model1.predict()
    p2 = model2.predict()
    assert np.allclose(p1, p2)

    # Same predictions after observing same byte
    model1.observe(65)  # 'A'
    model2.observe(65)
    p1 = model1.predict()
    p2 = model2.predict()
    assert np.allclose(p1, p2)


def test_model_predictions_valid():
    """Model predictions should be valid probability distributions."""
    from model import Model
    import numpy as np

    model = Model()
    for byte_val in b"test data":
        probs = model.predict()
        assert probs.shape == (256,)
        assert np.all(probs >= 0)
        assert abs(probs.sum() - 1.0) < 1e-5
        model.observe(byte_val)
