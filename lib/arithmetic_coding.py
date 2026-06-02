"""
Integer arithmetic coding encoder/decoder.

Adapted from Nayuki's Reference Arithmetic Coding (MIT License).
https://github.com/nayuki/Reference-arithmetic-coding

This module provides the entropy coding infrastructure. The agent's job is to
build a prediction model that outputs good probability distributions — this
library converts those distributions into compressed bits.

Interface:
    encoder.write(freqs, symbol)  — encode one symbol given cumulative freqs
    decoder.read(freqs)           — decode one symbol given cumulative freqs

Cumulative frequency format:
    freqs is a list/array of length (num_symbols + 1) where:
        freqs[0] = 0
        freqs[i] = sum of frequencies for symbols 0..i-1
        freqs[num_symbols] = total
    Every symbol must have freq >= 1 (i.e. freqs[i+1] > freqs[i]).
"""

import io


# ─── Bit-level I/O ───────────────────────────────────────────────────────────

class BitOutputStream:
    """Writes individual bits to a byte stream."""

    def __init__(self, out: io.RawIOBase):
        self._out = out
        self._current_byte = 0
        self._num_bits_filled = 0

    def write(self, bit: int):
        if bit not in (0, 1):
            raise ValueError(f"Bit must be 0 or 1, got {bit}")
        self._current_byte = (self._current_byte << 1) | bit
        self._num_bits_filled += 1
        if self._num_bits_filled == 8:
            self._out.write(bytes([self._current_byte]))
            self._current_byte = 0
            self._num_bits_filled = 0

    def close(self):
        """Pad remaining bits with zeros and flush."""
        while self._num_bits_filled != 0:
            self.write(0)


class BitInputStream:
    """Reads individual bits from a byte stream."""

    def __init__(self, inp: io.RawIOBase):
        self._inp = inp
        self._current_byte = 0
        self._num_bits_remaining = 0

    def read(self) -> int:
        if self._num_bits_remaining == 0:
            data = self._inp.read(1)
            if len(data) == 0:
                return -1  # EOF
            self._current_byte = data[0]
            self._num_bits_remaining = 8
        self._num_bits_remaining -= 1
        return (self._current_byte >> self._num_bits_remaining) & 1

    def close(self):
        pass


# ─── Arithmetic Coder Base ───────────────────────────────────────────────────

# Use 32-bit precision for the arithmetic coder state.
_NUM_STATE_BITS = 32
_FULL_RANGE = 1 << _NUM_STATE_BITS        # 2^32
_HALF_RANGE = _FULL_RANGE >> 1             # 2^31
_QUARTER_RANGE = _HALF_RANGE >> 1          # 2^30
_MIN_RANGE = _QUARTER_RANGE + 2            # Minimum working range
_MAX_TOTAL = min(_MIN_RANGE, (1 << 30))    # Max sum of frequencies
_STATE_MASK = _FULL_RANGE - 1              # 0xFFFFFFFF


class _ArithmeticCoderBase:
    """Base class with shared state for encoder and decoder."""

    def __init__(self):
        self._low = 0
        self._high = _STATE_MASK  # _FULL_RANGE - 1

    def _update(self, freqs, symbol):
        """Update the range [low, high) for the given symbol."""
        low = self._low
        high = self._high
        rng = high - low + 1

        if rng < _MIN_RANGE:
            raise AssertionError(f"Range too small: {rng}")

        total = freqs[-1]  # freqs[num_symbols]
        if total > _MAX_TOTAL:
            raise ValueError(
                f"Total frequency {total} exceeds maximum {_MAX_TOTAL}. "
                "Scale down your frequency table."
            )

        sym_low = freqs[symbol]
        sym_high = freqs[symbol + 1]

        if sym_low >= sym_high:
            raise ValueError(
                f"Symbol {symbol} has zero frequency "
                f"(cumfreqs[{symbol}]={sym_low}, cumfreqs[{symbol+1}]={sym_high})"
            )

        # Narrow the range
        self._low = low + sym_low * rng // total
        self._high = low + sym_high * rng // total - 1


# ─── Encoder ─────────────────────────────────────────────────────────────────

class ArithmeticEncoder(_ArithmeticCoderBase):
    """Encodes symbols into a bit stream using arithmetic coding."""

    def __init__(self, bit_out: BitOutputStream):
        super().__init__()
        self._output = bit_out
        self._num_underflow = 0

    def write(self, freqs, symbol: int):
        """Encode a single symbol. freqs is the cumulative frequency table."""
        self._update(freqs, symbol)
        self._normalize()

    def finish(self):
        """Flush remaining state to the output stream."""
        self._output.write(1)
        self._output.close()

    def _normalize(self):
        while True:
            if self._high < _HALF_RANGE:
                # Top bit is 0 for both low and high
                self._emit_bit(0)
            elif self._low >= _HALF_RANGE:
                # Top bit is 1 for both low and high
                self._emit_bit(1)
                self._low -= _HALF_RANGE
                self._high -= _HALF_RANGE
            elif self._low >= _QUARTER_RANGE and self._high < 3 * _QUARTER_RANGE:
                # Underflow case
                self._num_underflow += 1
                self._low -= _QUARTER_RANGE
                self._high -= _QUARTER_RANGE
            else:
                break
            self._low = self._low << 1
            self._high = (self._high << 1) | 1

    def _emit_bit(self, bit: int):
        self._output.write(bit)
        # Flush underflow bits (opposite of the emitted bit)
        for _ in range(self._num_underflow):
            self._output.write(bit ^ 1)
        self._num_underflow = 0


# ─── Decoder ─────────────────────────────────────────────────────────────────

class ArithmeticDecoder(_ArithmeticCoderBase):
    """Decodes symbols from a bit stream using arithmetic coding."""

    def __init__(self, bit_in: BitInputStream):
        super().__init__()
        self._input = bit_in
        # Initialize code value from the first _NUM_STATE_BITS bits
        self._code = 0
        for _ in range(_NUM_STATE_BITS):
            self._code = (self._code << 1) | self._read_bit()

    def read(self, freqs) -> int:
        """Decode and return a single symbol. freqs is the cumulative freq table."""
        total = freqs[-1]
        if total > _MAX_TOTAL:
            raise ValueError(
                f"Total frequency {total} exceeds maximum {_MAX_TOTAL}."
            )

        rng = self._high - self._low + 1
        offset = self._code - self._low
        value = ((offset + 1) * total - 1) // rng

        # Binary search for the symbol
        lo, hi = 0, len(freqs) - 2  # symbols are 0..num_symbols-1
        while lo < hi:
            mid = (lo + hi) >> 1
            if freqs[mid + 1] <= value:
                lo = mid + 1
            else:
                hi = mid
        symbol = lo

        # Verify
        if freqs[symbol] > value or value >= freqs[symbol + 1]:
            raise AssertionError("Decoder state inconsistency")

        self._update(freqs, symbol)
        self._normalize()
        return symbol

    def _normalize(self):
        while True:
            if self._high < _HALF_RANGE:
                pass  # Top bit is 0
            elif self._low >= _HALF_RANGE:
                self._code -= _HALF_RANGE
                self._low -= _HALF_RANGE
                self._high -= _HALF_RANGE
            elif self._low >= _QUARTER_RANGE and self._high < 3 * _QUARTER_RANGE:
                self._code -= _QUARTER_RANGE
                self._low -= _QUARTER_RANGE
                self._high -= _QUARTER_RANGE
            else:
                break
            self._low = self._low << 1
            self._high = (self._high << 1) | 1
            self._code = (self._code << 1) | self._read_bit()

    def _read_bit(self) -> int:
        bit = self._input.read()
        if bit == -1:
            # Past end of stream — pad with zeros (standard convention)
            bit = 0
        return bit
