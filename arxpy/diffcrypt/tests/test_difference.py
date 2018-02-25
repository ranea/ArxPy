"""Tests for the Difference module."""
import doctest
import unittest

from hypothesis import given   # , reproduce_failure
from hypothesis.strategies import integers

from arxpy.bitvector.core import Constant
from arxpy.bitvector.operation import (
    BvNot, BvXor, RotateLeft, RotateRight, Extract, Concat)

from arxpy.diffcrypt import difference
from arxpy.diffcrypt.difference import DiffVar, XorDiff, RXDiff

MIN_SIZE = 2
MAX_SIZE = 32


class TestDifference(unittest.TestCase):
    """Tests for the Difference class."""

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_get_difference(self, width, x, y):
        for diff_type in [XorDiff, RXDiff]:
            bvx = Constant(x % (2 ** width), width)
            bvy = Constant(y % (2 ** width), width)
            bvd = diff_type.get_difference(bvx, bvy)
            self.assertEqual(bvy, diff_type.get_pair_element(bvx, bvd))

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_xor_deterministic_propagation(self, width, x1, y1, x2, y2):
        x1 = Constant(x1 % (2 ** width), width)
        y1 = Constant(y1 % (2 ** width), width)
        d1 = XorDiff.get_difference(x1, y1)
        x2 = Constant(x2 % (2 ** width), width)
        y2 = Constant(y2 % (2 ** width), width)
        d2 = XorDiff.get_difference(x2, y2)
        vd1 = DiffVar("in1", width)
        vd2 = DiffVar("in2", width)

        d3 = XorDiff.get_difference(~x1, ~y1)
        vd3 = XorDiff.propagate(BvNot, vd1)
        self.assertEqual(d3, vd3.xreplace({vd1: d1}))

        d3 = XorDiff.get_difference(x1 ^ x2, y1 ^ y2)
        vd3 = XorDiff.propagate(BvXor, [vd1, vd2])
        self.assertEqual(d3, vd3.xreplace({vd1: d1, vd2: d2}))

        d3 = XorDiff.get_difference(x1 ^ x2, y1 ^ x2)  # Xor with a constant
        vd3 = XorDiff.propagate(BvXor, [vd1, x2])
        self.assertEqual(d3, vd3.xreplace({vd1: d1}))

        r = int(x2) % x1.width
        d3 = XorDiff.get_difference(RotateLeft(x1, r), RotateLeft(y1, r))
        vd3 = XorDiff.propagate(RotateLeft, [vd1, r])
        self.assertEqual(d3, vd3.xreplace({vd1: d1}))

        d3 = XorDiff.get_difference(RotateRight(x1, r), RotateRight(y1, r))
        vd3 = XorDiff.propagate(RotateRight, [vd1, r])
        self.assertEqual(d3, vd3.xreplace({vd1: d1}))

        i = int(x2) % x1.width
        j = int(y2) % (i + 1)
        d3 = XorDiff.get_difference(x1[i:j], y1[i:j])
        vd3 = XorDiff.propagate(Extract, [vd1, i, j])
        self.assertEqual(d3, vd3.xreplace({vd1: d1}))

        d3 = XorDiff.get_difference(Concat(x1, x2), Concat(y1, y2))
        vd3 = XorDiff.propagate(Concat, [vd1, vd2])
        self.assertEqual(d3, vd3.xreplace({vd1: d1, vd2: d2}))

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_rx_deterministic_propagation(self, width, x1, y1, x2, y2):
        x1 = Constant(x1 % (2 ** width), width)
        y1 = Constant(y1 % (2 ** width), width)
        d1 = RXDiff.get_difference(x1, y1)
        x2 = Constant(x2 % (2 ** width), width)
        y2 = Constant(y2 % (2 ** width), width)
        d2 = RXDiff.get_difference(x2, y2)
        vd1 = DiffVar("in1", width)
        vd2 = DiffVar("in2", width)

        d3 = RXDiff.get_difference(~x1, ~y1)
        vd3 = RXDiff.propagate(BvNot, vd1)
        self.assertEqual(d3, vd3.xreplace({vd1: d1}))

        d3 = RXDiff.get_difference(x1 ^ x2, y1 ^ y2)
        vd3 = RXDiff.propagate(BvXor, [vd1, vd2])
        self.assertEqual(d3, vd3.xreplace({vd1: d1, vd2: d2}))

        d3 = RXDiff.get_difference(x1 ^ x2, y1 ^ x2)  # Xor with a constant
        vd3 = RXDiff.propagate(BvXor, [vd1, x2])
        self.assertEqual(d3, vd3.xreplace({vd1: d1}))

        r = int(x2) % x1.width
        d3 = RXDiff.get_difference(RotateLeft(x1, r), RotateLeft(y1, r))
        vd3 = RXDiff.propagate(RotateLeft, [vd1, r])
        self.assertEqual(d3, vd3.xreplace({vd1: d1}))

        d3 = RXDiff.get_difference(RotateRight(x1, r), RotateRight(y1, r))
        vd3 = RXDiff.propagate(RotateRight, [vd1, r])
        self.assertEqual(d3, vd3.xreplace({vd1: d1}))


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(difference))
    return tests
