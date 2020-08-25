"""Tests for the Difference module."""
import doctest
import unittest

from hypothesis import given   # , reproduce_failure
from hypothesis.strategies import integers

from arxpy.bitvector.core import Constant, Variable
from arxpy.bitvector.operation import (
    BvNot, BvXor, RotateLeft, RotateRight, BvShl, BvLshr, Extract, Concat,
    BvAnd)
from arxpy.bitvector.extraop import make_partial_operation

from arxpy.differential import difference
from arxpy.differential.difference import XorDiff, RXDiff

MIN_SIZE = 2
MAX_SIZE = 32


class TestDifference(unittest.TestCase):
    """Tests for the Difference class."""

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_from_pair(self, width, x, y):
        for diff_type in [XorDiff, RXDiff]:
            bvx = Constant(x % (2 ** width), width)
            bvy = Constant(y % (2 ** width), width)
            bvd = diff_type.from_pair(bvx, bvy)
            self.assertEqual(bvy, bvd.get_pair_element(bvx))

    # noinspection PyPep8Naming
    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_xor_linear_op(self, width, x1, y1, x2, y2):
        x1 = Constant(x1 % (2 ** width), width)
        y1 = Constant(y1 % (2 ** width), width)
        d1 = XorDiff.from_pair(x1, y1)
        x2 = Constant(x2 % (2 ** width), width)
        y2 = Constant(y2 % (2 ** width), width)
        d2 = XorDiff.from_pair(x2, y2)

        self.assertEqual(
            XorDiff.from_pair(~x1, ~y1),
            XorDiff.derivative(BvNot, d1))

        self.assertEqual(
            XorDiff.from_pair(x1 ^ x2, y1 ^ y2),
            XorDiff.derivative(BvXor, [d1, d2]))

        cte = x2
        BvXor_fix = make_partial_operation(BvXor, tuple([None, cte]))
        self.assertEqual(
            XorDiff.from_pair(x1 ^ cte, y1 ^ cte),
            XorDiff.derivative(BvXor_fix, [d1]))

        cte = Variable("c", width)
        BvXor_fix = make_partial_operation(BvXor, tuple([None, cte]))
        self.assertEqual(
            XorDiff.from_pair(x1 ^ cte, y1 ^ cte),
            XorDiff.derivative(BvXor, [d1, XorDiff.from_pair(cte, cte)]))
        self.assertEqual(
            XorDiff.from_pair(x1 ^ cte, y1 ^ cte),
            XorDiff.derivative(BvXor_fix, [d1]))

        cte = x2
        BvAnd_fix = make_partial_operation(BvAnd, tuple([None, cte]))
        self.assertEqual(
            XorDiff.from_pair(x1 & cte, y1 & cte),
            XorDiff.derivative(BvAnd_fix, [d1]))

        r = int(x2) % x1.width
        RotateLeft_fix = make_partial_operation(RotateLeft, tuple([None, r]))
        RotateRight_fix = make_partial_operation(RotateRight, tuple([None, r]))
        self.assertEqual(
            XorDiff.from_pair(RotateLeft(x1, r), RotateLeft(y1, r)),
            XorDiff.derivative(RotateLeft_fix, d1))
        self.assertEqual(
            XorDiff.from_pair(RotateRight(x1, r), RotateRight(y1, r)),
            XorDiff.derivative(RotateRight_fix, d1))

        r = Constant(int(x2) % x1.width, width)
        BvShl_fix = make_partial_operation(BvShl, tuple([None, r]))
        BvLshr_fix = make_partial_operation(BvLshr, tuple([None, r]))
        self.assertEqual(
            XorDiff.from_pair(x1 << r, y1 << r),
            XorDiff.derivative(BvShl_fix, d1))
        self.assertEqual(
            XorDiff.from_pair(x1 >> r, y1 >> r),
            XorDiff.derivative(BvLshr_fix, d1))

        i = int(x2) % x1.width
        j = int(y2) % (i + 1)
        Extract_fix = make_partial_operation(Extract, tuple([None, i, j]))
        self.assertEqual(
            XorDiff.from_pair(x1[i:j], y1[i:j]),
            XorDiff.derivative(Extract_fix, d1))

        self.assertEqual(
            XorDiff.from_pair(Concat(x1, x2), Concat(y1, y2)),
            XorDiff.derivative(Concat, [d1, d2]))

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_rx_linear_op(self, width, x1, y1, x2, y2):
        x1 = Constant(x1 % (2 ** width), width)
        y1 = Constant(y1 % (2 ** width), width)
        d1 = RXDiff.from_pair(x1, y1)
        x2 = Constant(x2 % (2 ** width), width)
        y2 = Constant(y2 % (2 ** width), width)
        d2 = RXDiff.from_pair(x2, y2)

        self.assertEqual(
            RXDiff.from_pair(~x1, ~y1),
            RXDiff.derivative(BvNot, d1))

        self.assertEqual(
            RXDiff.from_pair(x1 ^ x2, y1 ^ y2),
            RXDiff.derivative(BvXor, [d1, d2]))

        cte = x2
        BvXor_fix = make_partial_operation(BvXor, tuple([None, cte]))
        self.assertEqual(
            RXDiff.from_pair(x1 ^ cte, y1 ^ cte),
            RXDiff.derivative(BvXor_fix, [d1]))

        cte = Variable("c", width)
        BvXor_fix = make_partial_operation(BvShl, tuple([None, cte]))
        with self.assertRaises(ValueError):
            XorDiff.derivative(BvXor_fix, d1)

        r = int(x2) % x1.width
        RotateLeft_fix = make_partial_operation(RotateLeft, tuple([None, r]))
        RotateRight_fix = make_partial_operation(RotateRight, tuple([None, r]))
        self.assertEqual(
            RXDiff.from_pair(RotateLeft(x1, r), RotateLeft(y1, r)),
            RXDiff.derivative(RotateLeft_fix, d1))
        self.assertEqual(
            RXDiff.from_pair(RotateRight(x1, r), RotateRight(y1, r)),
            RXDiff.derivative(RotateRight_fix, d1))


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(difference))
    return tests
