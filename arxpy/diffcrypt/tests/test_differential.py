"""Tests for the Difference module."""
import doctest
import itertools
import math
import unittest

from hypothesis import given, settings, unlimited, HealthCheck, example
from hypothesis.strategies import integers

from arxpy.bitvector.core import Constant
from arxpy.bitvector.context import Validation, Simplification

from arxpy.diffcrypt.difference import DiffVar
from arxpy.diffcrypt import differential
from arxpy.diffcrypt.differential import XDBvAdd, RXDBvAdd


DP_WIDTH = 8


@unittest.skip("heavy test")
class TestDifferential(unittest.TestCase):
    """Tests for the Difference class."""

    def _find_correct_pair(self, differential_op, d1, d2, d3):
        input_diff = differential_op.input_diff
        assert len(input_diff) == 2
        assert all(isinstance(d, DiffVar) for d in input_diff)

        width = d1.width
        diff_type = differential_op.diff_type
        op = differential_op.op

        for i, j in itertools.product(range(2 ** width), range(2 ** width)):
            x = Constant(i, width)
            y = Constant(j, width)
            pair1 = (x, diff_type.get_pair_element(x, d1))
            pair2 = (y, diff_type.get_pair_element(y, d2))
            pair3 = op(pair1[0], pair2[0]), op(pair1[1], pair2[1])
            if diff_type.get_difference(*pair3) == d3:
                return pair1, pair2, pair3
        else:
            return None

    def _count_correct_pairs(self, differential_op, d1, d2, d3):
        input_diff = differential_op.input_diff
        assert len(input_diff) == 2
        assert all(isinstance(d, DiffVar) for d in input_diff)

        width = d1.width
        diff_type = differential_op.diff_type
        op = differential_op.op
        correct_pairs = 0
        total_pairs = 0

        for i, j in itertools.product(range(2 ** width), range(2 ** width)):
            total_pairs += 1
            x = Constant(i, width)
            y = Constant(j, width)
            pair1 = (x, diff_type.get_pair_element(x, d1))
            pair2 = (y, diff_type.get_pair_element(y, d2))
            pair3 = op(pair1[0], pair2[0]), op(pair1[1], pair2[1])
            if diff_type.get_difference(*pair3) == d3:
                correct_pairs += 1

        return correct_pairs, total_pairs

    @given(
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
    )
    @settings(timeout=unlimited, suppress_health_check=[HealthCheck.hung_test])
    @example(d1=0x61, d2=0x4e, d3=0xa0)
    @example(d1=0x00, d2=0x01, d3=0x3e)
    @example(d1=0x00, d2=0x00, d3=0x00)
    @example(d1=0x09, d2=0xc4, d3=0xb2)
    @example(d1=0x00, d2=0xc2, d3=0x4c)
    def test_bvadd(self, d1, d2, d3):
        width = DP_WIDTH
        d1, d2, d3 = Constant(d1, width), Constant(d2, width), Constant(d3, width)
        s1, s2, s3 = DiffVar("s1", width), DiffVar("s2", width), DiffVar("s3", width)

        # TODO: perform exhaustive test
        for differential_type in [RXDBvAdd]:  # XDBvAdd
            dbvadd = differential_type([s1, s2], s3)
            is_valid = dbvadd.is_valid().xreplace({s1: d1, s2: d2, s3: d3})

            print("{}({},{} -> {})".format(differential_type.__name__, d1, d2, d3))

            if is_valid:
                with Validation(False), Simplification(False):
                    correct, total = self._count_correct_pairs(dbvadd, d1, d2, d3)

                msg = (
                    "{}({},{} -> {}) is VALID but no correct pairs were found"
                ).format(differential_type.__name__, d1, d2, d3)

                self.assertNotEqual(correct, 0, msg=msg)

                emp_weight = - math.log(correct / total, 2)
                emp_weight = dbvadd.weight_function(emp_weight)

                w = dbvadd.weight()
                theo_weight = int(w.xreplace({s1: d1, s2: d2, s3: d3}))
                max_weight = 2 ** (w.width) - 1

                error = max_weight * 0.10

                msg = (
                    "{}({},{} -> {})  has theoretical weight {} "
                    "but empirical weight {:.2f}"
                ).format(differential_type.__name__, d1, d2, d3,
                         theo_weight, emp_weight)

                self.assertGreaterEqual(emp_weight, theo_weight - error, msg=msg)
                self.assertLessEqual(emp_weight, theo_weight + error, msg=msg)
            else:
                with Validation(False), Simplification(False):
                    pairs = self._find_correct_pair(dbvadd, d1, d2, d3)

                msg = (
                    "{}({},{} -> {}) is INVALID but the correct pair {} was found"
                ).format(differential_type.__name__, d1, d2, d3, pairs)

                self.assertIsNone(pairs, msg=msg)


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(differential))
    return tests
