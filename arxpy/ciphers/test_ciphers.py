"""Tests for cryptographic primitives."""
import doctest
import math
import unittest

from hypothesis import given, settings, unlimited, HealthCheck  # , example
from hypothesis.strategies import integers

from arxpy.bitvector.core import Constant
from arxpy.bitvector.context import Validation, Simplification

from arxpy.diffcrypt.characteristic import CompositeCh
from arxpy.diffcrypt.difference import XorDiff, RXDiff, DiffVar
from arxpy.diffcrypt.itercipher import OptimalRelatedKeyCh
from arxpy.diffcrypt.smt import CompositeSmtProblem

from arxpy import ciphers
from arxpy.ciphers.simeck32_64 import Simeck32_64
from arxpy.ciphers.simeck48_96 import Simeck48_96
from arxpy.ciphers.simeck64_128 import Simeck64_128
from arxpy.ciphers.simon32_64 import Simon32_64
from arxpy.ciphers.simon48_72 import Simon48_72
from arxpy.ciphers.simon48_96 import Simon48_96
from arxpy.ciphers.simon64_128 import Simon64_128
from arxpy.ciphers.simon64_96 import Simon64_96
from arxpy.ciphers.speck32_64 import Speck32_64
from arxpy.ciphers.speck48_72 import Speck48_72
from arxpy.ciphers.speck48_96 import Speck48_96
from arxpy.ciphers.speck64_128 import Speck64_128
from arxpy.ciphers.speck64_96 import Speck64_96

block_ciphers = [
    Simeck32_64, Simeck48_96, Simeck64_128,
    Simon32_64, Simon48_72, Simon48_96, Simon64_96, Simon64_128,
    Speck32_64, Speck48_72, Speck48_96, Speck64_96, Speck64_128
]

MAX_INNER_WEIGHT = 8
MAX_ROUNDS = 8
VERBOSE = False
EXTRA_VERBOSE = False

DP_WIDTH = 16


class TestBlockCiphers(unittest.TestCase):
    """Test the block ciphers implemented."""

    def setUp(self):
        self.default_rounds = []
        for bc in block_ciphers:
            self.default_rounds.append(bc.rounds)

    def tearDown(self):
        for bc, default_rounds in zip(block_ciphers, self.default_rounds):
            bc.set_rounds(default_rounds)

    def test_eval(self):
        for bc, default_rounds in zip(block_ciphers, self.default_rounds):
            for rounds in [1, default_rounds // 2, default_rounds]:
                bc.set_rounds(rounds)

                zero_input = [0 for i in bc.input_widths]
                symbolic_input = bc._symbolic_input()

                bc(*zero_input)
                bc.symbolic_execution(*symbolic_input)

    @unittest.skip("heavy test")
    def test_automatic_search(self):
        for bc in block_ciphers:
            for diff_type in [XorDiff, RXDiff]:
                OptimalRelatedKeyCh(bc, diff_type, end=MAX_ROUNDS)

    @unittest.skip("heavy test")
    def test_manual_search(self):
        for bc, default_rounds in zip(block_ciphers, self.default_rounds):
            for diff_type in [XorDiff, RXDiff]:
                for rounds in range(1, MAX_ROUNDS):
                    bc.set_rounds(rounds)

                    input_diff = bc._symbolic_input("p", "k")
                    input_diff = [DiffVar.from_Variable(d) for d in input_diff]

                    ch = CompositeCh(bc, diff_type, input_diff)

                    for outer_target_weight in range(0, MAX_INNER_WEIGHT):
                        target_weight = [MAX_INNER_WEIGHT, outer_target_weight]

                        smt_problem = CompositeSmtProblem(ch, target_weight)

                        msg = (
                            "\nSearching optimal {} characteristic of "
                            "{}-round {} with target weight {}"
                            "\nFormula sizes: {}, {} \n{} \n{} \n{}"
                        ).format(diff_type.__name__, rounds, bc.__name__,
                                 target_weight, smt_problem.formula_size,
                                 smt_problem.pysmt_formula_size,
                                 ch.inner_ch, ch.outer_ch, smt_problem)
                        if EXTRA_VERBOSE:
                            print(msg)

                        if not smt_problem.solve():
                            continue

                        inner_assig, outer_assig = smt_problem.solve(get_assignment=True)

                        for assig, iter_ch in zip(
                            [inner_assig, outer_assig],
                            [ch.inner_ch, ch.outer_ch],
                        ):
                            tw = assig["weight"]
                            iter_func = iter_ch.func
                            error = sum(iter_func.input_widths) * 0.10
                            ew = iter_ch.empirical_weight(
                                list(assig["differences"].values()), False, tw)

                            msg = (
                                "{}-round {}-{} {} ch. with target weight {} "
                                "has (emp_weight, theo_weight) = ({:.2f}, {}) "
                            ).format(iter_func.rounds, iter_func.__name__,
                                     bc.__name__, diff_type.__name__,
                                     target_weight, ew, tw)

                            if VERBOSE:
                                print(msg)

                            if EXTRA_VERBOSE:
                                print(assig)

                            self.assertGreaterEqual(ew, tw - error, msg=msg)
                            self.assertLessEqual(ew, tw + error, msg=msg)

                        break

            bc.set_rounds(default_rounds)


@unittest.skip("heavy test")
class TestDifferentialF(unittest.TestCase):
    """Tests for the differential of the F function of Simon/Simeck."""

    def _find_correct_pair(self, differential_op, din, dout):
        input_diff = differential_op.input_diff
        assert len(input_diff) == 1
        assert all(isinstance(d, DiffVar) for d in input_diff)

        width = din.width
        diff_type = differential_op.diff_type
        op = differential_op.op

        for i in range(2 ** width):
            x = Constant(i, width)
            input_pair = (x, diff_type.get_pair_element(x, din))
            output_pair = op(input_pair[0]), op(input_pair[1])
            if diff_type.get_difference(*output_pair) == dout:
                return input_pair, output_pair
        else:
            return None

    def _count_correct_pairs(self, differential_op, din, dout):
        input_diff = differential_op.input_diff
        assert len(input_diff) == 1
        assert all(isinstance(d, DiffVar) for d in input_diff)

        width = din.width
        diff_type = differential_op.diff_type
        op = differential_op.op
        correct_pairs = 0
        total_pairs = 0

        for i in range(2 ** width):
            total_pairs += 1
            x = Constant(i, width)
            input_pair = (x, diff_type.get_pair_element(x, din))
            output_pair = op(input_pair[0]), op(input_pair[1])
            if diff_type.get_difference(*output_pair) == dout:
                correct_pairs += 1

        return correct_pairs, total_pairs

    @given(
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
    )
    @settings(timeout=unlimited, suppress_health_check=[HealthCheck.hung_test])
    def test_f(self, din, dout):
        width = DP_WIDTH
        din, dout = Constant(din, width), Constant(dout, width)
        sin, sout = DiffVar("sin", width), DiffVar("sout", width)

        for differential_type in [
            ciphers.simon32_64.F.differential(XorDiff),
            ciphers.simon32_64.F.differential(RXDiff),
            ciphers.simeck32_64.F.differential(XorDiff),
            ciphers.simeck32_64.F.differential(RXDiff)
        ]:
            dbvadd = differential_type(sin, sout)
            is_valid = dbvadd.is_valid().xreplace({sin: din, sout: dout})

            print("{}({} -> {})".format(differential_type.__name__, din, dout))

            if is_valid:
                with Validation(False), Simplification(False):
                    correct, total = self._count_correct_pairs(dbvadd, din, dout)

                msg = (
                    "{}({} -> {}) is VALID but no correct pairs were found"
                ).format(differential_type.__name__, din, dout)

                self.assertNotEqual(correct, 0, msg=msg)

                emp_weight = - math.log(correct / total, 2)
                emp_weight = dbvadd.weight_function(emp_weight)

                w = dbvadd.weight()
                theo_weight = int(w.xreplace({sin: din, sout: dout}))
                max_weight = 2 ** (w.width) - 1

                error = max_weight * 0.10

                msg = (
                    "{}({} -> {})  has theoretical weight {} "
                    "but empirical weight {:.2f}"
                ).format(differential_type.__name__, din, dout,
                         theo_weight, emp_weight)

                self.assertGreaterEqual(emp_weight, theo_weight - error, msg=msg)
                self.assertLessEqual(emp_weight, theo_weight + error, msg=msg)
            else:
                with Validation(False), Simplification(False):
                    pairs = self._find_correct_pair(dbvadd, din, dout)

                msg = (
                    "{}({} -> {}) is INVALID but the correct pair {} was found"
                ).format(differential_type.__name__, din, dout, pairs)

                self.assertIsNone(pairs, msg=msg)


def load_tests(loader, tests, ignore):
    """Load doctests."""
    tests.addTests(doctest.DocTestSuite(ciphers.simeck32_64))
    tests.addTests(doctest.DocTestSuite(ciphers.simeck48_96))
    tests.addTests(doctest.DocTestSuite(ciphers.simeck64_128))
    tests.addTests(doctest.DocTestSuite(ciphers.simon32_64))
    tests.addTests(doctest.DocTestSuite(ciphers.simon48_72))
    tests.addTests(doctest.DocTestSuite(ciphers.simon48_96))
    tests.addTests(doctest.DocTestSuite(ciphers.simon64_96))
    tests.addTests(doctest.DocTestSuite(ciphers.simon64_128))
    tests.addTests(doctest.DocTestSuite(ciphers.speck32_64))
    tests.addTests(doctest.DocTestSuite(ciphers.speck48_72))
    tests.addTests(doctest.DocTestSuite(ciphers.speck48_96))
    tests.addTests(doctest.DocTestSuite(ciphers.speck64_96))
    tests.addTests(doctest.DocTestSuite(ciphers.speck64_128))
    return tests
