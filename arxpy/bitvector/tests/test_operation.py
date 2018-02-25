"""Tests for the expr module."""
import doctest
import unittest

from hypothesis import given
from hypothesis.strategies import integers

from arxpy.bitvector.core import Constant, Variable, bitvectify
from arxpy.bitvector.operation import (
    BvAnd, BvOr, BvXor, BvComp, BvUlt, BvUle, BvUgt, BvUge, BvShl,
    BvLshr, RotateLeft, RotateRight, Concat, BvAdd, BvSub, BvMul, BvUdiv,
    BvUrem, BvNeg, BvNot, Extract, ZeroExtend, Repeat, Ite
)

MIN_SIZE = 2
MAX_SIZE = 32


simple_op = set([
    BvAnd, BvOr, BvXor, BvComp, BvUlt, BvUle, BvUgt, BvUge, BvShl,
    BvLshr, BvAdd, BvSub, BvMul, BvUdiv, BvUrem])
unary_op = set([BvNeg, BvNot])
others = set([RotateLeft, RotateRight, Concat, Extract, ZeroExtend, Repeat, Ite])


class TestOperation(unittest.TestCase):
    """Test for the Operation class and subclasses."""

    def test_initialization(self):
        x = Variable("x", 8)
        y = Variable("y", 8)
        b = Variable("b", 1)

        for op in simple_op:
            expr = op(x, y)
            self.assertEqual(expr, bitvectify(expr, op.output_width(x, y)))
            self.assertFalse(expr.is_Atom)
            self.assertEqual(expr.atoms(), {x, y})

        for op in unary_op:
            expr = op(x)
            self.assertEqual(expr, bitvectify(expr, op.output_width(x)))
            self.assertFalse(expr.is_Atom)
            self.assertEqual(expr.atoms(), {x})

        for op in [RotateLeft, RotateRight, ZeroExtend, Repeat]:
            expr = op(x, 2)
            self.assertEqual(expr, bitvectify(expr, op.output_width(x, 2)))
            self.assertFalse(expr.is_Atom)

        expr = Concat(x, y)
        self.assertEqual(expr, bitvectify(expr, Concat.output_width(x, y)))
        self.assertFalse(expr.is_Atom)
        self.assertEqual(expr.atoms(), {x, y})

        expr = Extract(x, 4, 2)
        self.assertEqual(expr, bitvectify(expr, Extract.output_width(x, 4, 2)))
        self.assertFalse(expr.is_Atom)
        self.assertEqual(expr.atoms(), {x})

        expr = Ite(b, x, y)
        self.assertEqual(expr, bitvectify(expr, Ite.output_width(b, x, y)))
        self.assertFalse(expr.is_Atom)
        self.assertEqual(expr.atoms(), {b, x, y})

    def test_commutativity(self):
        x = Variable("x", 8)
        y = Variable("y", 8)

        for op in simple_op:
            if getattr(op, "is_symmetric", False):
                self.assertEqual(op(x, y), op(y, x))
            else:
                self.assertNotEqual(op(x, y), op(y, x))

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_constant_comparisons(self, width, x, y):
        modulus = 2 ** width
        x, y = x % modulus, y % modulus
        bvx = Constant(x, width)
        bvy = Constant(y, width)

        self.assertEqual((x == y), (bvx == bvy))
        self.assertEqual((bvx == bvy), (bvx == y))

        self.assertEqual((x != y), (bvx != bvy))
        self.assertEqual((bvx != bvy), (bvx != y))

        self.assertEqual((x < y), (bvx < bvy))
        self.assertEqual((bvx < bvy), (bvx < y))

        self.assertEqual((x <= y), (bvx <= bvy))
        self.assertEqual((bvx <= bvy), (bvx <= y))

        self.assertEqual((x > y), (bvx > bvy))
        self.assertEqual((bvx > bvy), (bvx > y))

        self.assertEqual((x >= y), (bvx >= bvy))
        self.assertEqual((bvx >= bvy), (bvx >= y))

    @given(
        integers(min_value=MIN_SIZE, max_value=MIN_SIZE),
        integers(min_value=0),
        integers(min_value=0),
    )
    def test_pysmt_operations(self, width, x, y):
        try:
            from pysmt import shortcuts as sc
        except ImportError:
            return

        modulus = 2 ** width
        x = x % modulus
        y = y % modulus
        bvx = Constant(x, width)
        bvy = Constant(y, width)
        psx = sc.BV(x, width)
        psy = sc.BV(y, width)

        def eval_pysmt(pysmt_var):
            return pysmt_var.simplify().constant_value()

        self.assertEqual(~bvx, eval_pysmt(sc.BVNot(psx)))
        self.assertEqual(bvx & bvy, eval_pysmt(sc.BVAnd(psx, psy)))
        self.assertEqual(bvx | bvy, eval_pysmt(sc.BVOr(psx, psy)))
        self.assertEqual(bvx ^ bvy, eval_pysmt(sc.BVXor(psx, psy)))

        self.assertEqual(BvComp(bvx, bvy), eval_pysmt(sc.BVComp(psx, psy)))
        self.assertEqual((bvx < bvy), eval_pysmt(sc.BVULT(psx, psy)))
        self.assertEqual((bvx <= bvy), eval_pysmt(sc.BVULE(psx, psy)))
        self.assertEqual((bvx > bvy), eval_pysmt(sc.BVUGT(psx, psy)))
        self.assertEqual((bvx >= bvy), eval_pysmt(sc.BVUGE(psx, psy)))

        r = y % bvx.width
        self.assertEqual(bvx << bvy, eval_pysmt(sc.BVLShl(psx, psy)))
        self.assertEqual(bvx >> bvy, eval_pysmt(sc.BVLShr(psx, psy)))
        self.assertEqual(RotateLeft(bvx, r), eval_pysmt(sc.BVRol(psx, r)))
        self.assertEqual(RotateRight(bvx, r), eval_pysmt(sc.BVRor(psx, r)))

        bvb = Constant(y % 2, 1)
        psb = sc.Bool(bool(bvb))
        self.assertEqual(Ite(bvb, bvx, bvy), eval_pysmt(sc.Ite(psb, psx, psy)))
        j = y % bvx.width
        self.assertEqual(bvx[:j], eval_pysmt(sc.BVExtract(psx, start=j)))
        self.assertEqual(bvx[j:], eval_pysmt(sc.BVExtract(psx, end=j)))
        self.assertEqual(Concat(bvx, bvy), eval_pysmt(sc.BVConcat(psx, psy)))
        self.assertEqual(ZeroExtend(bvx, j), eval_pysmt(sc.BVZExt(psx, j)))
        self.assertEqual(Repeat(bvx, 1 + j), eval_pysmt(psx.BVRepeat(1 + j)))

        self.assertEqual(-bvx, eval_pysmt(sc.BVNeg(psx)))
        self.assertEqual(bvx + bvy, eval_pysmt(sc.BVAdd(psx, psy)))
        self.assertEqual(bvx - bvy, eval_pysmt(sc.BVSub(psx, psy)))
        self.assertEqual(bvx * bvy, eval_pysmt(sc.BVMul(psx, psy)))
        if bvy > 0:
            self.assertEqual(bvx / bvy, eval_pysmt(sc.BVUDiv(psx, psy)))
            self.assertEqual(bvx % bvy, eval_pysmt(sc.BVURem(psx, psy)))

    def test_invalid_operations(self):
        x = Variable("x", 8)
        y = Variable("y", 9)
        b = Variable("b", 1)
        max_value = (2 ** 8) - 1

        with self.assertRaises(TypeError):
            x ** 2
        with self.assertRaises(TypeError):
            x // 2
        with self.assertRaises(TypeError):
            abs(x)

        for op in simple_op:
            with self.assertRaises((AssertionError, TypeError)):
                op()
            with self.assertRaises((AssertionError, TypeError)):
                op(x, -1)  # invalid range
            with self.assertRaises((AssertionError, TypeError)):
                op(x, max_value + 1)  # invalid range
            with self.assertRaises((AssertionError, TypeError)):
                op(2, 3)  # at least 1 Term
            with self.assertRaises((AssertionError, TypeError)):
                op(x, y)  # != width
            with self.assertRaises((AssertionError, TypeError)):
                op(x)  # invalid # of args
            with self.assertRaises((AssertionError, TypeError)):
                op(x, x, x)

        for op in unary_op:
            with self.assertRaises((AssertionError, TypeError)):
                op()
            with self.assertRaises((AssertionError, TypeError)):
                op(-1)
            with self.assertRaises((AssertionError, TypeError)):
                op(max_value + 1)
            with self.assertRaises((AssertionError, TypeError)):
                op(1)
            with self.assertRaises((AssertionError, TypeError)):
                op(x, x)
            with self.assertRaises((AssertionError, TypeError)):
                op(x, x, x)

        for op in [ZeroExtend, Repeat]:
            with self.assertRaises((AssertionError, TypeError)):
                op()
            with self.assertRaises((AssertionError, TypeError)):
                op(x, -1)

        with self.assertRaises((AssertionError, TypeError)):
            Concat(x, -1)
        with self.assertRaises((AssertionError, TypeError)):
            Concat(x, 1)

        with self.assertRaises((AssertionError, TypeError)):
            Extract(x, 0, 1)
        with self.assertRaises((AssertionError, TypeError)):
            Extract(x, x, 1)

        with self.assertRaises((AssertionError, TypeError)):
            Ite(b, x, y)
        with self.assertRaises((AssertionError, TypeError)):
            Ite(x, x, x)
        with self.assertRaises((AssertionError, TypeError)):
            Ite(0, x, x)

    def _simple_properties(self, x):
        width = x.width

        allones = BvNot(Constant(0, width))

        assert ~~x == x
        # assert ~(x & y) == (~x) | (~y)
        # assert ~(x | y) == (~x) & (~y)

        assert x & 0 == 0 & x == 0
        assert x & allones == allones & x == x
        assert x & x == x
        assert x & (~x) == 0

        assert x | 0 == 0 | x == x
        assert x | allones == allones | x == allones
        assert x | x == x
        assert x | (~x) == allones

        assert x ^ 0 == 0 ^ x == x
        assert x ^ allones == allones ^ x == ~x
        assert x ^ x == 0
        assert x ^ (~x) == allones

        assert x << 0 == x >> 0 == x
        assert 0 << x == 0 >> x == 0
        if isinstance(x, Constant):
            r = min(2 * int(x), x.width)
            assert (x << x) << x == x << r
            assert (x >> x) >> x == x >> r
        elif isinstance(x, Variable) and x.width >= 2:
            assert (x << 1) << 1 == x << 2
            assert (x >> 1) >> 1 == x >> 2

        n = x.width
        assert RotateLeft(x, 0) == RotateRight(x, 0) == x
        if x.width > 2:
            assert RotateLeft(RotateLeft(x, 1), 1) == RotateLeft(x, 2)
            assert RotateRight(RotateRight(x, 1), 1) == RotateRight(x, 2)
        if x.width > 3:
            assert RotateLeft(RotateRight(x, 1), 2) == RotateRight(x, n - 1)
            assert RotateRight(RotateLeft(x, 1), 2) == RotateLeft(x, n - 1)

        if isinstance(x, Constant):
            i = int(x) % (width - 1)
        elif isinstance(x, Variable) and x.width >= 2:
            i = width - 2
        n = x.width
        assert x[:] == x
        assert x[:i][1:0] == x[i + 1:i]
        assert Concat(x, x)[n - 1:i] == x[:i]  # n - 1 <= x.width - 1
        assert Concat(x, x)[n + i:n] == x[i:]  # n >= x.wdith
        assert (x << i)[:i] == x[n - 1 - i:]  # i <= i
        assert RotateLeft(x, i)[:i + 1] == x[n - 1 - i: 1]  # i <= i + 1
        assert (x >> i)[n - i - 1:] == x[n - 1:i]  # n - i - 1 < n - i
        assert RotateRight(x, i)[n - i - 1:] == x[n - 1:i]
        # assert (x & y)[0] == x[0] & y[0]

        if isinstance(x, Constant):
            i = int(x) % (width - 1)
        else:
            assert x.width >= 2
            i = width - 2
        assert Concat(x[:i + 1], x[i:]) == x

        assert -(-x) == x
        # assert -(x + y) == -(x) + -(y)
        # assert -(x * y) == -(x) * y
        # assert -(x / y) == -(x) / y
        # assert -(x % y) == -(x) % y
        # assert -(x ^ y) == BvNot(x ^ y, evaluate=False)

        assert x + 0 == 0 + x == x
        assert x + (-x) == 0

        assert x - 0 == x
        assert 0 - x == -x
        assert x - x == 0

        assert x * 0 == 0 * x == 0
        assert x * 1 == 1 * x == x

        if x != 0:
            assert x / x == 1
            assert 0 / x == 0
            assert x / 1 == x

            assert x % x == 0 % x == x % 1 == 0

    @given(
        integers(min_value=MIN_SIZE, max_value=MAX_SIZE),
        integers(min_value=0),
    )
    def test_constant_properties(self, width, x):
        self._simple_properties(Constant(x % (2 ** width), width))

    def test_variable_properties(self):
        self._simple_properties(Variable("x", 8))

    def test_python_operators(self):
        x = Variable("x", 8)
        y = Variable("y", 8)

        self.assertEqual(BvNot(x), ~x)

        expr = x
        expr &= y
        self.assertEqual(BvAnd(x, y), x & y)
        self.assertEqual(x & y, expr)

        expr = x
        expr |= y
        self.assertEqual(BvOr(x, y), x | y)
        self.assertEqual(x | y, expr)

        expr = x
        expr ^= y
        self.assertEqual(BvXor(x, y), x ^ y)
        self.assertEqual(x ^ y, expr)

        self.assertEqual(BvUlt(x, y), (x < y))

        self.assertEqual(BvUle(x, y), (x <= y))

        self.assertEqual(BvUgt(x, y), (x > y))

        self.assertEqual(BvUge(x, y), (x >= y))

        expr = x
        expr <<= y
        self.assertEqual(BvShl(x, y), x << y)
        self.assertEqual(x << y, expr)

        expr = x
        expr >>= y
        self.assertEqual(BvLshr(x, y), x >> y)
        self.assertEqual(x >> y, expr)

        self.assertEqual(Extract(x, 4, 2), x[4:2])
        self.assertEqual(Extract(x, 4, 4), x[4:4])
        self.assertEqual(Extract(x, 7, 1), x[:1])
        self.assertEqual(Extract(x, 6, 0), x[6:])

        self.assertEqual(BvNeg(x), -x)

        expr = x
        expr += y
        self.assertEqual(BvAdd(x, y), x + y)
        self.assertEqual(x + y, expr)

        expr = x
        expr -= y
        self.assertEqual(BvSub(x, y), x - y)
        self.assertEqual(x - y, expr)

        expr = x
        expr *= y
        self.assertEqual(BvMul(x, y), x * y)
        self.assertEqual(x * y, expr)

        expr = x
        expr /= y
        self.assertEqual(BvUdiv(x, y), x / y)
        self.assertEqual(x / y, expr)

        expr = x
        expr %= y
        self.assertEqual(BvUrem(x, y), x % y)
        self.assertEqual(x % y, expr)

    def test_basic_simplify(self):
        x, y, z = Variable("x", 8), Variable("y", 8), Variable("z", 8)

        allones = Constant((2 ** 8) - 1, 8)
        notone = ~Constant(1, 8)

        # constants

        self.assertEqual(1 + (x + allones), x)
        self.assertEqual(1 ^ (x ^ 1), x)
        self.assertEqual(1 & (x & notone), 0)
        self.assertEqual(1 | (x | notone), allones)

        # compatible terms

        self.assertEqual(x + (y - x), y)
        self.assertEqual(x ^ (y ^ x), y)
        self.assertEqual(x & (y & x), x & y)
        self.assertEqual(x | (y | x), x | y)

        self.assertEqual((x ^ z) + (y - (x ^ z)), y)
        self.assertEqual((x + z) ^ (y ^ (x + z)), y)

        self.assertEqual((x + z) + (y - x), z + y)
        self.assertEqual((x ^ z) ^ (y ^ x), z ^ y)
        self.assertEqual((x & z) & (y & (~x)), 0)
        self.assertEqual((x | z) | (y | (~x)), allones)

    def test_advanced_simplify(self):
        k = Variable("k", 8)

        def f(x, y):
            x = (RotateRight(x, 7) + y) ^ k
            y = RotateLeft(y, 2) ^ x
            return x, y

        def f_inverse(x, y):
            y = RotateRight(x ^ y, 2)
            x = RotateLeft((x ^ k) - y, 7)
            return x, y

        x, y = Variable("x", 8), Variable("y", 8)

        self.assertEqual(f_inverse(*f(x, y)), (x, y))
        self.assertEqual(f(*f_inverse(x, y)), (x, y))


def load_tests(loader, tests, ignore):
    """Add doctests."""
    import arxpy.bitvector.operation
    tests.addTests(doctest.DocTestSuite(arxpy.bitvector.operation))
    return tests
