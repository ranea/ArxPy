"""Manipulate differences."""
import collections

from arxpy.bitvector import core
from arxpy.bitvector import operation
from arxpy.bitvector import extraop


def _tuplify(seq):
    if isinstance(seq, collections.abc.Sequence):
        return tuple(seq)
    else:
        return tuple([seq])


class Difference(object):
    """Represent differences.

    The *difference* between two `Term` :math:`x` and :math:`y`
    is defined as :math:`\\alpha = y - x`,
    where the *difference operation* :math:`-`
    is a bit-vector `Operation`. In other words, the pair
    :math:`(x, x + \\alpha)` has difference :math:`\\alpha`,
    where :math:`+` is the inverse of the difference operation.

    The most common difference used in differential cryptanalysis
    is the XOR difference `XorDiff` (where the difference operation
    is `BvXor`). Other examples are the additive difference
    (where the difference operation is `BvSub`) or the rotational-XOR
    difference `RXDiff`.

    Note that arithmetic with differences is not supported.
    For example, two `Difference` objects ``d1`` and ``d2``
    cannot be XORed, i.e., ``d1 ^ d2``.
    This can be done instead by performing the arithmetic with
    the difference values and converting the resulting
    `Term` to a difference, that is, ``Difference(d1.val ^ d2.val)``

    This class is not meant to be instantiated but to provide a base
    class for the different types of differences.

    Attributes:
        val: a `Term` representing the value of the difference.
        diff_op: the difference `Operation`.
        inv_diff_op: the inverse of the difference operation.

    """
    diff_op = None
    inv_diff_op = None

    def __init__(self, value):
        assert isinstance(value, core.Term)
        self.val = value

    def __str__(self):
        """Return the non-verbose string representation."""
        return "{}({})".format(type(self).__name__, str(self.val))

    __repr__ = __str__

    def __hash__(self):
        return hash(self.val)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.val == other.val
        else:
            return False

    def xreplace(self, rule):
        """Replace occurrences of differences within the expression.

        The argument ``rule`` is a dict-like object representing
        the replacement rule.

        This method is similar to SymPy `xreplace
        <https://docs.sympy.org/latest/modules/core.html?highlight=xreplace#
        sympy.core.basic.Basic.xreplace>`_ but with the restriction that
        only differences objects are allowed in ``rule``.
        """
        for d in rule:
            assert isinstance(d, type(self)) and isinstance(rule[d], type(self))

        rule = {d.val: rule[d].val for d in rule}
        return type(self)(self.val.xreplace(rule))

    def vrepr(self):
        """Return a verbose string representation."""
        return "{}({})".format(type(self).__name__, self.val.vrepr())

    @classmethod
    def from_pair(cls, x, y):
        """Return the `Difference` :math:`\\alpha = y - x` given two `Term`."""
        assert isinstance(x, core.Term)
        assert isinstance(y, core.Term)
        return cls(cls.diff_op(x, y))  # The order of the operands is important

    def get_pair_element(self, x):
        """Return the `Term` :math:`y` such that :math:`y = \\alpha + x`."""
        assert isinstance(x, core.Term)
        return self.inv_diff_op(x, self.val)

    @classmethod
    def derivative(cls, op, input_diff):
        """Return the derivative of ``op`` at the point ``input_diff``.

        The derivative of an `Operation` :math:`f`
        at the point :math:`\\alpha` (also called the input difference)
        is defined as :math:`f_{\\alpha} (x) = f(x + \\alpha) - f(x)`.
        Note that :math:`f_{\\alpha} (x)` is the difference of
        :math:`(f(x), f(x + \\alpha))`.

        If :math:`f` has multiple operands, :math:`\\alpha` is a list
        containing the `Difference` of each operand and the
        computation :math:`x + \\alpha` is defined component-wise, that is,
        :math:`x = (x_1, \dots, x_n)`, :math:`\\alpha = (\\alpha_1, \dots, \\alpha_n)`,
        and :math:`x + \\alpha = (x_1 + \\alpha_1, \dots, x_n + \\alpha_n)`.

        For some operations, there is a unique output difference :math:`\\beta`
        for every input difference :math:`\\alpha`, that is, :math:`f_{\\alpha}(x) = \\beta`
        is a constant function. In this case, this method returns the `Difference`
        :math:`\\beta`. Otherwise, it returns a `Derivative` object representing
        :math:`f_{\\alpha}`.

        Operations with scalar operands are not supported, but these
        operands can be removed with `make_partial_operation` and
        the derivative of the resulting operator can then be computed.

        Args:
            op:  a bit-vector operator
            input_diff: a list containing the difference of each operand
        """
        raise NotImplementedError("subclasses need to override this method")


class XorDiff(Difference):
    """Represent XOR differences.

    The XOR difference of two `Term` is given by the XOR
    of the terms. In other words, the *difference operation*
    of `XorDiff` is the `BvXor` (see `Difference`).

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.differential.difference import XorDiff
        >>> x, y = Constant(0b000, 3), Constant(0b000, 3)
        >>> alpha = XorDiff.from_pair(x, y)
        >>> alpha
        XorDiff(0b000)
        >>> alpha.get_pair_element(x)
        0b000
        >>> x, y = Constant(0b010, 3), Constant(0b101, 3)
        >>> alpha = XorDiff.from_pair(x, y)
        >>> alpha
        XorDiff(0b111)
        >>> alpha.get_pair_element(x)
        0b101
        >>> k = Variable("k", 8)
        >>> alpha = XorDiff.from_pair(k, k)
        >>> alpha
        XorDiff(0x00)
        >>> alpha.get_pair_element(k)
        k
    """

    diff_op = operation.BvXor
    inv_diff_op = operation.BvXor

    @classmethod
    def derivative(cls, op, input_diff):
        """Return the derivative of ``op`` at the point ``input_diff``.

        See `Difference.derivative` for more information.

            >>> from arxpy.bitvector.core import Variable, Constant
            >>> from arxpy.bitvector.operation import BvAdd, BvXor, RotateLeft, BvSub
            >>> from arxpy.bitvector.extraop import make_partial_operation
            >>> from arxpy.differential.difference import XorDiff
            >>> d1, d2 = XorDiff(Variable("d1", 8)), XorDiff(Variable("d2", 8))
            >>> XorDiff.derivative(BvXor, [d1, d2])
            XorDiff(d1 ^ d2)
            >>> Xor1 = make_partial_operation(BvXor, tuple([None, Constant(1, 8)]))
            >>> XorDiff.derivative(Xor1, d1)
            XorDiff(d1)
            >>> Rotate1 = make_partial_operation(RotateLeft, tuple([None, 1]))
            >>> XorDiff.derivative(Rotate1, d1)
            XorDiff(d1 <<< 1)
            >>> XorDiff.derivative(BvAdd, [d1, d2])
            XDA(XorDiff(d1), XorDiff(d2))
            >>> XorDiff.derivative(BvSub, [d1, d2])
            XDS(XorDiff(d1), XorDiff(d2))
            >>> CteAdd1 = make_partial_operation(BvAdd, tuple([None, Constant(1, 8)]))
            >>> XorDiff.derivative(CteAdd1, d1)
            XDCA_0x01(XorDiff(d1))

        """
        input_diff = _tuplify(input_diff)
        assert len(input_diff) == sum(op.arity)

        msg = "invalid arguments: op={}, input_diff={}".format(
            op.__name__,
            [d.vrepr() if isinstance(d, core.Term) else d for d in input_diff])

        if not all(isinstance(diff, cls) for diff in input_diff):
            raise ValueError(msg)

        if op == operation.BvNot:
            return input_diff[0]

        if op == operation.BvXor:
            return cls(op(*[d.val for d in input_diff]))

        if op == operation.Concat:
            return cls(op(*[d.val for d in input_diff]))

        if op == operation.BvAdd:
            from arxpy.differential import derivative
            return derivative.XDA(input_diff)

        if op == operation.BvSub:
            from arxpy.differential import derivative
            return derivative.XDS(input_diff)

        if issubclass(op, extraop.PartialOperation):
            if op.base_op == operation.BvXor:
                assert len(input_diff) == 1
                d1 = input_diff[0]
                val = op.fixed_args[0] if op.fixed_args[0] is not None else op.fixed_args[1]
                d2 = cls.from_pair(val, val)
                input_diff = [d1, d2]
                return cls(op.base_op(*[d.val for d in input_diff]))

            if op.base_op == operation.BvAnd:
                assert len(input_diff) == 1
                d1 = input_diff[0]
                val = op.fixed_args[0] if op.fixed_args[0] is not None else op.fixed_args[1]
                if isinstance(val, core.Constant):
                    return cls(op.base_op(d1.val, val))

            if op.base_op in [operation.RotateLeft, operation.RotateRight]:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None:
                    assert len(input_diff) == 1
                    d = input_diff[0]
                    return cls(op.base_op(d.val, op.fixed_args[1]))
                else:
                    raise ValueError(msg)

            if op.base_op in [operation.BvShl, operation.BvLshr]:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None:
                    assert len(input_diff) == 1
                    d = input_diff[0]
                    return cls(op.base_op(d.val, op.fixed_args[1]))
                else:
                    raise ValueError(msg)

            if op.base_op == operation.Extract:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None and op.fixed_args[2] is not None:
                    assert len(input_diff) == 1
                    d = input_diff[0]
                    return cls(op.base_op(d.val, op.fixed_args[1], op.fixed_args[2]))
                else:
                    raise ValueError(msg)

            if op.base_op == operation.Concat:
                assert len(input_diff) == 1
                d1 = input_diff[0]
                if op.fixed_args[0] is not None:
                    val = op.fixed_args[0]
                    input_diff = [cls.from_pair(val, val), d1]
                else:
                    val = op.fixed_args[1]
                    input_diff = [d1, cls.from_pair(val, val)]
                return cls(op.base_op(*[d.val for d in input_diff]))

            if op.base_op == operation.BvAdd:
                assert len(input_diff) == 1
                d = input_diff[0]
                cte = op.fixed_args[0] if op.fixed_args[0] is not None else op.fixed_args[1]
                from arxpy.differential import derivative
                return derivative.XDCA(d, cte)
            else:
                raise ValueError(msg)

        if hasattr(op, "xor_derivative"):
            return op.xor_derivative(input_diff)

        raise ValueError(msg)


class RXOp(operation.Operation):
    """The difference operation of `RXDiff`."""

    arity = [2, 0]
    is_symmetric = False
    is_simple = True

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        return operation.RotateLeft(x, 1) ^ y


class RXInvOp(operation.Operation):
    """The inverse of the  difference operation of `RXDiff`."""

    arity = [2, 0]
    is_symmetric = False
    is_simple = True

    @classmethod
    def condition(cls, x, d):
        return x.width == d.width

    @classmethod
    def output_width(cls, x, d):
        return x.width

    @classmethod
    def eval(cls, x, d):
        return operation.RotateLeft(x, 1) ^ d


class RXDiff(Difference):
    """Represent rotational-XOR (RX) differences.

    The pair ``(x, (x <<< 1) ^ d)`` has RX difference ``d``.
    In other words,  the RX difference of two `Terms` ``x`` and ``y``
    is defined as ``(x <<< 1) ^ y``.

    See `Difference` for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.differential.difference import RXDiff
        >>> x, y = Constant(0b000, 3), Constant(0b000, 3)
        >>> alpha = RXDiff.from_pair(x, y)
        >>> alpha
        RXDiff(0b000)
        >>> alpha.get_pair_element(x)
        0b000
        >>> x, y = Constant(0b000, 3), Constant(0b001, 3)
        >>> alpha = RXDiff.from_pair(x, y)
        >>> alpha
        RXDiff(0b001)
        >>> alpha.get_pair_element(x)
        0b001
        >>> k = Variable("k", 8)
        >>> alpha = RXDiff.from_pair(k, k)
        >>> alpha
        RXDiff(k ^ (k <<< 1))
        >>> alpha.get_pair_element(k)
        k
    """

    diff_op = RXOp
    inv_diff_op = RXInvOp

    @classmethod
    def derivative(cls, op, input_diff):
        """Return the derivative of ``op`` at the point ``input_diff``.

        See `Difference.derivative` for more information.

            >>> from arxpy.bitvector.core import Variable, Constant
            >>> from arxpy.bitvector.operation import BvAdd, BvXor, RotateLeft
            >>> from arxpy.bitvector.extraop import make_partial_operation
            >>> from arxpy.differential.difference import RXDiff
            >>> d1, d2 = RXDiff(Variable("d1", 8)), RXDiff(Variable("d2", 8))
            >>> RXDiff.derivative(BvXor, [d1, d2])
            RXDiff(d1 ^ d2)
            >>> Xor1 = make_partial_operation(BvXor, tuple([None, Constant(1, 8)]))
            >>> RXDiff.derivative(Xor1, d1)
            RXDiff(0x03 ^ d1)
            >>> Rotate1 = make_partial_operation(RotateLeft, tuple([None, 1]))
            >>> RXDiff.derivative(Rotate1, d1)
            RXDiff(d1 <<< 1)
            >>> RXDiff.derivative(BvAdd, [d1, d2])
            RXDA(RXDiff(d1), RXDiff(d2))

        """
        input_diff = _tuplify(input_diff)
        assert len(input_diff) == sum(op.arity)

        msg = "invalid arguments: op={}, input_diff={}".format(
            op.__name__,
            [d.vrepr() if isinstance(d, core.Term) else d for d in input_diff])

        if not all(isinstance(diff, cls) for diff in input_diff):
            raise ValueError(msg)

        if op == operation.BvNot:
            return input_diff[0]

        if op == operation.BvXor:
            return cls(op(*[d.val for d in input_diff]))

        if op == operation.BvAdd:
            from arxpy.differential import derivative
            return derivative.RXDA(input_diff)

        # Concact, BvSub

        if issubclass(op, extraop.PartialOperation):
            if op.base_op == operation.BvXor:
                assert len(input_diff) == 1
                d1 = input_diff[0]
                val = op.fixed_args[0] if op.fixed_args[0] is not None else op.fixed_args[1]
                d2 = cls.from_pair(val, val)
                input_diff = [d1, d2]
                return cls(op.base_op(*[d.val for d in input_diff]))

            if op.base_op in [operation.RotateLeft, operation.RotateRight]:
                if op.fixed_args[0] is None and op.fixed_args[1] is not None:
                    assert len(input_diff) == 1
                    d = input_diff[0]
                    return cls(op.base_op(d.val, op.fixed_args[1]))
                else:
                    raise ValueError(msg)

            # RX-model of BvAddCte not implemented (approximation with BvAdd too weak)
            # RX-model of BvShl and BvLshr not implemented (non-linear w.r.t RX-diffs)

        if hasattr(op, "rx_derivative"):
            return op.rx_derivative(input_diff)

        raise ValueError(msg)

