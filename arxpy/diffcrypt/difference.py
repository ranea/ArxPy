"""The Difference module provides types of differences and their propagations.

Given a pair of bit-vectors, several types of differences can be considered.
The most usual is the XOR difference.

Given two bit-vector variables X and Y with a difference D, this modules
implements the propagation of the difference D through the bit-vector
operators.
"""
from arxpy.bitvector import core
from arxpy.bitvector import operation


class DiffVar(core.Variable):
    """A bit-vector variable representing a difference."""

    @classmethod
    def from_Variable(cls, bvs):
        """Convert the Variable to a DiffVar."""
        return DiffVar(bvs.name, bvs.width)

    def to_Variable(self):
        """Convert the DiffVar to a Variable."""
        return core.Variable(self.name, self.width)


class Difference(object):
    """Base class for difference types.

    A *difference* is a bit-vector expression between two bit-vectors.
    For example, the XOR difference between X and Y the operation X ^ Y.

    To define a new difference, override the methods get_difference,
    get_pair_element and propagate.
    """

    @classmethod
    def get_difference(cls, x, y):
        """Return the difference of a given pair (x, y)."""
        raise NotImplementedError()

    @classmethod
    def get_pair_element(cls, x, diff):
        """Return the bit-vector y such that (x, y) has the given difference."""
        raise NotImplementedError()

    @classmethod
    def propagate(cls, op, input_diff, output_diff=None):
        """Propagate an input difference variable through a bit-vector op.

        The output difference variable only need to be specified is the
        propagation is probabilistic (e.g. XOR difference through
        modular addition).
        """
        raise NotImplementedError()


class XorDiff(Difference):
    """Compute and propagate XOR differences."""

    short_name = "XOR"

    @classmethod
    def get_difference(cls, x, y):
        """Return the XOR difference of a given pair (x, y).

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import XorDiff
            >>> x, y = Constant(0b000, 3), Constant(0b000, 3)
            >>> XorDiff.get_difference(x, y)
            0b000
            >>> x, y = Constant(0b000, 3), Constant(0b111, 3)
            >>> XorDiff.get_difference(x, y)
            0b111

        """
        assert isinstance(x, core.Term)
        assert isinstance(y, core.Term)
        return x ^ y

    @classmethod
    def get_pair_element(cls, x, diff):
        """Return the bit-vector y such that (x, y) has the given difference.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import XorDiff
            >>> x, diff = Constant(0b101, 3), Constant(0b000, 3)
            >>> XorDiff.get_pair_element(x, diff)
            0b101
            >>> x, diff = Constant(0b101, 3), Constant(0b001, 3)
            >>> XorDiff.get_pair_element(x, diff)
            0b100

        """
        assert isinstance(x, core.Term)
        assert isinstance(diff, core.Term)
        return x ^ diff

    @classmethod
    def propagate(cls, op, input_diff, output_diff=None):
        """Propagate an input difference variable through a bit-vector op.

            >>> from arxpy.bitvector.core import Variable, Constant
            >>> from arxpy.bitvector.operation import BvAdd, BvXor
            >>> from arxpy.diffcrypt.difference import XorDiff, DiffVar
            >>> d1, d2 = DiffVar("d1", 8), DiffVar("d2", 8)
            >>> XorDiff.propagate(BvXor, [d1, d2])
            d1 ^ d2
            >>> d3 = DiffVar("d3", 8)
            >>> XorDiff.propagate(BvAdd, [d1, d2], d3)
            xdp+((d1, d2), d3)

        Deterministic propagations return bit-vector terms while
        probabilistic propagations return Differential objects.

        """
        input_diff = core.tuplify(input_diff)
        assert len(input_diff) == sum(op.arity)

        msg = "unknown XOR propagation of {}({})".format(
            type(op).__name__,
            [d.vrepr() if isinstance(d, core.Term) else d for d in input_diff])

        if op == operation.BvNot:
            x = input_diff[0]
            if isinstance(x, DiffVar):
                return x
            else:
                raise NotImplementedError(msg)

        if op == operation.BvXor:
            newdiffs = []
            for d in input_diff:
                if isinstance(d, DiffVar):
                    newdiffs.append(d)
                elif isinstance(d, (core.Constant, core.Variable)):
                    newdiffs.append(core.Constant(0, d.width))
                else:
                    raise NotImplementedError(msg)

            return op(*newdiffs)

        if op in [operation.RotateLeft, operation.RotateRight]:
            x, r = input_diff
            if isinstance(x, DiffVar):
                return op(x, r)
            else:
                raise NotImplementedError(msg)

        if op == operation.Extract:
            x, i, j = input_diff
            if isinstance(x, DiffVar):
                return op(x, i, j)
            else:
                raise NotImplementedError(msg)

        if op == operation.Concat:
            if all(isinstance(d, DiffVar) for d in input_diff):
                return op(*input_diff)
            else:
                raise NotImplementedError(msg)

        if op == operation.BvAdd:
            if all(isinstance(d, DiffVar) for d in input_diff):
                from arxpy.diffcrypt import differential
                return differential.XDBvAdd(input_diff, output_diff)
            else:
                raise NotImplementedError(msg)

        if hasattr(op, "differential"):
            return op.differential(cls)(input_diff, output_diff)

        raise NotImplementedError(msg)


class RXDiff(Difference):
    """Compute and propagate Rotational-Xor (RX) differences.

    Given two bit-vector X and Y, the RX difference of the pair (X, Y)
    is the value X ^ (Y <<< 1). In other words, given the value D,
    the pair (X, (X ^ D) >>> 1) has RX difference D.
    """

    short_name = "RX"

    @classmethod
    def get_difference(cls, x, y):
        """Return the RX difference of a given pair (x, y).

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import RXDiff
            >>> x, y = Constant(0b000, 3), Constant(0b000, 3)
            >>> RXDiff.get_difference(x, y)
            0b000
            >>> x, y = Constant(0b000, 3), Constant(0b001, 3)
            >>> RXDiff.get_difference(x, y)
            0b010

        """
        assert isinstance(x, core.Term)
        assert isinstance(y, core.Term)
        return x ^ operation.RotateLeft(y, 1)  # 2018 version
        # return operation.RotateLeft(x, 1) ^ y  # 2016 version

    @classmethod
    def get_pair_element(cls, x, diff):
        """Return the bit-vector y such that (x, y) has the given difference.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import RXDiff
            >>> x, diff = Constant(0b000, 3), Constant(0b000, 3)
            >>> RXDiff.get_pair_element(x, diff)
            0b000
            >>> x, diff = Constant(0b000, 3), Constant(0b010, 3)
            >>> RXDiff.get_pair_element(x, diff)
            0b001

        """
        assert isinstance(x, core.Term)
        assert isinstance(diff, core.Term)
        return operation.RotateRight(x ^ diff, 1)
        # return operation.RotateLeft(x, 1) ^ diff

    @classmethod
    def propagate(cls, op, input_diff, output_diff=None):
        """Propagate an input difference variable through a bit-vector op.

            >>> from arxpy.bitvector.core import Variable, Constant
            >>> from arxpy.bitvector.operation import BvAdd, BvXor
            >>> from arxpy.diffcrypt.difference import RXDiff, DiffVar
            >>> d1, d2 = DiffVar("d1", 8), DiffVar("d2", 8)
            >>> RXDiff.propagate(BvXor, [d1, d2])
            d1 ^ d2
            >>> d3 = DiffVar("d3", 8)
            >>> RXDiff.propagate(BvAdd, [d1, d2], d3)
            rxdp+((d1, d2), d3)

        Deterministic propagations return bit-vector terms while
        probabilistic propagations return Differential objects.
        """
        input_diff = core.tuplify(input_diff)
        assert len(input_diff) == sum(op.arity)

        msg = "unknown RX propagation of {}({})".format(
            type(op).__name__,
            [d.vrepr() if isinstance(d, core.Term) else d for d in input_diff])

        if op == operation.BvNot:
            x = input_diff[0]
            if isinstance(x, DiffVar):
                return x
            else:
                raise NotImplementedError(msg)

        if op == operation.BvXor:
            if all(isinstance(d, DiffVar) for d in input_diff):
                return op(*input_diff)
            else:
                x, y = input_diff
                if isinstance(x, core.Constant) and isinstance(y, DiffVar):
                    cte, var = x, y
                elif isinstance(y, core.Constant) and isinstance(x, DiffVar):
                    cte, var = y, x
                else:
                    raise NotImplementedError(msg)

                cte = operation.BvXor(cte, operation.RotateLeft(cte, 1))
                return operation.BvXor(var, cte)

        if op in [operation.RotateLeft, operation.RotateRight]:
            x, r = input_diff
            if isinstance(x, DiffVar):
                return op(x, r)
            else:
                raise NotImplementedError(msg)

        if op == operation.BvAdd:
            if all(isinstance(d, DiffVar) for d in input_diff):
                from arxpy.diffcrypt import differential
                return differential.RXDBvAdd(input_diff, output_diff)
            else:
                raise NotImplementedError(msg)

        if hasattr(op, "differential"):
            return op.differential(cls)(input_diff, output_diff)

        raise NotImplementedError(msg)
