"""The Differential module provides differential probabilities of bv operators."""
from arxpy.bitvector import core
from arxpy.bitvector import operation

from arxpy.diffcrypt import difference


class Differential(object):
    """Base class to define differentials of bit-vector operations.

    A differential of a bit-vector operation specifies the probability
    that a pair of input and output differences propagates through
    the bit-vector operation.

    To define a new Differential, override the methods is_valid, weight
    and max_weight and specify the short_name, diff_type and operation.
    """

    short_name = None
    diff_type = None
    op = None

    def __init__(self, input_diff, output_diff):
        """Initialize the Differential with given input/output differences."""
        input_diff = core.tuplify(input_diff)
        assert len(input_diff) == sum(self.op.arity)
        self.input_diff = input_diff
        self.output_diff = output_diff

    def is_valid(self):
        """Return the bv expression for non-zero propagation probability."""
        raise NotImplementedError()

    def weight(self):
        """Return the weight of the differential.

        The weight of a differential is a bit-vector expression computing
        - log_2(p), where p is the differential probability.

        It is also possible to return a function of the weight, i.e.
        f(-log_2(p)). In that case, the method weight_function has to
        be overriden with such function f.
        """
        raise NotImplementedError()

    def weight_function(self, w):
        """Return the function used to compute the final weight.

        See weight() for more information. By default, the weight_function
        is the identity.
        """
        return w

    def inverse_weight_function(self, w):
        """Return the inverse of the weight function.

        See weight() for more information. By default, this function
        is the identity.
        """
        return w

    def _weight_var_name(self):
        """Return the name of the variable representing the weight."""
        ids, od = list(self.input_diff), self.output_diff
        for i in range(len(ids)):
            if isinstance(ids[i], core.Constant):
                ids[i] = hex(ids[i].val)[2:]  # without 0x
        return "w_{}_{}".format(''.join([str(i) for i in ids]), od)

    def __str__(self):
        assert self.short_name
        return "{}({}, {})".format(self.short_name, self.input_diff,
                                   self.output_diff)

    __repr__ = __str__


class XDBvAdd(Differential):
    """XOR differential of modular addition."""

    short_name = "xdp+"
    diff_type = difference.XorDiff
    op = operation.BvAdd

    def __init__(self, input_diff, output_diff):
        """Initialize the differential."""
        assert all(isinstance(d, difference.DiffVar) for d in input_diff)
        assert isinstance(output_diff, difference.DiffVar)
        super().__init__(input_diff, output_diff)

    def is_valid(self):
        """Return the bv expression for non-zero propagation probability.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import DiffVar
            >>> from arxpy.diffcrypt.differential import XDBvAdd
            >>> a, b, c = DiffVar("a", 8), DiffVar("b", 8), DiffVar("c", 8)
            >>> xda = XDBvAdd([a, b], c)
            >>> xda.is_valid()  # doctest: +ELLIPSIS
            0x00 == (((~(a << 0x01) ^ (b << 0x01)) & (~(a << 0x01) ...
            >>> zero = Constant(0, 8)
            >>> xda.is_valid().xreplace({a: zero, b: zero, c: zero})
            0b1

        """
        a, b = self.input_diff
        c = self.output_diff

        def eq(x, y, z):
            return (~x ^ y) & (~x ^ z)

        return operation.BvComp(
            eq(a << 1, b << 1, c << 1) & (a ^ b ^ c ^ (b << 1)),
            core.Constant(0, a.width))

    def weight(self):
        """Return the weight of the differential.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import DiffVar
            >>> from arxpy.diffcrypt.differential import XDBvAdd
            >>> a, b, c = DiffVar("a", 8), DiffVar("b", 8), DiffVar("c", 8)
            >>> xda = XDBvAdd([a, b], c)
            >>> xda.weight()  # doctest: +ELLIPSIS
            ((0x0f & ((0x33 & ((0x55 & ((~((b ^ ~a) & (c ^ ~a)) << 0x01) ...
            >>> zero = Constant(0, 8)
            >>> xda.weight().xreplace({a: zero, b: zero, c: zero})
            0x0

        """
        a, b = self.input_diff
        c = self.output_diff

        def eq(x, y, z):
            return (~x ^ y) & (~x ^ z)

        return _HammingWeight(~eq(a, b, c) << 1)  # ignore MSB


class RXDBvAdd(Differential):
    """Rotational-XOR differential of modular addition."""

    short_name = "rxdp+"
    diff_type = difference.RXDiff
    op = operation.BvAdd

    def __init__(self, input_diff, output_diff):
        """Initialize the differential."""
        assert all(isinstance(d, difference.DiffVar) for d in input_diff)
        assert isinstance(output_diff, difference.DiffVar)
        super().__init__(input_diff, output_diff)

    def is_valid(self):
        """Return the bv expression for non-zero propagation probability.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import DiffVar
            >>> from arxpy.diffcrypt.differential import RXDBvAdd
            >>> a, b, c = DiffVar("a", 8), DiffVar("b", 8), DiffVar("c", 8)
            >>> rxda = RXDBvAdd([a, b], c)
            >>> rxda.is_valid()  # doctest: +ELLIPSIS
            0b111111 == (~(((((c >> 0x01) ^ ((a >> 0x01) ^ (b >> 0x01))) ...
            >>> zero = Constant(0, 8)
            >>> rxda.is_valid().xreplace({a: zero, b: zero, c: zero})
            0b1

        """
        # (I ^ SHL)(da ^ db ^ dc)[1:] <= SHL((da ^ dc) OR (db ^ dc))[1:]

        alpha, beta = self.input_diff
        gamma = self.output_diff
        # da, db, dc = alpha[:1], beta[:1], gamma[:1]  # boolector error
        da, db, dc = alpha >> 1, beta >> 1, gamma >> 1  # ignore LSB

        lhs = (da ^ db ^ dc) ^ ((da ^ db ^ dc) << 1)
        rhs = ((da ^ dc) | (db ^ dc)) << 1

        def bitwise_implication(x, y):
            return (~x) | y

        n = lhs.width
        return operation.BvComp(
            bitwise_implication(lhs[n - 2:1], rhs[n - 2:1]),  # ignore MSB, LSB
            ~ core.Constant(0, n - 2))
        # return operation.BvComp(
        #     bitwise_implication(lhs[:1], rhs[:1]),
        #     ~ core.Constant(0, n - 1))

    def weight(self):
        """Return the weight of the differential.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import DiffVar
            >>> from arxpy.diffcrypt.differential import RXDBvAdd
            >>> a, b, c = DiffVar("a", 8), DiffVar("b", 8), DiffVar("c", 8)
            >>> rxda = RXDBvAdd([a, b], c)
            >>> rxda.weight()  # doctest: +ELLIPSIS
            (0b00010 * (0b00 ∘ (((0x0f & ((0x33 & ((0x55 & ((0b00 ...
            >>> zero = Constant(0, 8)
            >>> rxda.weight().xreplace({a: zero, b: zero, c: zero})
            0b00011

        """
        alpha, beta = self.input_diff
        gamma = self.output_diff
        da, db, dc = alpha[:1], beta[:1], gamma[:1]
        # da, db, dc = alpha >> 1, beta >> 1, gamma >> 1

        rhs = ((da ^ dc) | (db ^ dc)) << 1
        hw = _HammingWeight(rhs[:1])  # ignore LSB

        max_hw = rhs.width - 1
        weight_width = max((2 * max_hw + 6).bit_length(), hw.width, 3)  # 0b110

        # let lhs = LSB(lhs) = da ^ db ^ dc
        #     rhs = LSB(rhs) = 0
        # case A (2 * 1.415): lhs => rhs
        # case B (2 * 3):     lhs ^ 1 => rhs

        def bitwise_implication(x, y):
            return (~x) | y

        cte_part = operation.Ite(
            bitwise_implication(da[0] ^ db[0] ^ dc[0], core.Constant(0, 1)),
            core.Constant(3, weight_width),
            core.Constant(6, weight_width)
        )

        hw_extend = operation.ZeroExtend(hw, weight_width - hw.width)

        return 2 * hw_extend + cte_part

    def weight_function(self, w):
        return w * 2

    def inverse_weight_function(self, w):
        return w / 2


class _HammingWeight(operation.Operation):
    """The hamming weight operation of a bit-vector.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.diffcrypt.differential import _HammingWeight
        >>> _HammingWeight(Constant(0b1010, 4))
        0b010
        >>> _HammingWeight(Constant(0b101, 3))
        0b10
        >>> _HammingWeight(Variable("x", 4))  # doctest: +NORMALIZE_WHITESPACE
        ((0x3 & ((0x5 & x) + (0x5 & (x >> 0x1)))) +
        (0x3 & (((0x5 & x) + (0x5 & (x >> 0x1))) >> 0x2)))[2:]
        >>> _HammingWeight(Variable("x", 3))  # doctest: +NORMALIZE_WHITESPACE
        ((0x3 & ((0x5 & ((0b0 ∘ x) >> 0x1)) + (0x5 & (0b0 ∘ x)))) +
        (0x3 & (((0x5 & ((0b0 ∘ x) >> 0x1)) + (0x5 & (0b0 ∘ x))) >> 0x2)))[1:]
    """

    arity = [1, 0]
    is_symmetric = False
    short_name = "HW"

    @classmethod
    def output_width(cls, x):
        return x.width.bit_length()

    @classmethod
    def eval(cls, bv):
        def bv_pattern(pattern, width):
            """Repeat the pattern until obtain a bv of given width."""
            assert width % pattern.width == 0
            return operation.Repeat(pattern, width // pattern.width)

        def simple_pattern(width):
            """Obtain the pattern 0...01...1 with given 0-width."""
            zeroes = core.Constant(0, width)
            return operation.Concat(zeroes, ~zeroes)

        original_width = bv.width
        while (bv.width & (bv.width - 1)) != 0:
            bv = operation.ZeroExtend(bv, 1)
        width_log2 = bv.width.bit_length() - 1

        m_ctes = []
        for i in range(width_log2):
            m_ctes.append(bv_pattern(simple_pattern(2 ** i), bv.width))

        for i, m in enumerate(m_ctes):
            bv = (bv & m) + ((bv >> 2 ** i) & m)

        return bv[original_width.bit_length() - 1:]
