"""Manipulate derivatives of bit-vector operations."""
import collections
import functools
import math
import operator
import fractions

from arxpy.bitvector import core
from arxpy.bitvector import operation
from arxpy.bitvector import extraop

from arxpy.differential import difference


def _tuplify(seq):
    if isinstance(seq, collections.abc.Sequence):
        return tuple(seq)
    else:
        return tuple([seq])


class Derivative(object):
    """Represent derivatives of bit-vector operations.

    Given a `Difference` operation :math:`-`  and its inverse :math:`+`,
    the derivative of an `Operation` :math:`f`
    at the input difference :math:`\\alpha` is defined as
    :math:`f_{\\alpha} (x) = f(x + \\alpha) - f(x)`
    (see also `Difference.derivative`).

    .. note:: Derivatives of functions with scalar args are not supported.

    This class is not meant to be instantiated but to provide a base
    class for creating derivatives.

    Attributes:
        diff_type: the type of `Difference`
        op: the bit-vector `Operation` :math:`f`
        input_diff: a list containing the `Difference` of each bit-vector
            operand.
    """
    diff_type = None
    # op  # some subclasses need this attribute as an object (not class) attribute

    def __init__(self, input_diff):  # noqa: 102
        input_diff = _tuplify(input_diff)

        assert self.op.arity[1] == 0  # no scalars
        assert len(input_diff) == sum(self.op.arity)
        assert all((isinstance(d, type(self).diff_type) for d in input_diff))

        self.input_diff = input_diff

    def __str__(self):
        return "{}{}".format(type(self).__name__, self.input_diff)

    __repr__ = __str__

    def eval(self, *x):
        """Evaluate the derivative at :math:`x`.

        Return:
            `Difference`: the corresponding output difference
        """
        assert all(isinstance(d, core.Term) for d in x)
        f_x = self.op(*x)
        y = [a_i.get_pair_element(x_i) for x_i, a_i in zip(x, self.input_diff)]
        f_y = self.op(*y)
        return type(self).diff_type.from_pair(f_x, f_y)

    def is_possible(self, output_diff):
        """Return whether the given output `Difference` is possible.

        An output difference :math:`\\beta` is possible if exists
        :math:`x` such that :math:`f_{\\alpha} (x) = \\beta`.

        If the output difference is a constant value, this method returns
        the `Constant` ``0b1`` or ``0b0`` depending on whether the output
        difference in possible. If the output difference is symbolic,
        this method returns a bit-vector `Term` that evaluates to ``0b1``
        or ``0b0`` depending on whether the symbolic output difference is
        replaced by a valid output difference.
        """
        raise NotImplementedError("subclasses need to override this method")

    def has_probability_one(self, output_diff):
        """Return whether the input difference propagates to the given output difference
        with probability one."""
        raise NotImplementedError()

    def weight(self, output_diff):
        """Return the weight of a possible output `Difference`.

        Let :math:`\\beta` be the given output difference. The *probability of the differential*
        :math:`p = Pr(\\alpha \\xrightarrow{f} \\beta)` is defined as the *proportion* of
        :math:`f_{\\alpha}`-preimages of :math:`\\beta`, that is,
        :math:`p \ = \ \# \{ x \ : \ f_{\\alpha} (x) = \\beta \} / 2^{n}`,
        where :math:`n` is the bit-width of :math:`x`.

        By default, the weight is defined as the closest integer of :math:`- \log_2(p)`,
        but some derivatives may consider other definitions of weight.
        """
        raise NotImplementedError()

    def max_weight(self):
        """Return the maximum value the weight can achieve."""
        raise NotImplementedError()

    def exact_weight(self, output_diff):
        """Return the weight without rounding to the closest integer.

        It is assumed the exact weight is always smaller than the weight.
        """
        raise NotImplementedError()

    def num_frac_bits(self):
        """Return the number of fractional bits in the weight."""
        raise NotImplementedError()

    def error(self):
        """Return the maximum difference between the weight and the exact weight.

        This method returns an upper bound (in absolute value) of the maximum difference
        (over all input and output difference) between the weight and the exact weight.
        """
        raise NotImplementedError()

    def _replace_input_diff(self, new_input_diff, **kwargs):
        """Return a new derivative object with the input difference replaced."""
        return type(self)(new_input_diff, **kwargs)


class XDA(Derivative):
    """Represent the `XorDiff` `Derivative` of `BvAdd`.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.derivative import XDA
        >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
        >>> f = XDA(alpha)
        >>> x = Constant(0, 4), Constant(0, 4)
        >>> f.eval(*x)  # f(x + alpha) - f(x)
        XorDiff(0x0)
        >>> f.max_weight(), f.error(), f.num_frac_bits()
        (3, 0, 0)

    """

    diff_type = difference.XorDiff
    op = operation.BvAdd

    def is_possible(self, output_diff):
        """Return whether the given output `XorDiff` is possible.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.derivative import XDA
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XDA(alpha)
            >>> beta = XorDiff(Constant(0, 4))
            >>> f.is_possible(beta)
            0b1
            >>> a0, a1, b = Variable("a0", 4), Variable("a1", 4), Variable("b", 4)
            >>> alpha = XorDiff(a0), XorDiff(a1)
            >>> f = XDA(alpha)
            >>> beta = XorDiff(b)
            >>> result = f.is_possible(beta)
            >>> result
            0x0 == ((~(a0 << 0x1) ^ (a1 << 0x1)) & (~(a0 << 0x1) ^ (b << 0x1)) & ((a0 << 0x1) ^ b ^ a0 ^ a1))
            >>> result.xreplace({a0: Constant(0, 4), a1: Constant(0, 4), b: Constant(0, 4)})
            0b1
            >>> a1 = Constant(0, 4)
            >>> alpha = XorDiff(a0), XorDiff(a1)
            >>> f = XDA(alpha)
            >>> beta = XorDiff(b)
            >>> result = f.is_possible(beta)
            >>> result
            0x0 == (~(a0 << 0x1) & ~(b << 0x1) & (a0 ^ b))

        See `Derivative.is_possible` for more information.
        """
        a, b = [d.val for d in self.input_diff]
        c = output_diff.val

        one = core.Constant(1, a.width)

        def eq(x, y, z):
            if not isinstance(x, core.Constant):
                if isinstance(y, core.Constant):
                    return eq(y, x, z)
                elif isinstance(z, core.Constant):
                    return eq(z, x, y)

            return (~x ^ y) & (~x ^ z)

        def xor_shift(x, y, z):
            if not isinstance(x, core.Constant):
                if isinstance(y, core.Constant):
                    return xor_shift(y, x, z)
                elif isinstance(z, core.Constant):
                    return xor_shift(z, x, y)

            return (x ^ y ^ z ^ (x << one))

        return operation.BvComp(
            eq(a << one, b << one, c << one) & xor_shift(a, b, c),
            core.Constant(0, a.width))

        # # https://doi.org/10.1007/3-540-36231-2_5
        # dx1, dx2 = [d.val for d in self.input_diff]
        # dy = output_diff.val
        # one = core.Constant(1, dx1.width)
        #
        # impossible = ~( ((dx1^dy) << one) | ((dx2^dy) << one) ) & (dx1^dx2^dy^(dx2<<one))
        #
        # return operation.BvComp(impossible, core.Constant(0, dx1.width))

    def has_probability_one(self, output_diff):
        """Return whether the input diff propagates to the output diff with probability one.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.derivative import XDA
            >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
            >>> f = XDA(alpha)
            >>> f.has_probability_one(XorDiff(Constant(0, 4)))
            0b1

        """
        is_possible = self.is_possible(output_diff)

        a, b = [d.val for d in self.input_diff]
        c = output_diff.val
        n = a.width

        def eq(x, y, z):
            if not isinstance(x, core.Constant):
                if isinstance(y, core.Constant):
                    return eq(y, x, z)
                elif isinstance(z, core.Constant):
                    return eq(z, x, y)

            return (~x ^ y) & (~x ^ z)

        # not optimized

        return is_possible & operation.BvComp(
            (~eq(a, b, c))[n-2:],
            core.Constant(0, n - 1)
        )

    def weight(self, output_diff):
        """Return the weight of a possible output `XorDiff`.

        For XDA, the probability of a valid differential is :math:`2^{-i}`
        for some :math:`i`, and the weight is defined as the exponent
        :math:`i`.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.derivative import XDA
            >>> n = 4
            >>> alpha = XorDiff(Constant(0, n)), XorDiff(Constant(0, n))
            >>> f = XDA(alpha)
            >>> f.weight(XorDiff(Constant(0, n)))
            0b00
            >>> a0, a1, b = Variable("a0", n), Variable("a1", n), Variable("b", n)
            >>> alpha = XorDiff(a0), XorDiff(a1)
            >>> f = XDA(alpha)
            >>> result = f.weight(XorDiff(b))
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            (0b0 :: (~((a1 ^ ~a0) & (b ^ ~a0))[0])) + (0b0 :: (~((a1 ^ ~a0) & (b ^ ~a0))[1])) +
            (0b0 :: (~((a1 ^ ~a0) & (b ^ ~a0))[2]))
            >>> result.xreplace({a0: Constant(0, n), a1: Constant(0, n), b: Constant(0, n)})
            0b00

        See `Derivative.weight` for more information.
        """
        a, b = [d.val for d in self.input_diff]
        c = output_diff.val
        n = a.width

        def eq(x, y, z):
            if not isinstance(x, core.Constant):
                if isinstance(y, core.Constant):
                    return eq(y, x, z)
                elif isinstance(z, core.Constant):
                    return eq(z, x, y)

            return (~x ^ y) & (~x ^ z)

        return extraop.PopCount((~eq(a, b, c))[n-2:])  # ignore MSB

        # # https://doi.org/10.1007/3-540-36231-2_5
        # dx1, dx2 = [d.val for d in self.input_diff]
        # dy = output_diff.val
        # one = core.Constant(1, dx1.width)
        #
        # w = ((dx1^dy) << one) | ((dx2^dy) << one)
        #
        # return extraop.PopCount(w)

    def max_weight(self):
        width = self.input_diff[0].val.width - 1  # MSB is ignored
        return width  # as an integer

    def exact_weight(self, output_diff):
        return int(self.weight(output_diff))

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class XDS(XDA):
    """Represent the `XorDiff` `Derivative` of `BvSub`.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.derivative import XDS
        >>> alpha = XorDiff(Constant(0, 4)), XorDiff(Constant(0, 4))
        >>> f = XDS(alpha)
        >>> x = Constant(0, 4), Constant(0, 4)
        >>> f.eval(*x)  # f(x + alpha) - f(x)
        XorDiff(0x0)
        >>> f.max_weight(), f.error(), f.num_frac_bits()
        (3, 0, 0)

    The differential model of the modular substraction
    is the same as the model for the modular addition
    (see `XDA`).
    """

    diff_type = difference.XorDiff
    op = operation.BvSub


class RXDA(Derivative):
    """Represent the `RXDiff` `Derivative` of `BvAdd`.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.differential.difference import RXDiff
            >>> from arxpy.differential.derivative import RXDA
            >>> alpha = RXDiff(Constant(0, 16)), RXDiff(Constant(0, 16))
            >>> f = RXDA(alpha)
            >>> x = Constant(0, 16), Constant(0, 16)
            >>> f.eval(*x)  # f(x + alpha) - f(x)
            RXDiff(0x0000)
            >>> f.max_weight(), f.error(), f.num_frac_bits()
            (136, 0.03999347239201656, 3)

    """

    diff_type = difference.RXDiff
    op = operation.BvAdd
    precision = 3  # 0, 2, 3, 6, 7, 8, 10

    def is_possible(self, output_diff):
        """Return whether the given output `RXDiff` is possible.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.differential.difference import RXDiff
            >>> from arxpy.differential.derivative import RXDA
            >>> alpha = RXDiff(Constant(0, 4)), RXDiff(Constant(0, 4))
            >>> f = RXDA(alpha)
            >>> beta = RXDiff(Constant(0, 4))
            >>> f.is_possible(beta)
            0b1
            >>> a0, a1, b = Variable("a0", 4), Variable("a1", 4), Variable("b", 4)
            >>> alpha = RXDiff(a0), RXDiff(a1)
            >>> f = RXDA(alpha)
            >>> beta = RXDiff(b)
            >>> result = f.is_possible(beta)
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            0b11 == (~(((((a0[:1]) ^ (a1[:1]) ^ (b[:1])) << 0b001) ^ (a0[:1]) ^ (a1[:1]) ^ (b[:1]))[:1]) |
            ((((a0[:1]) ^ (b[:1])) | ((a1[:1]) ^ (b[:1])))[1:]))
            >>> result.xreplace({a0: Constant(0, 4), a1: Constant(0, 4), b: Constant(0, 4)})
            0b1
            >>> a1 = Constant(0, 4)
            >>> alpha = RXDiff(a0), RXDiff(a1)
            >>> f = RXDA(alpha)
            >>> beta = RXDiff(b)
            >>> result = f.is_possible(beta)
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            0b11 == (~(((((a0[:1]) ^ (b[:1])) << 0b001) ^ (a0[:1]) ^ (b[:1]))[:1])
            | ((((a0[:1]) ^ (b[:1])) | (b[:1]))[1:]))

        See `Derivative.is_possible` for more information.
        """
        # (I ^ SHL)(da ^ db ^ dc)[1:] <= SHL((da ^ dc) OR (db ^ dc))[1:]

        # one = core.Constant(1, self.input_diff[0].val.width)  # alt v1

        alpha, beta = [d.val for d in self.input_diff]
        gamma = output_diff.val
        # da, db, dc = alpha >> one, beta >> one, gamma >> one  # alt v1
        da, db, dc = alpha[:1], beta[:1], gamma[:1]  # ignore LSB

        one = core.Constant(1, da.width)

        # lhs = (da ^ db ^ dc) ^ ((da ^ db ^ dc) << one)  # alt v1
        # rhs = ((da ^ dc) | (db ^ dc)) << one
        lhs = ((da ^ db ^ dc) ^ ((da ^ db ^ dc) << one))[:1]
        rhs = (((da ^ dc) | (db ^ dc)) << one)[:1]

        def bitwise_implication(x, y):
            return (~x) | y

        # alt v1
        # n = lhs.width
        # return operation.BvComp(
        #     bitwise_implication(lhs[n - 2:1], rhs[n - 2:1]),  # ignore MSB, LSB
        #     ~ core.Constant(0, n - 2))
        return operation.BvComp(bitwise_implication(lhs, rhs), ~core.Constant(0, lhs.width))  # alt v1

    def has_probability_one(self, output_diff):
        """Return whether the input diff propagates to the output diff with probability one."""
        return core.Constant(0, 1)

    @staticmethod
    def decimal2bin(number, precision):
        """Return a binary representation of a positive real number."""
        assert number > 0
        binary_str = bin(int(number))
        frac = number - int(number)
        for i in range(precision):
            frac *= 2
            frac_bit = int(frac)
            if frac_bit == 1:
                frac -= frac_bit
                binary_str += '1'
            else:
                binary_str += '0'
        return binary_str

    def weight(self, output_diff):
        """Return the weight of a possible output `RXDiff`.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.differential.difference import RXDiff
            >>> from arxpy.differential.derivative import RXDA
            >>> n = 4
            >>> alpha = RXDiff(Constant(0, n)), RXDiff(Constant(0, n))
            >>> f = RXDA(alpha)
            >>> f.weight(RXDiff(Constant(0, n)))
            0b001001
            >>> a0, a1, b = Variable("a0", n), Variable("a1", n), Variable("b", n)
            >>> alpha = RXDiff(a0), RXDiff(a1)
            >>> f = RXDA(alpha)
            >>> result = f.weight(RXDiff(b))
            >>> result  # doctest:+NORMALIZE_WHITESPACE
            ((0x0 :: ((0b0 :: ((((a0[:1]) ^ (b[:1])) | ((a1[:1]) ^ (b[:1])))[0])) + (0b0 :: ((((a0[:1]) ^ (b[:1])) |
            ((a1[:1]) ^ (b[:1])))[1])))) << 0b000011) + (Ite(~((a0[1]) ^ (a1[1]) ^ (b[1])), 0b001001, 0b011000))
            >>> result.xreplace({a0: Constant(0, n), a1: Constant(0, n), b: Constant(0, n)})
            0b001001

        See `Derivative.weight` for more information.
        """
        # one = core.Constant(1, self.input_diff[0].val.width)  # alt v1
        # two = core.Constant(2, self.input_diff[0].val.width)

        alpha, beta = [d.val for d in self.input_diff]
        gamma = output_diff.val
        da, db, dc = alpha[:1], beta[:1], gamma[:1]  # alt v1
        # da, db, dc = alpha >> one, beta >> one, gamma >> one  # alt v1

        rhs = ((da ^ dc) | (db ^ dc))[da.width-2:]  # equiv to shift left
        # rhs = ((da ^ dc) | (db ^ dc)) << two  # alt v1
        hw = extraop.PopCount(rhs)

        max_hw = rhs.width - 1

        # (max_hw + 3) = maximum integer part
        k = self.__class__.precision  # num fraction bits
        weight_width = (max_hw + 3).bit_length() + k

        # let lhs = LSB(lhs) = da ^ db ^ dc
        #     rhs = LSB(rhs) = 0
        # case A (w=1.415): lhs => rhs
        # case B (w=3):     lhs ^ 1 => rhs
        # 1.415 = -log2(pr propagation of a rotational pair with offset 1)
        # bin(1.415) = 1.01101010001111010111

        n = alpha.width
        w_rotational_pair = -(math.log2((1 + 2**(1 - n) + 0.5 + 2 ** (-n))) - 2)
        w_rotational_pair = int(self.__class__.decimal2bin(w_rotational_pair, k), base=2)

        def bitwise_implication(x, y):
            return (~x) | y

        cte_part = operation.Ite(
            bitwise_implication(da[0] ^ db[0] ^ dc[0], core.Constant(0, 1)),
            core.Constant(w_rotational_pair, weight_width),
            core.Constant(3, weight_width) << k
        )

        hw_extend = operation.ZeroExtend(hw, weight_width - hw.width)

        return (hw_extend << k) + cte_part

    def max_weight(self):
        width = self.input_diff[0].val.width - 2  # LSB, MSB ignored
        return (width + 3) << self.__class__.precision  # as an integer

    def exact_weight(self, output_diff):
        """Return the weight without rounding to the closest integer.

        When the input/output differences have width smaller than 16,
        the exact weight does not coincide with the actual real weight.
        For example, for width=8, the error can be as high as 1.2.
        Moreover, in this case the exact weight is not always smaller
        than the weight.
        """
        alpha, beta = [d.val for d in self.input_diff]
        gamma = output_diff.val
        da, db, dc = alpha[:1], beta[:1], gamma[:1]

        one = core.Constant(1, da.width)

        rhs = ((da ^ dc) | (db ^ dc)) << one
        # rhs = ((da ^ dc) | (db ^ dc))[da.width-2:]  # alt v1
        hw = extraop.PopCount(rhs)

        result = int(hw)

        def bitwise_implication(x, y):
            return (~x) | y

        n = alpha.width
        w_rotational_pair = -(math.log2((1 + 2**(1 - n) + 0.5 + 2**(-n))) - 2)

        if bitwise_implication(da[0] ^ db[0] ^ dc[0], core.Constant(0, 1)) == core.Constant(1, 1):
            result += w_rotational_pair
        else:
            result += 3

        # if debug:
        #     print("\n\n ~~ ")
        #     print("da:           ", da.bin())
        #     print("db:           ", db.bin())
        #     print("dc:           ", dc.bin())
        #     print("rhs:          ", rhs.bin())
        #     print("hw:           ", int(hw))
        #     print("bw_implies:   ", bitwise_implication(da[0] ^ db[0] ^ dc[0], core.Constant(0, 1)).bin())
        #     print("result:       ", result)
        #     print("\n\n")

        return result

    def num_frac_bits(self):
        return self.__class__.precision

    def error(self):
        """"Return the maximum difference between the weight and the exact weight.

            >>> # docstring for debugging purposes
            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.differential.difference import RXDiff
            >>> from arxpy.differential.derivative import RXDA
            >>> old_prec = RXDA.precision
            >>> errors = []
            >>> n = 16
            >>> for p in range(0, 12):
            ...     RXDA.precision = p
            ...     alpha = RXDiff(Constant(0, n)), RXDiff(Constant(0, n))
            ...     f = RXDA(alpha)
            ...     errors.append(round(f.error(), 4))
            >>> RXDA.precision = old_prec
            >>> errors
            [0.415, 0.415, 0.165, 0.04, 0.04, 0.0087, 0.0087, 0.0009, 0.0009, 0.0009, 0.0009, 0.0004]

        As opposed to `XDCA`, the exact weight in `RXDA` is bigger.
        """
        n = self.input_diff[0].val.width
        k = self.__class__.precision

        w_rotational_pair = -(math.log2((1 + 2**(1 - n) + 0.5 + 2 ** (-n))) - 2)

        w_rotational_pair_approx = int(core.Constant(
            int(self.__class__.decimal2bin(w_rotational_pair, k), base=2),
            k + 1
        )) * 1.0 / 2**k

        # w_rotational_pair_approx < w_rotational_pair
        theo_error = w_rotational_pair - w_rotational_pair_approx

        if n < 16:
            theo_error += 1.02  # offset found empirically for n=8

        return theo_error


class XDCA(Derivative):
    """Represent the derivative of addition by a constant w.r.t XOR differences.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.derivative import XDCA
        >>> alpha, cte = XorDiff(Constant(0, 4)), Constant(1, 4)
        >>> f = XDCA(alpha, cte)
        >>> x = Constant(0, 4)
        >>> f.eval(x)  # f(x + alpha) - f(x)
        XorDiff(0x0)
        >>> f.max_weight(), f.error(), f.num_frac_bits()
        (6, 0.0860713320559343, 1)

    The class attribute ``precision`` controls how many fraction bits
    are used to approximate the weight. Its default value is ``3``,
    and lower values lead to simpler formulas with higher errors.
    """

    diff_type = difference.XorDiff
    # op is a instance attribute
    precision = 3

    def __init__(self, input_diff, cte):  # noqa: 102
        assert cte != 0
        input_diff = _tuplify(input_diff)
        n = input_diff[0].val.width
        cte = core.bitvectify(cte, width=n)
        self.op = extraop.make_partial_operation(operation.BvAdd, tuple([None, cte]))
        self.op.constant = cte  # temporary patch
        super().__init__(input_diff)

        index_first_one = 0
        for i in range(0, cte.width):
            if cte[i] == 1:
                index_first_one = i
                break

        self._index_first_one = index_first_one
        self._effective_width = n - 1 - index_first_one
        self._effective_precision = max(min(type(self).precision, self._effective_width - 2), 0)  # 2 found empirically

    def __str__(self):
        return "{}_{}({})".format(type(self).__name__, self.op.constant, self.input_diff[0])

    __repr__ = __str__

    # noinspection PyPep8Naming
    def _is_possible(self, output_diff, debug=False):
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.op.constant
        n = a.width

        one = core.Constant(1, n)

        assert self._effective_width == n - 1

        def carry(x, y):
            # carry[i] = Maj(x[i-1], y[i-1], carry[i-1])
            return (x + y) ^ x ^ y

        # # S_0 well defined
        case11_ = (u << one) & (v << one)  # i-bit is True if S_i = 11*
        case00_ = (~(u << one)) & (~(v << one))  # i-bit is True if S_i = 00*
        case__1 = u ^ v
        a = a << one  # i-bit holds a[i-1]

        local_eq_a = ~(a ^ (a << one))  # i-bit is True if a[i-1] == a[i-2]
        case00_prev = case00_ << one  # i-bit is True if S_{i-1} = 00*

        c = carry(local_eq_a & case00_, (~case00_prev))

        case001 = case00_ & case__1
        bad_case11_ = case11_ & ~(case__1 ^ local_eq_a) & (c | ~case00_prev)

        bad_events = (case001 | bad_case11_)
        is_valid = operation.BvComp(bad_events, core.Constant(0, bad_events.width))

        if debug:
            print("\n\n ~~ ")
            print("u:           ", u.bin())
            print("v:           ", v.bin())
            print("a:           ", a.bin())
            print("case11_:     ", case11_.bin())
            print("case00_:     ", case00_.bin())
            print("case__1:     ", case__1.bin())
            print("case001:     ", case001.bin())
            print("case00_prev: ", case00_prev.bin())
            print("local_eq_a:  ", local_eq_a.bin())
            print("c:           ", c.bin())
            print("bad_case11_: ", bad_case11_.bin())
            print("bad_events:  ", bad_events.bin())
            print("is_valid    :", is_valid.bin())
            print("\n\n")

        return is_valid

    def is_possible(self, output_diff):
        """Return whether the given output `XorDiff` is possible.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.derivative import XDCA
            >>> alpha, cte = XorDiff(Constant(0, 4)), Constant(1, 4)
            >>> f = XDCA(alpha, cte)
            >>> f.is_possible(XorDiff(Constant(0, 4)))
            0b1
            >>> u, v = Variable("u", 4), Variable("v", 4)
            >>> f = XDCA(XorDiff(u), cte)
            >>> result = f.is_possible(XorDiff(v))
            >>> result  # doctest: +NORMALIZE_WHITESPACE
            0x0 == (((u << 0x1) & (v << 0x1) & ~(0x9 ^ u ^ v) & (~((~(u << 0x1) & ~(v << 0x1)) << 0x1) |
                   (~((~(u << 0x1) & ~(v << 0x1)) << 0x1) ^ ((0x9 & ~(u << 0x1) & ~(v << 0x1)) + ~((~(u << 0x1) &
                   ~(v << 0x1)) << 0x1)) ^ (0x9 & ~(u << 0x1) & ~(v << 0x1))))) | (~(u << 0x1) & ~(v << 0x1) & (u ^ v)))
            >>> result.xreplace({u: Constant(0, 4), v: Constant(0, 4)})
            0b1

        See `Derivative.is_possible` for more information.
        """
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.op.constant

        effective_width = self._effective_width
        index_one = self._index_first_one

        if effective_width == 0:
            return operation.BvComp(u, v)
        elif effective_width == 1:
            return operation.BvComp(u[index_one:], v[index_one:]) & \
                   operation.BvComp(~u[index_one] ^ (u[index_one+1] ^ v[index_one+1]), core.Constant(1, 1))
        else:
            if index_one == 0:
                c = core.Constant(1, 1)
            else:
                c = operation.BvComp(u[index_one-1:], v[index_one-1:])
            u = difference.XorDiff(u[:index_one])
            v = difference.XorDiff(v[:index_one])
            der = type(self)([u], a[:index_one])
            return c & der._is_possible(v)

    def _has_probability_one(self, output_diff):
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.op.constant
        n = a.width

        assert self._effective_width == n - 1

        one = core.Constant(1, n)

        def all_ones(width):
            return ~ core.Constant(0, width)

        case00_ = (~(u << one)) & (~(v << one))  # i-bit is True if S_i = 00*
        case__1 = u ^ v
        case000 = case00_ & (~case__1)

        return operation.BvComp(case000, all_ones(n))

    def has_probability_one(self, output_diff):
        """Return whether the input diff propagates to the output diff with probability one.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.derivative import XDCA
            >>> alpha, cte = XorDiff(Constant(0, 4)), Constant(1, 4)
            >>> f = XDCA(alpha, cte)
            >>> f.has_probability_one(XorDiff(Constant(0, 4)))
            0b1
            >>> u, v = Variable("u", 4), Variable("v", 4)
            >>> f = XDCA(XorDiff(u), cte)
            >>> result = f.has_probability_one(XorDiff(v))
            >>> result
            0xf == (~(u << 0x1) & ~(v << 0x1) & ~(u ^ v))
            >>> result.xreplace({u: Constant(0, 4), v: Constant(0, 4)})
            0b1

        """
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.op.constant

        effective_width = self._effective_width
        index_one = self._index_first_one

        if effective_width == 0:
            return operation.BvComp(u, v)
        elif effective_width == 1:
            return operation.BvComp(u[index_one:], v[index_one:]) & \
                   operation.BvComp(~u[index_one] ^ (u[index_one+1] ^ v[index_one+1]), core.Constant(1, 1))
        else:
            if index_one == 0:
                c = core.Constant(1, 1)
            else:
                c = operation.BvComp(u[index_one-1:], v[index_one-1:])
            u = difference.XorDiff(u[:index_one])
            v = difference.XorDiff(v[:index_one])
            der = type(self)([u], a[:index_one])
            return c & der._has_probability_one(v)

    # noinspection SpellCheckingInspection
    def weight(self, output_diff, prefix=None, debug=False, version=2):
        """Return the weight of a possible output `XorDiff`.

        If the output difference is symbolic, a pair
        ``(weight, assertions)`` is returned, where ``assertions`` is
        a tuple of equalities fixing some temporary variables.
        If the output difference is a constant value,
        only the value of the weight is returned.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.bitvector.context import NotEvaluation
            >>> from arxpy.bitvector.printing import BvWrapPrinter
            >>> from arxpy.bitvector.extraop import PopCount, PopCountSum2, PopCountSum3, PopCountDiff, Reverse, LeadingZeros
            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.derivative import XDCA
            >>> alpha, cte = XorDiff(Constant(0, 4)), Constant(1, 4)
            >>> f = XDCA(alpha, cte)
            >>> f.weight(XorDiff(Constant(0, 4)))
            0x0
            >>> alpha = XorDiff(Variable("u", 4))
            >>> f = XDCA(alpha, cte)
            >>> beta = XorDiff(Variable("v", 4))
            >>> with NotEvaluation([Reverse, PopCount, PopCountDiff, PopCountSum2, PopCountSum3, LeadingZeros]):
            ...     weight_value, assertions = f.weight(beta, prefix="")
            >>> print(BvWrapPrinter().doprint(weight_value))
            -(::(PopCountDiff((~_0lz & ((~u & ~v) << 0x1)) | ((u ^ v) << 0x1),
                              (_1rev & _4rev) ^ (0x1 + _1rev) ^ (0x1 + _1rev + (_1rev & _4rev)),
                 0b0,
              ::(0b0,
                 PopCount(&(_4rev & ((_1rev & _4rev) ^ (0x1 + _1rev) ^ (0x1 + _1rev + (_1rev & _4rev))),
                            ~(((_1rev & _4rev) ^ (0x1 + _1rev) ^ (0x1 + _1rev + (_1rev & _4rev))) << 0x1)
            >>> assertions  # doctest: +NORMALIZE_WHITESPACE
            [_0lz == LeadingZeros(~((~u & ~v) << 0x1)),
            _1rev == Reverse((~u & ~v) << 0x1),
            _2rev == Reverse(~(((~u & ~v) << 0x1) >> 0x1) & ((~u & ~v) << 0x1) & (~(0x2 ^ u ^ v) >> 0x1)),
            _3rev == Reverse(_2rev ^ _1rev ^ (_1rev + _2rev)),
            _4rev == Reverse((((0x1 & (((~u & ~v) << 0x1) >> 0x1)) + (0x2 & (((~u & ~v) << 0x1) >> 0x1) & ~((~u & ~v) << 0x1))) & ~(_3rev | (~(((~u & ~v) << 0x1) >> 0x1) & ((~u & ~v) << 0x1) & (~(0x2 ^ u ^ v) >> 0x1)))) | ((~(((~u & ~v) << 0x1) >> 0x1) & ((~u & ~v) << 0x1) & (~(0x2 ^ u ^ v) >> 0x1)) - (((0x1 & (((~u & ~v) << 0x1) >> 0x1)) + (0x2 & (((~u & ~v) << 0x1) >> 0x1) & ~((~u & ~v) << 0x1))) & (_3rev | (~(((~u & ~v) << 0x1) >> 0x1) & ((~u & ~v) << 0x1) & (~(0x2 ^ u ^ v) >> 0x1))))))]


        See `Derivative.weight` for more information.
        """
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.op.constant

        effective_width = self._effective_width
        index_one = self._index_first_one

        if effective_width <= 1:
            return core.Constant(0, 1)
        else:
            u = difference.XorDiff(u[:index_one])
            v = difference.XorDiff(v[:index_one])
            der = type(self)([u], a[:index_one])
            return der._weight(v, prefix, debug, version)

    def _weight(self, output_diff, prefix=None, debug=False, version=2):
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.op.constant
        n = a.width
        one = core.Constant(1, n)

        assert self._effective_width == n - 1

        assert version in [0, 1, 2]  # 0-reference, 1-w/o extra reverse, 2-s_000 and no HW2 in fr

        if prefix is None:
            prefix = "tmp" + str(abs(hash(u) + hash(v) + hash(a)))

        if isinstance(u, core.Constant) and isinstance(v, core.Constant):
            are_cte_differences = True
        else:
            self._i_auxvar = 0
            assertions = []
            are_cte_differences = False

        def rev(x):
            if are_cte_differences:
                return extraop.Reverse(x)
            else:
                aux = core.Variable("{}_{}rev".format(prefix, self._i_auxvar), x.width)
                self._i_auxvar += 1
                assertions.append(operation.BvComp(aux, extraop.Reverse(x)))
                return aux

        def lz(x):
            if are_cte_differences:
                return extraop.LeadingZeros(x)
            else:
                aux = core.Variable("{}_{}lz".format(prefix, self._i_auxvar), x.width)
                self._i_auxvar += 1
                assertions.append(operation.BvComp(aux, extraop.LeadingZeros(x)))
                return aux

        def carry(x, y):
            return (x + y) ^ x ^ y

        def rev_carry(x, y):
            return rev(carry(rev(x), rev(y)))

        if version in [0, 1]:
            s00_old = (~(u << one)) & (~(v << one))  # i-bit is True if S_{i} = 00*
        else:
            s00_old = ((~u) & (~v)) << one
        s00_ = s00_old & (~lz(~s00_old))  # if x is 001*...*, then lz(x) = 1100...0

        if version == 0:
            e_i1 = s00_ & (~ (s00_ >> one))  # e_{i-1}
            e_ili = ~s00_ & (s00_ >> one)  # e_{i-l_i}
        else:
            e_i1 = s00_old & (~ (s00_old >> one))  # e_{i-1}
            e_ili = ~s00_old & (s00_old >> one)  # e_{i-l_i}

        q = ~( (a << one) ^ (u ^ v) )  # q[i] = ~(a[i-1]^u[i]^v[i])
        q = ((q >> one ) & e_i1)  # q[i-1, i-3] = (a[i-1]^u[i]^v[i], 0, 0)

        if version == 0:
            s = ((a << one) & e_ili) + (a & (s00_ >> one))
        else:
            s = ((a << one) & e_ili) + (a & (s00_old >> one))

        if version == 0:
            d = rev_carry(s00_, q) | q
        else:
            rev_s00_old = rev(s00_old)
            d = rev(carry(rev_s00_old, rev(q))) | q

        w = (q - (s & d)) | (s & (~d))

        if version == 0:
            w = w << one
            h = rev_carry(s00_ << one, w & (s00_ << one))
        elif version == 1:
            rev_w = rev(w) >> one
            rev_h = carry( (rev_s00_old + one) >> one, rev_w & (rev(s00_)) >> one)
        else:
            rev_w = rev(w)
            rev_h = carry(rev_s00_old + one, rev_w & rev_s00_old)

        sbnegb = (u ^ v) << one  # i-bit is True if S_{i} = (b, \neg b, *)

        if version == 0:
            int = extraop.PopCountDiff(sbnegb | s00_, h)   # or hw(sbminb_) + (hw(s00_) - hw(h))
        else:
            int = extraop.PopCountDiff(sbnegb | s00_, rev_h)

        def smart_add(x, y):
            if x.width == y.width:
                return x + y
            elif x.width < y.width:
                return operation.ZeroExtend(x, y.width - x.width) + y
            else:
                return x + operation.ZeroExtend(y, x.width - y.width)

        def smart_sub(x, y):
            # cannot be replaced by smart_add(x, -y)
            if x.width == y.width:
                return x - y
            elif x.width < y.width:
                return operation.ZeroExtend(x, y.width - x.width) - y
            else:
                return x - operation.ZeroExtend(y, x.width - y.width)

        k = self._effective_precision

        if k == 0:
            int_frac = int
        elif k == 1:
            int = operation.Concat(int, core.Constant(0, 1))
            if version == 0:
                f1 = extraop.PopCount(w & h & (~(h >> one)))  # each one adds 2^(-1)
            else:
                f1 = extraop.PopCount(rev_w & rev_h & (~(rev_h << one)))
            int_frac = smart_sub(int, f1)
        else:
            two = core.Constant(2, n)
            three = core.Constant(3, n)
            four = core.Constant(4, n)

            if version == 0:
                f12 = extraop.PopCountSum2(
                    w & h & (~(h >> one)),
                    w & h & ((~(h >> one)) | (~(h >> two)) & (h >> one))
                )  # each one adds 2^(-2), that's why ~(h >> one) need to be counted twice
            elif version == 1:
                f12 = extraop.PopCountSum2(
                    rev_w & rev_h & (~(rev_h << one)),
                    rev_w & rev_h & ((~(rev_h << one)) | (~(rev_h << two)) & (rev_h << one))
                )
            else:
                f12 = extraop.PopCount(
                    # ( ( rev_w & rev_h & (~(rev_h << one)) ) >> one ) |
                    ( ( (rev_w & rev_h) >> one) & (~rev_h)  ) |
                    (rev_w & rev_h & ((~(rev_h << one)) | (~(rev_h << two)) & (rev_h << one)))
                )

            if k == 2:
                int = operation.Concat(int, core.Constant(0, 2))
                int_frac = smart_sub(int, f12)
            elif k == 3:
                # f3 cannot be included in f12, since ~(h >> one) would need to be counted 4 times
                if version == 0:
                    f3 = extraop.PopCount(w & h & (h >> one) & (h >> two) & (~(h >> three)))
                else:
                    f3 = extraop.PopCount(rev_w & rev_h & (rev_h << one) & (rev_h << two) & (~(rev_h << three)))
                int = operation.Concat(int, core.Constant(0, 3))
                f12 = operation.Concat(f12, core.Constant(0, 1))
                int_frac = smart_sub(int, smart_add(f12, f3))
            elif k == 4:
                if version == 0:
                    f34 = extraop.PopCountSum2(
                        w & h & (h >> one) & (h >> two) & (~(h >> three)),
                        w & h & (h >> one) & (h >> two) & ((~(h >> three)) | (~(h >> four) & (h >> three)))
                    )
                elif version == 1:
                    f34 = extraop.PopCountSum2(
                        rev_w & rev_h & (rev_h << one) & (rev_h << two) & (~(rev_h << three)),
                        rev_w & rev_h & (rev_h << one) & (rev_h << two) & ((~(rev_h << three)) | (~(rev_h << four) & (rev_h << three)))
                    )
                else:
                    f34 = extraop.PopCount(
                        # ( (rev_w & rev_h & (rev_h << one) & (rev_h << two) & (~(rev_h << three))) >> one ) |
                        ( ((rev_w & rev_h) >> one) & rev_h & (rev_h << one) & (~(rev_h << two))) |
                        (rev_w & rev_h & (rev_h << one) & (rev_h << two) & ((~(rev_h << three)) | (~(rev_h << four) & (rev_h << three))))
                    )
                int = operation.Concat(int, core.Constant(0, 4))
                f12 = operation.Concat(f12, core.Constant(0, 2))
                int_frac = smart_sub(int, smart_add(f12, f34))
            else:
                raise ValueError("precision must be between 0 and 4")

        if debug:
            print("\n\n ~~ ")
            print("u:            ", u.bin())
            print("v:            ", v.bin())
            print("a:            ", a.bin())
            print("s00_:         ", s00_.bin())
            print("e_i1:         ", e_i1.bin())
            print("e_ili1:       ", e_ili.bin())
            print("q:            ", q.bin())
            print("s:            ", s.bin())
            print("d:            ", d.bin())
            print("w:            ", w.bin())
            if version == 0:
                print("h:            ", h.bin())
            else:
                print("rev_w:        ", rev_w.bin())
                print("rev_h:        ", rev_h.bin())
            print("sbnegb:       ", sbnegb.bin())
            print("int:          ", int.bin())
            if k == 1:
                print("f1:           ", f1.bin())
            elif k > 1:
                print("f12:          ", f12.bin())
                if k == 3:
                    print("f3:           ", f3.bin())
                elif k == 4:
                    print("f34:          ", f34.bin())
            print("int_frac:     ", int_frac.bin())

        if are_cte_differences:
            return int_frac
        else:
            return int_frac, assertions

    def max_weight(self):
        effective_width = self._effective_width

        if effective_width <= 1:
            return 0
        else:
            return effective_width * (2 ** self._effective_precision)

    def exact_weight(self, output_diff):
        u = self.input_diff[0].val
        v = output_diff.val
        a = self.op.constant
        n = a.width

        assert isinstance(u, core.Constant) and isinstance(v, core.Constant)

        def probability2weight(pr):
            if pr == 0:
                return math.inf
            elif pr == 1:
                return 0  # avoid 0.0
            else:
                return abs(- math.log2(pr))

        if u[0] != v[0]:
            return probability2weight(0)

        probability = fractions.Fraction(1)
        delta = fractions.Fraction(0)
        for i in range(n - 1):
            if [u[i], v[i], u[i+1] ^ v[i+1]] == [0, 0, 0]:
                delta = fractions.Fraction(int(a[i]) + delta, 2)

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [0, 0, 1]:
                return probability2weight(0)

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [1, 1, 0]:
                probability *= 1 - (int(a[i]) + delta - (2 * int(a[i]) * delta))
                delta = int(a[i])

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [1, 1, 1]:
                probability *= int(a[i]) + delta - (2 * int(a[i]) * delta)
                delta = fractions.Fraction(1, 2)

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [0, 1, 0]:
                probability *= fractions.Fraction(1, 2)
                delta = int(a[i])

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [0, 1, 1]:
                probability *= fractions.Fraction(1, 2)

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [1, 0, 0]:
                probability *= fractions.Fraction(1, 2)
                delta = int(a[i])

            if [u[i], v[i], u[i+1] ^ v[i+1]] == [1, 0, 1]:
                probability *= fractions.Fraction(1, 2)

        return probability2weight(probability)

    def num_frac_bits(self):
        return self._effective_precision

    # noinspection SpellCheckingInspection
    @classmethod
    @functools.lru_cache()
    def _len2error(cls, n, precision, verbose=False):
        """Return a dictionary mapping difference lengths to their maximum error.

            >>> # docstring for debugging purposes
            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.derivative import XDCA
            >>> _ = XDCA._len2error(8, 4, True)
            chainlen2biterror: {3: 0.03, 5: 0.02}
            worst_chainlen: 3
            <BLANKLINE>
             > difflen: 4
             > decompositions: [[3]]
             > worst_decomp: [3]
             > difflen2differror[4]: 0.0860713320559343
             > difflen: 5
             > decompositions: [[5], [3]]
             > worst_decomp: [5]
             > difflen2differror[5]: 0.08607133205593431
             > difflen: 7
             > decompositions: [[3, 3], [3], [5]]
             > worst_decomp: [3, 3]
             > difflen2differror[7]: 0.1721426641118686
            <BLANKLINE>
            prime_lens: [3, 5]
            difflen2differror: [0, 0, 0, 0.09, 0.09, 0.09, 0.17, 0.17]
            <BLANKLINE>
            >>> [round(d, 2) for d in XDCA._len2error(8, 0)]
            [0, 0, 0, 1.0, 1.0, 1.0, 2.0, 2.0]
            >>> [round(d, 2) for d in XDCA._len2error(8, 1)]
            [0, 0, 0, 0.09, 0.58, 0.58, 0.58, 0.67]
            >>> [round(d, 2) for d in XDCA._len2error(8, 2)]
            [0, 0, 0, 0.09, 0.09, 0.33, 0.33, 0.33]
            >>> [round(d, 2) for d in XDCA._len2error(8, 3)]
            [0, 0, 0, 0.09, 0.09, 0.09, 0.21, 0.21]
            >>> [round(d, 2) for d in XDCA._len2error(8, 4)]
            [0, 0, 0, 0.09, 0.09, 0.09, 0.17, 0.17]

        """
        assert n <= 32

        def get_chainlen2biterror():
            def bit_error(chainlen):
                # given a chain of length l_i,
                # if pi != 2^{-i}, then p_i <= 2^{l_i - 1} - 1
                # and there are (at most) l_i - 2 bits right to the leftmost active bit.
                # (e.g. l_i = 4, p_i <= 7 = 0b111, two "fraction bits")
                k = precision

                if chainlen - 2 <= k:
                    # 0.086... is the error when all fraction bits are sued
                    return 0.0860713320559343 / chainlen

                t = [t_i * 2**(-k) for t_i in range(2**k)]  # t[i] = t_i * 2**(-k)
                chainerror = max(math.log2(1 + t_i + 2**(-k)) - t_i for t_i in t)

                return chainerror / chainlen

            d = {}
            worst_chainlen = 2
            worst_chainlen_error = 0

            for chainlen in range(3, n):
                # chainlen = l_i
                chainerror = bit_error(chainlen)

                if chainerror > worst_chainlen_error:
                    worst_chainlen_error = chainerror
                    worst_chainlen = chainlen
                else:
                    if ((chainlen // worst_chainlen) * worst_chainlen) * worst_chainlen_error >= chainlen * chainerror:
                        # assume chainlen = worst_chainlen * k
                        # if error(worst_chainlen) * k) >= error(chainlen), we ignore chainleb
                        continue

                d[chainlen] = chainerror

            return d

        chainlen2biterror = get_chainlen2biterror()

        worst_chainlen = max(chainlen2biterror.items(), key=operator.itemgetter(1))[0]

        def error_chainlen(chainlen):
            return chainlen2biterror[chainlen] * chainlen

        def error_decomposition(decomposition):
            error = 0
            for chainlen in decomposition:
                error += error_chainlen(chainlen)
            return error

        # a prime length correspond to a single chain of maximum error
        # (decomposing the chain in sub-chains do not increase the error)
        prime_lens = [3]
        if worst_chainlen not in prime_lens:
            prime_lens.append(worst_chainlen)

        def get_decompositions(diff_len):
            if diff_len in prime_lens:
                return [[diff_len]]  # trivial decomposition

            diff_decompositions = []
            for m in prime_lens:
                if m <= diff_len:
                    m_decompositions = get_decompositions(diff_len - m)
                    for decomp in m_decompositions:
                        diff_decompositions.append(sorted([m] + decomp))
                    else:
                        diff_decompositions.append([m])

            return diff_decompositions

        difflen2differror = [None for _ in range(n)]
        difflen2differror[:4] = [0, 0, 0, error_chainlen(3)]

        if verbose:
            print("chainlen2biterror:", {k: round(v, 2) for k, v in chainlen2biterror.items()})
            print("worst_chainlen:", worst_chainlen)
            # print("maxchain:", maxchain)
            print()

        for difflen in range(4, n):
            if difflen % worst_chainlen == 0:
                difflen2differror[difflen] = (difflen // worst_chainlen) * error_chainlen(worst_chainlen)
                continue

            decompositions = []

            if difflen in chainlen2biterror:
                decompositions.append([difflen])

            decompositions.extend(get_decompositions(difflen))
            worst_decomp = max(decompositions, key=error_decomposition)

            if worst_decomp[0] == difflen:
                prime_lens.append(difflen)

            # noinspection PyTypeChecker
            difflen2differror[difflen] = error_decomposition(worst_decomp)

            if verbose:
                print(" > difflen:", difflen)
                print(" > decompositions:", decompositions)
                print(" > worst_decomp:", worst_decomp)
                print(" > difflen2differror[{}]: {}".format(difflen, difflen2differror[difflen]))

        if verbose:
            print("\nprime_lens:", prime_lens)
            print("difflen2differror:", [round(d, 2) for d in difflen2differror])
            print()

        return difflen2differror

    def error(self):
        n = self.op.constant.width
        effective_width = self._effective_width

        len2error = type(self)._len2error(n, self._effective_precision)
        assert effective_width < len(len2error)
        return len2error[effective_width]

    def _replace_input_diff(self, new_input_diff, **kwargs):
        return super()._replace_input_diff(new_input_diff, cte=self.op.constant)
