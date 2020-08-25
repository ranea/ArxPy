"""Provide additional bit-vector operators."""
import functools

from arxpy.bitvector import core
from arxpy.bitvector import operation


def repeat_pattern(pattern, width):
    """Repeat the pattern until obtain a bit-vector of given width."""
    assert width % pattern.width == 0
    return operation.Repeat(pattern, width // pattern.width)


def pattern01(width):
    """Obtain the pattern 0...01...1 with given 0-width."""
    zeroes = core.Constant(0, width)
    return operation.Concat(zeroes, ~zeroes)


class PopCount(operation.Operation):
    """Count the number of 1's in the bit-vector.

    This operation is also known as the hamming weight of a bit-vector.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.extraop import PopCount
        >>> PopCount(Constant(0b1010, 4))
        0b010
        >>> PopCount(Constant(0b101, 3))
        0b10
        >>> PopCount(Variable("x", 4))  # doctest: +NORMALIZE_WHITESPACE
        ((0x3 & ((x - (0x5 & (x >> 0x1))) >> 0x2)) + (0x3 & (x - (0x5 & (x >> 0x1)))))[2:]
        >>> PopCount(Variable("x", 3))  # doctest: +NORMALIZE_WHITESPACE
        (0b0 :: (x[0])) + (0b0 :: (x[1])) + (0b0 :: (x[2]))

    """

    arity = [1, 0]
    is_symmetric = False

    @classmethod
    def output_width(cls, x):
        return x.width.bit_length()

    @classmethod
    def eval(cls, bv):
        # Source: Hacker's Delight

        if bv.width < 4:
            w = cls.output_width(bv)
            return sum([operation.ZeroExtend(bv[i], w-1) for i in range(bv.width)])

        # extend the bv until power of 2 length
        original_width = bv.width
        while (bv.width & (bv.width - 1)) != 0:
            bv = operation.ZeroExtend(bv, 1)
        width_log2 = bv.width.bit_length() - 1

        m_ctes = []
        for i in range(width_log2):
            m_ctes.append(repeat_pattern(pattern01(2 ** i), bv.width))

        if bv.width > 32:
            for i, m in enumerate(m_ctes):
                bv = (bv & m) + ((bv >> core.Constant(2 ** i, bv.width)) & m)
            return bv[original_width.bit_length() - 1:]

        for i, m in enumerate(m_ctes):
            if i == 0:
                bv = bv - ((bv >> core.Constant(1, bv.width)) & m)
            elif i == 1:
                bv = (bv & m) + ((bv >> core.Constant(2 ** i, bv.width)) & m)  # generic case
            elif i == 2:
                bv = (bv + (bv >> core.Constant(4, bv.width))) & m
            elif i == 3:
                bv = bv + (bv >> core.Constant(8, bv.width))
            elif i == 4:
                bv = bv + (bv >> core.Constant(16, bv.width))

        return bv[original_width.bit_length() - 1:]


class Reverse(operation.Operation):
    """Reverse the bit-vector.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.extraop import PopCount
        >>> Reverse(Constant(0b1010, 4))
        0x5
        >>> Reverse(Constant(0b001, 3))
        0b100
        >>> Reverse(Variable("x", 4))  # doctest: +NORMALIZE_WHITESPACE
        (0x3 & (((0x5 & (x >> 0x1)) | ((0x5 & x) << 0x1)) >> 0x2)) |
        ((0x3 & ((0x5 & (x >> 0x1)) | ((0x5 & x) << 0x1))) << 0x2)
        >>> Reverse(Variable("x", 3))  # doctest: +NORMALIZE_WHITESPACE
        (x[0]) :: (x[1]) :: (x[2])

    """

    arity = [1, 0]
    is_symmetric = False

    @classmethod
    def output_width(cls, x):
        return x.width

    @classmethod
    def eval(cls, bv):
        # Source: Hacker's Delight

        if bv.width == 1:
            return bv
        elif bv.width == 2:
            return operation.RotateLeft(bv, 1)
        elif bv.width == 3:
            return operation.Concat(operation.Concat(bv[0], bv[1]), bv[2])

        original_width = bv.width
        while (bv.width & (bv.width - 1)) != 0:
            bv = operation.ZeroExtend(bv, 1)
        width_log2 = bv.width.bit_length() - 1

        m_ctes = []
        for i in range(width_log2):
            m_ctes.append(repeat_pattern(pattern01(2 ** i), bv.width))

        if bv.width > 32:
            for i, m in list(enumerate(m_ctes)):
                bv = ((bv & m) << core.Constant(2 ** i, bv.width)) | ((bv >> core.Constant(2 ** i, bv.width)) & m)
            return bv[:bv.width - original_width]

        for i, m in list(enumerate(m_ctes))[:3]:
            bv = ((bv & m) << core.Constant(2 ** i, bv.width)) | ((bv >> core.Constant(2 ** i, bv.width)) & m)  # generic case

        if len(m_ctes) == 4:
            bv = ((bv & m_ctes[3]) << core.Constant(8, bv.width)) | ((bv >> core.Constant(8, bv.width)) & m_ctes[3])
        elif len(m_ctes) == 5:
            rol = operation.RotateLeft
            ror = operation.RotateRight
            bv = ror(bv & m_ctes[3], 8) | (rol(bv, 8) & m_ctes[3])

        return bv[:bv.width - original_width]


class PopCountSum2(operation.Operation):
    """Count the number of 1's of two bit-vectors.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.extraop import PopCountSum2
        >>> PopCountSum2(Constant(0b000, 3), Constant(0b001, 3))
        0b001
        >>> PopCountSum2(~Constant(0, 15), ~Constant(0, 15))
        0b11110
        >>> PopCountSum2(Variable("x", 4), Variable("y", 4))  # doctest: +NORMALIZE_WHITESPACE
        (0x3 & ((x - (0x5 & (x >> 0x1))) >> 0x2)) + (0x3 & (x - (0x5 & (x >> 0x1)))) +
        (0x3 & ((y - (0x5 & (y >> 0x1))) >> 0x2)) + (0x3 & (y - (0x5 & (y >> 0x1))))
        >>> PopCountSum2(Variable("x", 3), Variable("y", 3))  # doctest: +NORMALIZE_WHITESPACE
        (0b0 :: ((0b0 :: (x[0])) + (0b0 :: (x[1])) + (0b0 :: (x[2])))) +
        (0b0 :: ((0b0 :: (y[0])) + (0b0 :: (y[1])) + (0b0 :: (y[2]))))

    """

    arity = [2, 0]
    is_symmetric = True
    is_simple = True

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        if x.width < 4:
            return PopCount.output_width(x) + 1
        else:
            return (x.width + y.width).bit_length()

    @classmethod
    def eval(cls, x, y):
        # Source: Hacker's Delight

        if x.width < 4:
            # the HW of a 1-bit/2-bit vector requires 1-bit/2-bit (HW(0b1)=0b1, HW(0b11)=0b10)
            # thus, the sum of two HW of these sizes require an extra bit
            # the HW of a 3-bit vector requires 2-bit (Hw(0b111)=0b11)
            # and the sum of two HW of 3-bit also require an extra bit
            return operation.ZeroExtend(PopCount(x), 1) + operation.ZeroExtend(PopCount(y), 1)
        elif x.width > 32:
            width = cls.output_width(x, y)
            x = PopCount(x)
            x = operation.ZeroExtend(x, width - x.width)
            y = PopCount(y)
            y = operation.ZeroExtend(y, width - y.width)
            return x + y

        orig_x, orig_y = x, y
        while (x.width & (x.width - 1)) != 0:
            x = operation.ZeroExtend(x, 1)
        while (y.width & (y.width - 1)) != 0:
            y = operation.ZeroExtend(y, 1)
        width_log2 = x.width.bit_length() - 1

        m_ctes = []
        for i in range(width_log2):
            m_ctes.append(repeat_pattern(pattern01(2 ** i), x.width))

        bv = core.Constant(0, x.width)
        for i, m in enumerate(m_ctes):
            if i == 0:
                x = x - ((x >> core.Constant(1, bv.width)) & m)
                y = y - ((y >> core.Constant(1, bv.width)) & m)
                bv = x + y
            elif i == 1:
                x = (x & m) + ((x >> core.Constant(2 ** i, bv.width)) & m)  # generic case
                y = (y & m) + ((y >> core.Constant(2 ** i, bv.width)) & m)
                bv = x + y
            elif i == 2:
                bv = (bv & m) + ((bv >> core.Constant(4, bv.width)) & m)
            elif i == 3:
                bv = bv + (bv >> core.Constant(8, bv.width))
            elif i == 4:
                bv = bv + (bv >> core.Constant(16, bv.width))

        return bv[cls.output_width(orig_x, orig_y) - 1:]


class PopCountSum3(operation.Operation):
    """Count the number of 1's of three bit-vectors.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.extraop import PopCountSum3
        >>> PopCountSum3(Constant(0b000, 3), Constant(0b001, 3), Constant(0b011, 3))
        0x3
        >>> PopCountSum3(Variable("x", 4), Variable("y", 4), Variable("z", 4))  # doctest: +NORMALIZE_WHITESPACE
        (0x3 & ((x - (0x5 & (x >> 0x1))) >> 0x2)) + (0x3 & (x - (0x5 & (x >> 0x1)))) +
        (0x3 & ((y - (0x5 & (y >> 0x1))) >> 0x2)) + (0x3 & (y - (0x5 & (y >> 0x1)))) +
        (0x3 & ((z - (0x5 & (z >> 0x1))) >> 0x2)) + (0x3 & (z - (0x5 & (z >> 0x1))))
        >>> PopCountSum3(Variable("x", 3), Variable("y", 3), Constant(0, 3))  # doctest: +NORMALIZE_WHITESPACE
        (0b00 :: ((0b0 :: (x[0])) + (0b0 :: (x[1])) + (0b0 :: (x[2])))) +
        (0b00 :: ((0b0 :: (y[0])) + (0b0 :: (y[1])) + (0b0 :: (y[2]))))

    """

    arity = [3, 0]
    is_symmetric = True
    is_simple = True

    @classmethod
    def condition(cls, x, y, z):
        return x.width == y.width == z.width

    @classmethod
    def output_width(cls, x, y, z):
        if x.width < 4:
            offset = 1
            if x.width == 3:
                offset = 2
            return PopCount.output_width(x) + offset
        else:
            return (x.width + y.width + z.width).bit_length()

    @classmethod
    def eval(cls, x, y, z):
        # Source: Hacker's Delight

        if x.width < 4:
            # the HW of a 1-bit/2-bit vector requires 1-bit/2-bit (HW(0b1)=0b1, HW(0b11)=0b10)
            # thus, the sum of three HW of these sizes require an extra bit (3*0b1=0b11, 3*0b10=0b110)
            # the HW of a 3-bit vector requires 2-bit (Hw(0b111)=0b11)
            # but the sum of three HW of 3-bit require two extra bit (3*0b11 = 0b1001)
            offset = 1
            if x.width == 3:
                offset = 2
            x = operation.ZeroExtend(PopCount(x), offset)
            y = operation.ZeroExtend(PopCount(y), offset)
            z = operation.ZeroExtend(PopCount(z), offset)
            return x + y + z
        elif x.width > 32:
            width = cls.output_width(x, y, z)
            x = PopCount(x)
            x = operation.ZeroExtend(x, width - x.width)
            y = PopCount(y)
            y = operation.ZeroExtend(y, width - y.width)
            z = PopCount(z)
            z = operation.ZeroExtend(z, width - z.width)
            return x + y + z

        orig_x, orig_y, orig_z = x, y, z
        while (x.width & (x.width - 1)) != 0:
            x = operation.ZeroExtend(x, 1)
        while (y.width & (y.width - 1)) != 0:
            y = operation.ZeroExtend(y, 1)
        while (z.width & (z.width - 1)) != 0:
            z = operation.ZeroExtend(z, 1)
        width_log2 = x.width.bit_length() - 1

        m_ctes = []
        for i in range(width_log2):
            m_ctes.append(repeat_pattern(pattern01(2 ** i), x.width))

        bv = core.Constant(0, x.width)
        for i, m in enumerate(m_ctes):
            if i == 0:
                x = x - ((x >> core.Constant(1, bv.width)) & m)
                y = y - ((y >> core.Constant(1, bv.width)) & m)
                z = z - ((z >> core.Constant(1, bv.width)) & m)
                bv = x + y + z
            elif i == 1:
                x = (x & m) + ((x >> core.Constant(2 ** i, bv.width)) & m)  # generic case
                y = (y & m) + ((y >> core.Constant(2 ** i, bv.width)) & m)
                z = (z & m) + ((z >> core.Constant(2 ** i, bv.width)) & m)
                bv = x + y + z
            elif i == 2:
                bv = (bv & m) + ((bv >> core.Constant(4, bv.width)) & m)
            elif i == 3:
                bv = bv + (bv >> core.Constant(8, bv.width))
            elif i == 4:
                bv = bv + (bv >> core.Constant(16, bv.width))

        return bv[cls.output_width(orig_x, orig_y, orig_z) - 1:]


class PopCountDiff(operation.Operation):
    """Compute the difference of the hamming weight of two words

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.extraop import PopCountDiff
        >>> PopCountDiff(Constant(0b011, 3), Constant(0b100, 3))
        0b01
        >>> PopCountDiff(Variable("x", 4), Variable("y", 4))  # doctest: +NORMALIZE_WHITESPACE
        (((0x3 & ((x - (0x5 & (x >> 0x1))) >> 0x2)) + (0x3 & (x - (0x5 & (x >> 0x1)))) +
        (0x3 & ((~y - (0x5 & (~y >> 0x1))) >> 0x2)) + (0x3 & (~y - (0x5 & (~y >> 0x1))))) - 0x4)[2:]
        >>> PopCountDiff(Variable("x", 3), Variable("y", 3))  # doctest: +NORMALIZE_WHITESPACE
        ((0b0 :: (x[0])) + (0b0 :: (x[1])) + (0b0 :: (x[2]))) -
        ((0b0 :: (y[0])) + (0b0 :: (y[1])) + (0b0 :: (y[2])))

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width.bit_length()

    @classmethod
    def eval(cls, x, y):
        # Source: Hacker's Delight

        if x.width < 4:
            return PopCount(x) - PopCount(y)
        elif x.width > 32:
            return PopCount(x) - PopCount(y)

        orig_x, orig_y = x, y
        while (x.width & (x.width - 1)) != 0:
            x = operation.ZeroExtend(x, 1)
        while (y.width & (y.width - 1)) != 0:
            y = operation.ZeroExtend(y, 1)
        width_log2 = x.width.bit_length() - 1

        m_ctes = []
        for i in range(width_log2):
            m_ctes.append(repeat_pattern(pattern01(2 ** i), x.width))

        bv = core.Constant(0, x.width)
        for i, m in enumerate(m_ctes):
            if i == 0:
                x = x - ((x >> core.Constant(1, bv.width)) & m)
                y = (~y) - (((~y) >> core.Constant(1, bv.width)) & m)
                bv = x + y
            elif i == 1:
                x = (x & m) + ((x >> core.Constant(2 ** i, bv.width)) & m)  # generic case
                y = (y & m) + ((y >> core.Constant(2 ** i, bv.width)) & m)
                bv = x + y
            elif i == 2:
                bv = (bv & m) + ((bv >> core.Constant(4, bv.width)) & m)
            elif i == 3:
                bv = bv + (bv >> core.Constant(8, bv.width))
            elif i == 4:
                bv = bv + (bv >> core.Constant(16, bv.width))

        return (bv - y.width)[cls.output_width(orig_x, orig_y) - 1:]


class LeadingZeros(operation.Operation):
    """Return a bit-vector with the leading zeros set to one and the rest to zero.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.extraop import PopCount
        >>> LeadingZeros(Constant(0b11111, 5))
        0b00000
        >>> LeadingZeros(Constant(0b001, 3))
        0b110
        >>> LeadingZeros(Variable("x", 4))  # doctest: +NORMALIZE_WHITESPACE
        ~(((x | (x >> 0x1)) >> 0x2) | x | (x >> 0x1))
        >>> LeadingZeros(Variable("x", 3))  # doctest: +NORMALIZE_WHITESPACE
        ~((((((0b0 :: x) >> 0x1) | (0b0 :: x)) >> 0x2) | ((0b0 :: x) >> 0x1) | (0b0 :: x))[2:])

    """

    arity = [1, 0]
    is_symmetric = False

    @classmethod
    def output_width(cls, x):
        return x.width

    @classmethod
    def eval(cls, bv):
        # Source: Hacker's Delight

        if bv.width == 1:
            return ~bv

        original_width = bv.width
        while (bv.width & (bv.width - 1)) != 0:
            bv = operation.ZeroExtend(bv, 1)
        width_log2 = bv.width.bit_length() - 1

        for i in range(width_log2):
            bv = bv | (bv >> core.Constant(2 ** i, bv.width))
        return ~bv[original_width - 1:]


class PartialOperation(operation.Operation):
    """Represent the bit-vector operations with fixed arguments."""
    pass


# temporary hack to create singletons
@functools.lru_cache(maxsize=None)
def make_partial_operation(bv_op, fixed_args):
    """Return a new operation based on the given one but with fixed arguments.

    The argument ``fixed_args`` is a `tuple` with the same length as the
    number of operands of the given function, containing ``None`` or `Constant` elements.
    If ``fixed_args[i]`` is ``None``, the i-th operand is not fixed;
    otherwise, the i-th operand is replaced with ``fixed_args[i]``.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvAdd
        >>> BvCteAdd = make_partial_operation(BvAdd, tuple([None, Constant(1, 4)]))
        >>> BvCteAdd(Variable("a", 4))
        BvAdd(·, 0x1)(a)
        >>> BvCteAdd(Constant(1, 4))
        0x2
        >>> assert BvCteAdd == make_partial_operation(BvAdd, tuple([None, Constant(1, 4)]))

    """
    assert isinstance(fixed_args, tuple)
    assert len(fixed_args) == sum(bv_op.arity)
    assert any(arg is not None for arg in fixed_args), "{} {}".format(bv_op, fixed_args)
    assert any(arg is None for arg in fixed_args), "{} {}".format(bv_op, fixed_args)

    if hasattr(bv_op, "operand_types"):
        operand_types = bv_op.operand_types
    else:
        operand_types = [core.Term for _ in range(len(fixed_args))]

    num_terms_fixed = 0
    num_scalars_fixed = 0
    free_operand_types = []
    fixed_args_str = []
    for arg, type_arg in zip(fixed_args, operand_types):
        if arg is None:
            free_operand_types.append(type_arg)
            fixed_args_str.append("·")
            continue

        assert isinstance(arg, type_arg)
        if type_arg == int:
            num_scalars_fixed += 1
        elif isinstance(arg, core.Term):
            num_terms_fixed += 1
        else:
            assert False
        fixed_args_str.append(str(arg))

    _fixed_args = fixed_args

    # subclassing operation may introduce side effects (but disable simplifications)
    class MyPartialOperation(PartialOperation):
        arity = [bv_op.arity[0] - num_terms_fixed, bv_op.arity[1] - num_scalars_fixed]
        is_symmetric = bv_op.is_symmetric
        is_simple = bv_op.is_simple and sum(arity) > 1
        operand_types = free_operand_types
        alt_name = "{}({})".format(bv_op.__name__, ', '.join(fixed_args_str))

        base_op = bv_op
        fixed_args = _fixed_args

        @classmethod
        def _get_full_args(cls, *args):
            full_args = []
            index = 0
            for fixed_arg in cls.fixed_args:
                if fixed_arg is None:
                    full_args.append(args[index])
                    index += 1
                else:
                    full_args.append(fixed_arg)
            return full_args

        @classmethod
        def condition(cls, *args):
            return cls.base_op.condition(*cls._get_full_args(*args))

        @classmethod
        def output_width(cls, *args):
            return cls.base_op.output_width(*cls._get_full_args(*args))

        @classmethod
        def eval(cls, *args):
            if all(isinstance(a, (core.Constant, int)) for a in args):
                return cls.base_op.eval(*cls._get_full_args(*args))

    MyPartialOperation.__name__ = "{}({})".format(bv_op.__name__, ', '.join(fixed_args_str))

    return MyPartialOperation
