"""Simon family of block ciphers."""
import enum
import math

from arxpy.bitvector.core import Constant
from arxpy.bitvector.operation import (
    Operation, RotateLeft, RotateRight, BvComp, Ite, ZeroExtend
)
from arxpy.bitvector.extraop import PopCount

from arxpy.differential.derivative import Derivative
from arxpy.differential.difference import XorDiff

from arxpy.primitives.primitives import KeySchedule, Encryption, Cipher


class SimonRF(Operation):
    """The non-linear part of the round function of Simon.

    This corresponds to ((x <<< a) & (x <<< b)) ^ (x <<< c),
    where (a, b, c) = (8, 1, 2).
    """

    a = 8
    b = 1
    c = 2

    arity = [1, 0]
    is_symmetric = False

    @classmethod
    def output_width(cls, x):
        return x.width

    @classmethod
    def eval(cls, x):
        if isinstance(x, Constant):
            return (RotateLeft(x, cls.a) &
                    RotateLeft(x, cls.b)) ^ RotateLeft(x, cls.c)

    @classmethod
    def xor_derivative(cls, input_diff):
        return XDSimonRF(input_diff)


class XDSimonRF(Derivative):
    """Represent the derivative of SimonRF w.r.t XOR differences.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.differential.difference import XorDiff
        >>> alpha = XorDiff(Constant(0, 16))
        >>> f = XDSimonRF(alpha)
        >>> x = Constant(0, 16)
        >>> f.eval(x)  # f(x + alpha) - f(x)
        XorDiff(0x0000)
        >>> f.max_weight(), f.error(), f.num_frac_bits()
        (15, 0, 0)

    """

    diff_type = XorDiff
    op = SimonRF

    def is_possible(self, output_diff):
        """Return whether the given output `XorDiff` is possible.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.bitvector.context import NotEvaluation
            >>> from arxpy.bitvector.printing import BvWrapPrinter
            >>> from arxpy.differential.difference import XorDiff
            >>> alpha = XorDiff(Constant(0, 16))
            >>> f = XDSimonRF(alpha)
            >>> f.is_possible(XorDiff(Constant(0, 16)))
            0b1
            >>> u, v = Variable("u", 16), Variable("v", 16)
            >>> f = XDSimonRF(XorDiff(u))
            >>> with NotEvaluation([PopCount]):
            ...     result = f.is_possible(XorDiff(v))
            >>> print(BvWrapPrinter().doprint(result))
            Ite(0xffff == u,
                0b0 == (PopCount(v ^ (u <<< 2))[0]),
                ==(0x0000,
                   |(~(u <<< 8) & (u <<< 1) & (u <<< 15) & (v ^ (u <<< 2) ^ ((v ^ (u <<< 2)) <<< 7)),
                     ~((u <<< 1) | (u <<< 8)) & (v ^ (u <<< 2))
            >>> result.xreplace({u: Constant(0, 16), v: Constant(0, 16)})
            0b1

        See `Derivative.is_possible` for more information.
        """
        a, b, c = self.op.a, self.op.b, self.op.c
        n = self.input_diff[0].val.width
        assert math.gcd(n, a - b) == 1 and a > b and n % 2 == 0

        alpha = self.input_diff[0].val
        Rol = RotateLeft
        varibits = Rol(alpha, a) | Rol(alpha, b)
        r = (2 * a - b) % n
        doublebits = (Rol(alpha, b) & (~Rol(alpha, a)) & Rol(alpha, r))

        beta = output_diff.val
        gamma = beta ^ Rol(alpha, c)

        def is_even(x):
            return BvComp(x[0], Constant(0, 1))

        case2 = BvComp(
            Constant(0, n),
            (gamma & (~varibits)) | ((gamma ^ Rol(gamma, a - b)) & doublebits)
        )

        condition = Ite(
            BvComp(alpha, ~Constant(0, n)),
            is_even(PopCount(gamma)),
            case2
        )

        return condition

    def has_probability_one(self, output_diff):
        """Return whether the input diff propagates to the output diff with probability one.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.differential.difference import XorDiff
            >>> alpha = XorDiff(Constant(0, 16))
            >>> f = XDSimonRF(alpha)
            >>> f.has_probability_one(XorDiff(Constant(0, 16)))
            0b1

        """
        a, b, c = self.op.a, self.op.b, self.op.c
        n = self.input_diff[0].val.width
        assert math.gcd(n, a - b) == 1 and a > b and n % 2 == 0

        alpha = self.input_diff[0].val
        Rol = RotateLeft
        varibits = Rol(alpha, a) | Rol(alpha, b)
        r = (2 * a - b) % n
        doublebits = (Rol(alpha, b) & (~Rol(alpha, a)) & Rol(alpha, r))

        beta = output_diff.val
        gamma = beta ^ Rol(alpha, c)

        case2 = BvComp(
            Constant(0, n),
            (gamma & (~varibits)) | ((gamma ^ Rol(gamma, a - b)) & doublebits)
        )

        hw = varibits ^ doublebits  # no need to PopCount

        condition = ~BvComp(alpha, ~Constant(0, n))
        condition &= case2
        condition &= BvComp(hw, Constant(0, hw.width))

        return condition

    def weight(self, output_diff):
        """Return the weight of a possible output `XorDiff`.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> from arxpy.bitvector.context import NotEvaluation
            >>> from arxpy.bitvector.printing import BvWrapPrinter
            >>> from arxpy.bitvector.extraop import PopCount
            >>> from arxpy.differential.difference import XorDiff
            >>> alpha = XorDiff(Constant(0, 16))
            >>> f = XDSimonRF(alpha)
            >>> f.weight(XorDiff(Constant(0, 16)))
            0b00000
            >>> alpha = XorDiff(Variable("u", 16))
            >>> f = XDSimonRF(alpha)
            >>> beta = XorDiff(Variable("v", 16))
            >>> with NotEvaluation([PopCount]):
            ...     print(BvWrapPrinter().doprint(f.weight(beta)))
            Ite(0xffff == u,
                0b01111,
                PopCount((~(u <<< 8) & (u <<< 1) & (u <<< 15)) ^ ((u <<< 1) | (u <<< 8)))

        See `Derivative.weight` for more information.
        """
        a, b = self.op.a, self.op.b
        n = self.input_diff[0].val.width

        alpha = self.input_diff[0].val
        Rol = RotateLeft
        varibits = Rol(alpha, a) | Rol(alpha, b)
        r = (2 * a - b) % n
        doublebits = (Rol(alpha, b) & (~Rol(alpha, a)) & Rol(alpha, r))

        hw = PopCount(varibits ^ doublebits)
        width = max((n - 1).bit_length(), hw.width)

        value = Ite(
            BvComp(alpha, ~Constant(0, n)),
            Constant(n - 1, width),
            ZeroExtend(hw, width - hw.width)
        )

        return value

    def max_weight(self):
        width = self.input_diff[0].val.width - 1  # n - 1
        return width  # as an integer

    def exact_weight(self, output_diff):
        return int(self.weight(output_diff))

    def num_frac_bits(self):
        return 0

    def error(self):
        return 0


class SimonInstance(enum.Enum):
    simon_32_64 = enum.auto()
    simon_48_96 = enum.auto()
    simon_64_128 = enum.auto()


def get_Simon_instance(simon_instance):
    """Return an instance of the Simon family."""
    if simon_instance == SimonInstance.simon_32_64:
        default_rounds = 32
        n = 16
        m = 4
        z = "11111010001001010110000111001101111101000100101011000011100110"
    elif simon_instance == SimonInstance.simon_48_96:
        default_rounds = 36
        n = 24
        m = 4
        z = "10001110111110010011000010110101000111011111001001100001011010"
    elif simon_instance == SimonInstance.simon_64_128:
        default_rounds = 44
        n = 32
        m = 4
        z = "11011011101011000110010111100000010010001010011100110100001111"
    else:
        raise ValueError("invalid instance of Simon")

    class SimonKeySchedule(KeySchedule):
        """Key schedule function."""

        rounds = default_rounds
        input_widths = [n for _ in range(m)]
        output_widths = [n for _ in range(default_rounds)]

        @classmethod
        def set_rounds(cls, new_rounds):
            cls.rounds = new_rounds
            cls.input_widths = [n for _ in range(min(m, new_rounds))]
            cls.output_widths = [n for _ in range(new_rounds)]

        @classmethod
        def eval(cls, *master_key):
            if cls.rounds <= m:
                return list(reversed(master_key))[:cls.rounds]

            k = [None for i in range(cls.rounds)]
            k[:m] = list(reversed(master_key))

            for i in range(m, cls.rounds):
                tmp = RotateRight(k[i - 1], 3)
                if m == 4:
                    tmp ^= k[i - 3]
                tmp ^= RotateRight(tmp, 1)
                k[i] = ~k[i - m] ^ tmp ^ int(z[(i - m) % 62]) ^ 3

            return k

    class SimonEncryption(Encryption):
        """Encryption function."""

        rounds = default_rounds
        input_widths = [n, n]
        output_widths = [n, n]
        round_keys = None

        @classmethod
        def set_rounds(cls, new_rounds):
            cls.rounds = new_rounds

        @classmethod
        def eval(cls, x, y):
            for i in range(cls.rounds):
                x, y = (y ^ SimonRF(x) ^ cls.round_keys[i], x)
            return x, y

    class SimonCipher(Cipher):
        key_schedule = SimonKeySchedule
        encryption = SimonEncryption
        rounds = default_rounds
        _simon_instance = simon_instance

        @classmethod
        def set_rounds(cls, new_rounds):
            # assert new_rounds >= 2
            cls.rounds = new_rounds
            cls.key_schedule.set_rounds(new_rounds)
            cls.encryption.set_rounds(new_rounds)

        @classmethod
        def test(cls):
            old_rounds = cls.rounds
            cls.set_rounds(default_rounds)

            if cls._simon_instance == SimonInstance.simon_32_64:
                plaintext = (0x6565, 0x6877)
                key = (0x1918, 0x1110, 0x0908, 0x0100)
                assert cls(plaintext, key) == (0xc69b, 0xe9bb)
            elif cls._simon_instance == SimonInstance.simon_48_96:
                plaintext = (0x726963, 0x20646e)
                key = (0x1a1918, 0x121110, 0x0a0908, 0x020100)
                assert cls(plaintext, key) == (0x6e06a5, 0xacf156)
            elif cls._simon_instance == SimonInstance.simon_64_128:
                plaintext = (0x656b696c, 0x20646e75)
                key = (0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100)
                assert cls(plaintext, key) == (0x44c8fc20, 0xb9dfa07a)
            else:
                raise ValueError("invalid instance of Simon")

            cls.set_rounds(old_rounds)

    return SimonCipher
