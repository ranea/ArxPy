"""Simon 32/64: Simon block cipher with 32-bit block size and 64-bit key size."""
from math import gcd

from arxpy.bitvector.core import Constant, tuplify
from arxpy.bitvector.operation import (
    Operation, RotateLeft, RotateRight, BvComp, Ite, ZeroExtend
)

from arxpy.diffcrypt.difference import XorDiff, DiffVar, RXDiff
from arxpy.diffcrypt.differential import Differential, _HammingWeight
from arxpy.diffcrypt.itercipher import IterFunction, IterBlockCipher


class F(Operation):
    """The function F of the round function of Simon 32/64.

    The F function of of Simon 32/64 corresponds to
    ((x <<< a) & (x <<< b) ^ (x <<< c),
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
            return (RotateLeft(x, cls.a) & RotateLeft(x, cls.b)) ^ RotateLeft(x, cls.c)

    @classmethod
    def differential(cls, diff_type):
        assert diff_type in [XorDiff, RXDiff]
        if diff_type == XorDiff:
            return XDF
        elif diff_type == RXDiff:
            return RXDF


class KeySchedule(IterFunction):
    """Key schedule function of Simon32/64."""

    rounds = 32
    input_widths = [16, 16, 16, 16]
    output_widths = [16 for i in range(32)]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.output_widths = [16 for i in range(new_rounds)]

    @classmethod
    def eval(cls, k3, k2, k1, k0):
        if cls.rounds <= 4:
            return [k0, k1, k2, k3][:cls.rounds]

        k = [None for i in range(cls.rounds)]
        k[:4] = [k0, k1, k2, k3]
        m = 4
        z = "11111010001001010110000111001101111101000100101011000011100110"

        for i in range(4, cls.rounds):
            tmp = RotateRight(k[i - 1], 3) ^ k[i - 3]
            tmp ^= RotateRight(tmp, 1)
            k[i] = ~k[i - m] ^ tmp ^ int(z[(i - m) % 62]) ^ 3

        return k


class Encryption(IterFunction):
    """Encryption function of Simon32/64."""

    rounds = 32
    input_widths = [16, 16] + [16 for i in range(32)]
    output_widths = [16, 16]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.input_widths = [16, 16] + [16 for i in range(new_rounds)]

    @classmethod
    def eval(cls, x, y, *round_keys):
        for i in range(cls.rounds):
            x, y = (y ^ F(x) ^ round_keys[i], x)

        return x, y


class Simon32_64(IterBlockCipher):
    """Simon 32/64 block cipher.

        >>> from arxpy.ciphers.simon32_64 import Simon32_64
        >>> plaintext = (0x6565, 0x6877)
        >>> key = (0x1918, 0x1110, 0x0908, 0x0100)
        >>> Simon32_64(*plaintext, *key)
        (0xc69b, 0xe9bb)

    """

    rounds = 32
    input_widths = [16, 16] + [16, 16, 16, 16]
    output_widths = [16, 16]
    inner_func = KeySchedule
    outer_func = Encryption

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.inner_func.set_rounds(new_rounds)
        cls.outer_func.set_rounds(new_rounds)


class XDF(Differential):
    """XOR differential of the F operation."""

    short_name = "xdpF"
    diff_type = XorDiff
    op = F

    def __init__(self, input_diff, output_diff):
        """Initialize the differential."""
        assert all(isinstance(d, DiffVar) for d in tuplify(input_diff))
        assert isinstance(output_diff, DiffVar)
        super().__init__(input_diff, output_diff)

    def is_valid(self):
        """Return the bv expression for non-zero propagation probability.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import DiffVar
            >>> from arxpy.ciphers.simon32_64 import XDF
            >>> x, y = DiffVar("x", 16), DiffVar("y", 16)
            >>> d = XDF(x, y)
            >>> d.is_valid()  # doctest: +ELLIPSIS
            Ite(0xffff == x, 0b0 == (((0x00ff & ((0x0f0f & ((0x3333 ...
            >>> zero = Constant(0, 16)
            >>> d.is_valid().xreplace({x: zero, y: zero})
            0b1

        """
        a, b, c = self.op.a, self.op.b, self.op.c
        n = self.input_diff[0].width
        assert gcd(n, a - b) == 1 and a > b and n % 2 == 0

        alpha = self.input_diff[0]
        beta = self.output_diff
        BvRol = RotateLeft
        varibits = BvRol(alpha, a) | BvRol(alpha, b)
        doublebits = BvRol(alpha, b) & (~BvRol(alpha, a)) & BvRol(alpha, 2 * a - b)
        gamma = beta ^ BvRol(alpha, c)

        def is_even(x):
            return BvComp(x[0], Constant(0, 1))

        case2 = BvComp(
            Constant(0, n),
            (gamma & (~varibits)) | ((gamma ^ BvRol(gamma, a - b)) & doublebits)
        )

        condition = Ite(
            BvComp(alpha, ~Constant(0, n)),
            is_even(_HammingWeight(gamma)),
            case2
        )

        return condition

    def weight(self):
        """Return the weight of the differential.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import DiffVar
            >>> from arxpy.ciphers.simon32_64 import XDF
            >>> x, y = DiffVar("x", 16), DiffVar("y", 16)
            >>> d = XDF(x, y)
            >>> d.weight()  # doctest: +ELLIPSIS
            Ite(0xffff == x, 0b01111, ((0x00ff & ((0x0f0f & ((0x3333 ...
            >>> zero = Constant(0, 16)
            >>> d.weight().xreplace({x: zero, y: zero})
            0b00000

        """
        a, b = self.op.a, self.op.b
        n = self.input_diff[0].width

        alpha = self.input_diff[0]
        BvRol = RotateLeft
        varibits = BvRol(alpha, a) | BvRol(alpha, b)
        doublebits = BvRol(alpha, b) & (~BvRol(alpha, a)) & BvRol(alpha, 2 * a - b)

        hw = _HammingWeight(varibits ^ doublebits)
        width = max((n - 1).bit_length(), hw.width)

        value = Ite(
            BvComp(alpha, ~Constant(0, n)),
            Constant(n - 1, width),
            ZeroExtend(hw, width - hw.width)
        )

        return value


class RXDF(Differential):
    """RX differential of the F operation."""

    short_name = "rxdpF"
    diff_type = RXDiff
    op = F

    def __init__(self, input_diff, output_diff):
        """Initialize the differential."""
        assert all(isinstance(d, DiffVar) for d in tuplify(input_diff))
        assert isinstance(output_diff, DiffVar)
        super().__init__(input_diff, output_diff)

    def is_valid(self):
        """Return the bv expression for non-zero propagation probability.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import DiffVar
            >>> from arxpy.ciphers.simon32_64 import RXDF
            >>> x, y = DiffVar("x", 16), DiffVar("y", 16)
            >>> d = RXDF(x, y)
            >>> d.is_valid()  # doctest: +ELLIPSIS
            Ite(0xffff == (x <<< 1), 0b0 == (((0x00ff & ((0x0f0f & ((0x3333 ...
            >>> zero = Constant(0, 16)
            >>> d.is_valid().xreplace({x: zero, y: zero})
            0b1

        """
        width = self.input_diff[0].width
        x = DiffVar("x", width)
        y = DiffVar("y", width)

        return self.op.differential(XorDiff)(x, y).is_valid().xreplace({
            x: RotateLeft(self.input_diff[0], 1),
            y: RotateLeft(self.output_diff, 1)
        })

    def weight(self):
        """Return the weight of the differential.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.diffcrypt.difference import DiffVar
            >>> from arxpy.ciphers.simon32_64 import RXDF
            >>> x, y = DiffVar("x", 16), DiffVar("y", 16)
            >>> d = RXDF(x, y)
            >>> d.weight()  # doctest: +ELLIPSIS
            Ite(0xffff == (x <<< 1), 0b01111, ((0x00ff & ((0x0f0f & ((0x3333 ...
            >>> zero = Constant(0, 16)
            >>> d.weight().xreplace({x: zero, y: zero})
            0b00000

        """
        width = self.input_diff[0].width
        x = DiffVar("x", width)
        y = DiffVar("y", width)

        return self.op.differential(XorDiff)(x, y).weight().xreplace({
            x: RotateLeft(self.input_diff[0], 1),
            y: RotateLeft(self.output_diff, 1)
        })
