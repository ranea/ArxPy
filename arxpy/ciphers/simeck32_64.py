"""Simeck 32/64: Simeck block cipher with 32-bit block size and 64-bit key size."""
from arxpy.diffcrypt.difference import XorDiff, RXDiff
from arxpy.diffcrypt.itercipher import IterFunction, IterBlockCipher
from arxpy.ciphers.simon32_64 import F, XDF, RXDF


class SkF(F):
    """The function F of the round function of Simeck 32/64.

    The F function of Simeck 32/64 corresponds to
    ((x <<< a) & (x <<< b) ^ (x <<< c),
    where (a, b, c) = (5, 0, 1).
    """

    a = 5
    b = 0
    c = 1

    @classmethod
    def differential(cls, diff_type):
        assert diff_type in [XorDiff, RXDiff]
        if diff_type == XorDiff:
            return SkXDF
        elif diff_type == RXDiff:
            return SkRXDF


def rf(x, y, k):
    """The round function of Simeck32/64."""
    return y ^ SkF(x) ^ k, x


class KeySchedule(IterFunction):
    """Key schedule function of Simeck32/64."""

    rounds = 32
    input_widths = [16, 16, 16, 16]
    output_widths = [16 for i in range(32)]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.output_widths = [16 for i in range(new_rounds)]

    @classmethod
    def eval(cls, t2, t1, t0, k0):
        k = [None for i in range(cls.rounds)]
        k[0] = k0
        t = [t0, t1, t2, None]

        z = list(reversed("10011010010000101011101100011111"))  # 0x9A42BB1F

        for i in range(cls.rounds - 1):
            C = (2 ** 16 - 4) ^ int(z[i])
            result = rf(t[i % 4], k[i], C)
            t[(i + 3) % 4], k[i + 1] = result

        return k


class Encryption(IterFunction):
    """Encryption function of Simeck32/64."""

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
            x, y = rf(x, y, round_keys[i])

        return x, y


class Simeck32_64(IterBlockCipher):
    """Simeck 32/64 block cipher.

        >>> from arxpy.ciphers.simeck32_64 import Simeck32_64
        >>> plaintext = (0x6565, 0x6877)
        >>> key = (0x1918, 0x1110, 0x0908, 0x0100)
        >>> Simeck32_64(*plaintext, *key)
        (0x770d, 0x2c76)

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


class SkXDF(XDF):
    """XOR differential of the F operation."""

    short_name = "xdpSkF"
    op = SkF


class SkRXDF(RXDF):
    """RX differential of the F operation."""

    short_name = "rxdpSkF"
    op = SkF
