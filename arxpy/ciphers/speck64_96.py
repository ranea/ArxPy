"""Speck64/96: Speck block cipher with 64-bit block size and 96-bit key size."""

from arxpy.bitvector.operation import RotateLeft, RotateRight

from arxpy.diffcrypt.itercipher import IterFunction, IterBlockCipher

default_rounds = 26
n = 32
m = 3
alpha = 8
beta = 3


def rf(x, y, k):
    """Round function of Speck64/96."""
    x = (RotateRight(x, alpha) + y) ^ k
    y = RotateLeft(y, beta) ^ x
    return x, y


class KeySchedule(IterFunction):
    """Key schedule function of Speck64/96."""

    rounds = default_rounds
    input_widths = [n for i in range(m)]
    output_widths = [n for i in range(default_rounds)]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.output_widths = [n for i in range(new_rounds)]

    @classmethod
    def eval(cls, *master_key):
        assert len(master_key) == m
        round_keys = [None for i in range(cls.rounds)]
        round_keys[0] = master_key[-1]
        l_values = list(reversed(master_key[:-1]))
        l_values.append(None)

        for i in range(cls.rounds - 1):
            result = rf(l_values[i % len(l_values)], round_keys[i], i)
            l_values[(i + m - 1) % len(l_values)], round_keys[i + 1] = result

        return round_keys


class Encryption(IterFunction):
    """Encryption function of Speck64/96."""

    rounds = default_rounds
    input_widths = [n, n] + [n for i in range(default_rounds)]
    output_widths = [n, n]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.input_widths = [n, n] + [n for i in range(new_rounds)]

    @classmethod
    def eval(cls, x, y, *round_keys):
        for i in range(cls.rounds):
            x, y = rf(x, y, round_keys[i])

        return x, y


class Speck64_96(IterBlockCipher):
    """Speck64/96 block cipher.

        >>> from arxpy.ciphers.speck64_96 import Speck64_96
        >>> key = (0x13121110, 0x0b0a0908, 0x03020100)
        >>> plaintext = (0x74614620, 0x736e6165)
        >>> Speck64_96(*plaintext, *key)
        (0x9f7952ec, 0x4175946c)

    """

    rounds = default_rounds
    input_widths = [n, n] + [n for i in range(m)]
    output_widths = [n, n]
    inner_func = KeySchedule
    outer_func = Encryption

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.inner_func.set_rounds(new_rounds)
        cls.outer_func.set_rounds(new_rounds)
