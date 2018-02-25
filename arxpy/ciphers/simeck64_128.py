"""Simeck 64/128: Simeck block cipher with 64-bit block size and 128-bit key size."""
from arxpy.diffcrypt.itercipher import IterFunction, IterBlockCipher
from arxpy.ciphers.simeck32_64 import rf

default_rounds = 44
n = 32
m = 4
z = list(reversed(bin(0x938BCA3083F)[2:]))


class KeySchedule(IterFunction):
    """Key schedule function of Simeck64/128."""

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

        k = [None for i in range(cls.rounds)]
        k[0] = master_key[-1]
        t = list(reversed(master_key[:-1]))
        t.append(None)

        for i in range(cls.rounds - 1):
            C = (2 ** n - 4) ^ int(z[i % len(z)])
            result = rf(t[i % 4], k[i], C)
            t[(i + 3) % 4], k[i + 1] = result

        return k


class Encryption(IterFunction):
    """Encryption function of Simeck64/128."""

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


class Simeck64_128(IterBlockCipher):
    """Simeck 64/128 block cipher.

        >>> from arxpy.ciphers.simeck64_128 import Simeck64_128
        >>> key = (0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100)
        >>> plaintext = (0x656b696c, 0x20646e75)
        >>> Simeck64_128(*plaintext, *key)
        (0x45ce6902, 0x5f7ab7ed)

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
