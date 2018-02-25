"""Speck32/64: Speck block cipher with 32-bit block size and 64-bit key size."""

from arxpy.bitvector.operation import RotateLeft, RotateRight

from arxpy.diffcrypt.itercipher import IterFunction, IterBlockCipher


def rf(x, y, k):
    """Round function of Speck32/64."""
    x = (RotateRight(x, 7) + y) ^ k
    y = RotateLeft(y, 2) ^ x
    return x, y


class KeySchedule(IterFunction):
    """Key schedule function of Speck32/64."""

    rounds = 22
    input_widths = [16, 16, 16, 16]
    output_widths = [16 for i in range(22)]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.output_widths = [16 for i in range(new_rounds)]

    @classmethod
    def eval(cls, l2, l1, l0, k0):
        round_keys = [None for i in range(cls.rounds)]
        round_keys[0] = k0
        l_values = [l0, l1, l2, None]

        for i in range(cls.rounds - 1):
            result = rf(l_values[i % 4], round_keys[i], i)
            l_values[(i + 3) % 4], round_keys[i + 1] = result

        return round_keys


class Encryption(IterFunction):
    """Encryption function of Speck32/64."""

    rounds = 22
    input_widths = [16, 16] + [16 for i in range(22)]
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


class Speck32_64(IterBlockCipher):
    """Speck32/64 block cipher.

        >>> from arxpy.ciphers.speck32_64 import Speck32_64
        >>> plaintext = (0x6574, 0x694c)
        >>> key = (0x1918, 0x1110, 0x0908, 0x0100)
        >>> Speck32_64(*plaintext, *key)
        (0xa868, 0x42f2)

    """

    rounds = 22
    input_widths = [16, 16] + [16, 16, 16, 16]
    output_widths = [16, 16]
    inner_func = KeySchedule
    outer_func = Encryption

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.inner_func.set_rounds(new_rounds)
        cls.outer_func.set_rounds(new_rounds)
