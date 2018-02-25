"""Simon 64/128: Simon block cipher with 64-bit block size and 128-bit key size."""
from arxpy.bitvector.operation import RotateRight
from arxpy.diffcrypt.itercipher import IterFunction, IterBlockCipher
from arxpy.ciphers.simon32_64 import F

default_rounds = 44
n = 32
m = 4
z = "11011011101011000110010111100000010010001010011100110100001111"


class KeySchedule(IterFunction):
    """Key schedule function of Simon64/128."""

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


class Encryption(IterFunction):
    """Encryption function of Simon64/128."""

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
            x, y = (y ^ F(x) ^ round_keys[i], x)

        return x, y


class Simon64_128(IterBlockCipher):
    """Simon 64/128 block cipher.

        >>> from arxpy.ciphers.simon64_128 import Simon64_128
        >>> key = (0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100)
        >>> plaintext = (0x656b696c, 0x20646e75)
        >>> Simon64_128(*plaintext, *key)
        (0x44c8fc20, 0xb9dfa07a)

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
