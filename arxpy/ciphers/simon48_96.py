"""Simon 48/96: Simon block cipher with 48-bit block size and 96-bit key size."""
from arxpy.bitvector.operation import RotateRight
from arxpy.diffcrypt.itercipher import IterFunction, IterBlockCipher
from arxpy.ciphers.simon32_64 import F

default_rounds = 36
n = 24
m = 4
z = "10001110111110010011000010110101000111011111001001100001011010"


class KeySchedule(IterFunction):
    """Key schedule function of Simon48/96."""

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
    """Encryption function of Simon48/96."""

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


class Simon48_96(IterBlockCipher):
    """Simon 48/96 block cipher.

        >>> from arxpy.ciphers.simon48_96 import Simon48_96
        >>> key = (0x1a1918, 0x121110, 0x0a0908, 0x020100)
        >>> plaintext = (0x726963, 0x20646e)
        >>> Simon48_96(*plaintext, *key)
        (0x6e06a5, 0xacf156)

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
