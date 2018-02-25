"""Simon 48/72: Simon block cipher with 48-bit block size and 72-bit key size."""
from arxpy.bitvector.operation import RotateRight
from arxpy.diffcrypt.itercipher import IterFunction, IterBlockCipher
from arxpy.ciphers.simon32_64 import F

default_rounds = 36
n = 24
m = 3
z = "11111010001001010110000111001101111101000100101011000011100110"


class KeySchedule(IterFunction):
    """Key schedule function of Simon48/72."""

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
    """Encryption function of Simon48/72."""

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


class Simon48_72(IterBlockCipher):
    """Simon 48/72 block cipher.

        >>> from arxpy.ciphers.simon48_72 import Simon48_72
        >>> key = (0x121110, 0x0a0908, 0x020100)
        >>> plaintext = (0x612067, 0x6e696c)
        >>> Simon48_72(*plaintext, *key)
        (0xdae5ac, 0x292cac)

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
