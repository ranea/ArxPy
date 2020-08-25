"""Hight cipher."""
from arxpy.bitvector.core import Constant
from arxpy.bitvector.operation import RotateLeft as ROL

from arxpy.primitives.primitives import KeySchedule, Encryption, Cipher


class HightKeySchedule(KeySchedule):
    """Key schedule function."""

    rounds = 32
    input_widths = [8 for _ in range(16)]
    output_widths = [8 for _ in range(8 + 4 * 32)]

    @classmethod
    def set_rounds(cls, new_rounds):
        assert new_rounds >= 2
        cls.rounds = new_rounds
        cls.output_widths = [8 for _ in range(8 + 4 * cls.rounds)]

    @classmethod
    def eval(cls, *mk):
        mk = list(reversed(mk))  # mk[i] = mki

        wk = [None for _ in range(8)]
        for i in range(8):
            if i <= 3:
                wk[i] = mk[i + 12]
            else:
                wk[i] = mk[i - 4]

        d = [
            0x5a, 0x6d, 0x36, 0x1b, 0x0d, 0x06, 0x03, 0x41,
            0x60, 0x30, 0x18, 0x4c, 0x66, 0x33, 0x59, 0x2c,
            0x56, 0x2b, 0x15, 0x4a, 0x65, 0x72, 0x39, 0x1c,
            0x4e, 0x67, 0x73, 0x79, 0x3c, 0x5e, 0x6f, 0x37,
            0x5b, 0x2d, 0x16, 0x0b, 0x05, 0x42, 0x21, 0x50,
            0x28, 0x54, 0x2a, 0x55, 0x6a, 0x75, 0x7a, 0x7d,
            0x3e, 0x5f, 0x2f, 0x17, 0x4b, 0x25, 0x52, 0x29,
            0x14, 0x0a, 0x45, 0x62, 0x31, 0x58, 0x6c, 0x76,
            0x3b, 0x1d, 0x0e, 0x47, 0x63, 0x71, 0x78, 0x7c,
            0x7e, 0x7f, 0x3f, 0x1f, 0x0f, 0x07, 0x43, 0x61,
            0x70, 0x38, 0x5c, 0x6e, 0x77, 0x7b, 0x3d, 0x1e,
            0x4f, 0x27, 0x53, 0x69, 0x34, 0x1a, 0x4d, 0x26,
            0x13, 0x49, 0x24, 0x12, 0x09, 0x04, 0x02, 0x01,
            0x40, 0x20, 0x10, 0x08, 0x44, 0x22, 0x11, 0x48,
            0x64, 0x32, 0x19, 0x0c, 0x46, 0x23, 0x51, 0x68,
            0x74, 0x3a, 0x5d, 0x2e, 0x57, 0x6b, 0x35, 0x5a
        ]
        d = [Constant(d_i, 8) for d_i in d]
        sk = [None for _ in range(4*cls.rounds)]
        for i in range(8):
            for j in range(8):
                if 16 * i + j >= 4*cls.rounds:
                    continue
                sk[16 * i + j] = mk[(j - i) % 8] + d[16 * i + j]

            for j in range(8):
                if 16 * i + j + 8 >= 4*cls.rounds:
                    continue
                sk[16 * i + j + 8] = mk[((j - i) % 8) + 8] + d[16 * i + j + 8]

        return wk + sk


class HightEncryption(Encryption):
    """Encryption function."""

    rounds = 32
    input_widths = [8 for _ in range(8)]
    output_widths = [8 for _ in range(8)]
    round_keys = None

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds

    @classmethod
    def eval(cls, *p):  # p7,...,p0
        wk = cls.round_keys[:8]
        sk = cls.round_keys[8:]

        # initial transformation
        x = list(reversed(p))
        x[0] += wk[0]
        x[2] ^= wk[1]
        x[4] += wk[2]
        x[6] ^= wk[3]

        def f0(bv):
            return ROL(bv, 1) ^ ROL(bv, 2) ^ ROL(bv, 7)

        def f1(bv):
            return ROL(bv, 3) ^ ROL(bv, 4) ^ ROL(bv, 6)

        y = [None for _ in range(8)]

        for i in range(cls.rounds - 1):
            y[0] = x[7] ^ (f0(x[6]) + sk[4 * i + 3])
            y[1] = x[0]
            y[2] = x[1] + (f1(x[0]) ^ sk[4 * i])
            y[3] = x[2]
            y[4] = x[3] ^ (f0(x[2]) + sk[4 * i + 1])
            y[5] = x[4]
            y[6] = x[5] + (f1(x[4]) ^ sk[4 * i + 2])
            y[7] = x[6]

            x = y[:]

        y[0] = x[0]
        y[1] = x[1] + (f1(x[0]) ^ sk[-4])
        y[2] = x[2]
        y[3] = x[3] ^ (f0(x[2]) + sk[-3])
        y[4] = x[4]
        y[5] = x[5] + (f1(x[4]) ^ sk[-2])
        y[6] = x[6]
        y[7] = x[7] ^ (f0(x[6]) + sk[-1])

        x = y[:]

        # final transformation
        c = x[:]
        if cls.rounds == 32:
            c[0] = x[0] + wk[4]
            c[2] = x[2] ^ wk[5]
            c[4] = x[4] + wk[6]
            c[6] = x[6] ^ wk[7]

        return list(reversed(c))


class HightCipher(Cipher):
    key_schedule = HightKeySchedule
    encryption = HightEncryption
    rounds = 32

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.key_schedule.set_rounds(new_rounds)
        cls.encryption.set_rounds(new_rounds)

    @classmethod
    def test(cls):
        """Test Hight with official test vectors."""
        # https://tools.ietf.org/html/draft-kisa-hight-00#section-5
        old_rounds = cls.rounds
        cls.set_rounds(32)

        plaintext = (0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00)
        key = (0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
               0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff)
        assert cls(plaintext, key) == (0x00, 0xf4, 0x18, 0xae, 0xd9, 0x4f, 0x03, 0xf2)

        plaintext = (0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77)
        key = (0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00)
        assert cls(plaintext, key) == (0x23, 0xce, 0x9f, 0x72, 0xe5, 0x43, 0xe6, 0xd8)

        cls.set_rounds(old_rounds)
