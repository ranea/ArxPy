"""Threefish 256-bit cipher."""
import itertools

from arxpy.bitvector.core import Constant
from arxpy.bitvector.operation import RotateLeft as ROL

from arxpy.primitives.primitives import KeySchedule, Encryption, Cipher


FIXED_TWEAK = True


class ThreefishFixedTweakKeySchedule(KeySchedule):
    """Key schedule function."""

    rounds = 72 // 4
    input_widths = [64, 64, 64, 64]
    output_widths = [64 for i in range(4 * (72 // 4 + 1))]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.output_widths = [64 for _ in range(4 * (new_rounds + 1))]

    @classmethod
    def eval(cls, k0, k1, k2, k3):
        nw = 4
        nr_div4 = cls.rounds

        round_keys = [[None for _ in range(nw)] for _ in range(nr_div4 + 1)]
        c240 = Constant(0x1BD11BDAA9FC1A22, 64)

        k = [k0, k1, k2, k3, c240 ^ k0 ^ k1 ^ k2 ^ k3]
        t = [Constant(0, k0.width), Constant(0, k0.width), Constant(0, k0.width)]

        for s in range(nr_div4 + 1):
            round_keys[s][0] = k[s % (nw + 1)]
            round_keys[s][1] = k[(s + 1) % (nw + 1)] + t[s % 3]
            round_keys[s][2] = k[(s + 2) % (nw + 1)] + t[(s + 1) % 3]
            round_keys[s][3] = k[(s + 3) % (nw + 1)] + Constant(s, 64)

        return tuple(itertools.chain(*round_keys))


class ThreefishKeySchedule(KeySchedule):
    """Key schedule function."""

    rounds = 72 // 4
    input_widths = [64, 64, 64, 64, 64, 64]
    output_widths = [64 for i in range(4 * (72 // 4 + 1))]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.output_widths = [64 for _ in range(4 * (new_rounds + 1))]

    @classmethod
    def eval(cls, k0, k1, k2, k3, t0, t1):
        nw = 4
        nr_div4 = cls.rounds

        round_keys = [[None for _ in range(nw)] for _ in range(nr_div4 + 1)]
        c240 = Constant(0x1BD11BDAA9FC1A22, 64)

        k = [k0, k1, k2, k3, c240 ^ k0 ^ k1 ^ k2 ^ k3]
        t2 = t0 ^ t1
        t = [t0, t1, t2]

        for s in range(nr_div4 + 1):
            round_keys[s][0] = k[s % (nw + 1)]
            round_keys[s][1] = k[(s + 1) % (nw + 1)] + t[s % 3]
            round_keys[s][2] = k[(s + 2) % (nw + 1)] + t[(s + 1) % 3]
            round_keys[s][3] = k[(s + 3) % (nw + 1)] + Constant(s, 64)

        return tuple(itertools.chain(*round_keys))


class ThreefishEncryption(Encryption):
    """Encryption function."""

    rounds = 72 // 4
    input_widths = [64, 64, 64, 64]
    output_widths = [64, 64, 64, 64]
    round_keys = None
    R = [
        [14, 16],
        [52, 57],
        [23, 40],
        [5, 37],
        [25, 33],
        [46, 12],
        [58, 22],
        [32, 32],
    ]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds

    @classmethod
    def mix(cls, x0, x1, d, j):
        y0 = x0 + x1
        y1 = ROL(x1, cls.R[d % 8][j]) ^ y0
        return y0, y1

    @classmethod
    def pi(cls, i):
        return [0, 3, 2, 1][i]

    @classmethod
    def eval(cls, p0, p1, p2, p3):
        v = [p0, p1, p2, p3]
        e = [None, None, None, None]
        f = [None, None, None, None]

        nw = 4
        nr_div4 = cls.rounds

        k = [[None for _ in range(nw)] for _ in range(nr_div4 + 1)]
        for i, rk in enumerate(cls.round_keys):
            k[i // nw][i % nw] = rk

        for d in range(nr_div4 * 4):
            for i in range(nw):
                if d % 4 == 0:
                    e[i] = v[i] + k[d // 4][i]
                else:
                    e[i] = v[i]

            for j in range(2):
                f[2 * j: 2 * j + 2] = cls.mix(e[2 * j], e[2 * j + 1], d, j)

            for i in range(nw):
                v[i] = f[cls.pi(i)]

        c = [None, None, None, None]
        for i in range(nw):
            c[i] = v[i] + k[nr_div4][i]

        return c


class ThreefishCipher(Cipher):
    key_schedule = ThreefishFixedTweakKeySchedule if FIXED_TWEAK else ThreefishKeySchedule
    encryption = ThreefishEncryption
    rounds = 72 // 4

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.key_schedule.set_rounds(new_rounds)
        cls.encryption.set_rounds(new_rounds)

    @classmethod
    def test(cls):
        """Test Threefish with official test vectors."""
        old_rounds = cls.rounds
        cls.set_rounds(72 // 4)

        if not FIXED_TWEAK:
            key = [0, 0, 0, 0] + [0, 0]
        else:
            key = [0, 0, 0, 0]
        plaintext = [0, 0, 0, 0]
        ciphertext = (0x94eeea8b1f2ada84, 0xadf103313eae6670, 0x952419a1f4b16d53, 0xd83f13e63c9f6b11)

        assert cls(plaintext, key) == ciphertext

        if not FIXED_TWEAK:
            key = [0x1716151413121110, 0x1F1E1D1C1B1A1918, 0x2726252423222120, 0x2F2E2D2C2B2A2928]
            tweak = [0x0706050403020100, 0x0F0E0D0C0B0A0908]
            plaintext = [0xF8F9FAFBFCFDFEFF, 0xF0F1F2F3F4F5F6F7, 0xE8E9EAEBECEDEEEF, 0xE0E1E2E3E4E5E6E7]
            ciphertext = (0xDF8FEA0EFF91D0E0, 0xD50AD82EE69281C9, 0x76F48D58085D869D, 0xDF975E95B5567065)

            assert cls(plaintext, key + tweak) == ciphertext

        cls.set_rounds(old_rounds)
