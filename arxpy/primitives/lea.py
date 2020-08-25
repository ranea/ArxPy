"""Lea 128-bit cipher."""
import itertools

from arxpy.bitvector.core import Constant
from arxpy.bitvector.operation import RotateLeft as ROL
from arxpy.bitvector.operation import RotateRight as ROR

from arxpy.primitives.primitives import KeySchedule, Encryption, Cipher


class LeaKeySchedule(KeySchedule):
    """Key schedule function."""

    rounds = 24
    input_widths = [32, 32, 32, 32]
    output_widths = [32 for i in range(24 * 6)]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.output_widths = [32 for _ in range(new_rounds * 6)]

    # noinspection PyPep8Naming
    @classmethod
    def eval(cls, k0, k1, k2, k3):
        round_keys = [None for _ in range(cls.rounds)]
        T0, T1, T2, T3 = k0, k1, k2, k3

        delta = [
            Constant(0xc3efe9db, 32), Constant(0x44626b02, 32),
            Constant(0x79e27c8a, 32), Constant(0x78df30ec, 32),
        ]

        for i in range(cls.rounds):
            T0 = ROL(T0 + ROL(delta[i % 4], i), 1)
            T1 = ROL(T1 + ROL(delta[i % 4], (i + 1)), 3)
            T2 = ROL(T2 + ROL(delta[i % 4], (i + 2)), 6)
            T3 = ROL(T3 + ROL(delta[i % 4], (i + 3)), 11)
            round_keys[i] = T0, T1, T2, T1, T3, T1

        return tuple(itertools.chain(*round_keys))  # flatten


class LeaEncryption(Encryption):
    """Encryption function."""

    rounds = 24
    input_widths = [32, 32, 32, 32]
    output_widths = [32, 32, 32, 32]
    round_keys = None

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds

    @classmethod
    def eval(cls, p0, p1, p2, p3):
        x0, x1, x2, x3 = p0, p1, p2, p3

        rk = [[None for _ in range(6)] for _ in range(cls.rounds)]
        for i, k in enumerate(cls.round_keys):
            rk[i // 6][i % 6] = k

        for i in range(cls.rounds):
            k0, k1, k2, k3, k4, k5 = rk[i]
            tmp = x0
            x0 = ROL((x0 ^ k0) + (x1 ^ k1), 9)
            x1 = ROR((x1 ^ k2) + (x2 ^ k3), 5)
            x2 = ROR((x2 ^ k4) + (x3 ^ k5), 3)
            x3 = tmp

        return x0, x1, x2, x3


class LeaCipher(Cipher):
    key_schedule = LeaKeySchedule
    encryption = LeaEncryption
    rounds = 24

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.key_schedule.set_rounds(new_rounds)
        cls.encryption.set_rounds(new_rounds)

    @classmethod
    def test(cls):
        old_rounds = cls.rounds
        cls.set_rounds(24)
        key = [
            Constant(0x3c2d1e0f, 32),
            Constant(0x78695a4b, 32),
            Constant(0xb4a59687, 32),
            Constant(0xf0e1d2c3, 32)
        ]
        pt = [
            Constant(0x13121110, 32),
            Constant(0x17161514, 32),
            Constant(0x1b1a1918, 32),
            Constant(0x1f1e1d1c, 32)
        ]
        ct = [
            Constant(0x354ec89f, 32),
            Constant(0x18c6c628, 32),
            Constant(0xa7c73255, 32),
            Constant(0xfd8b6404, 32)
        ]
        assert cls(pt, key) == tuple(ct)
        cls.set_rounds(old_rounds)
