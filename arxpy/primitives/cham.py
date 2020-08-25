"""Cham family of block ciphers."""
import enum

from arxpy.bitvector.core import Constant
from arxpy.bitvector.operation import RotateLeft, RotateRight

from arxpy.primitives.primitives import KeySchedule, Encryption, Cipher


class ChamInstance(enum.Enum):
    cham_64_128 = enum.auto()
    cham_128_128 = enum.auto()
    cham_128_256 = enum.auto()


def get_Cham_instance(cham_instance):
    """Return an instance of the Cham family."""
    if cham_instance == ChamInstance.cham_64_128:
        default_rounds = 80
        n = 64
        kappa = 128
        w = 16
    elif cham_instance == ChamInstance.cham_128_128:
        default_rounds = 80
        n = 128
        kappa = 128
        w = 32
    elif cham_instance == ChamInstance.cham_128_256:
        default_rounds = 96
        n = 128
        kappa = 256
        w = 32
    else:
        raise ValueError("invalid instance of Cham")


    class ChamKeySchedule(KeySchedule):
        """Key schedule function."""

        rounds = default_rounds
        input_widths = [w for _ in range(kappa // w)]
        output_widths = [w for _ in range(2 * kappa // w)]

        @classmethod
        def set_rounds(cls, new_rounds):
            # assert new_rounds >= kappa // w
            cls.rounds = new_rounds
            cls.input_widths = [w for _ in range(kappa // w)]
            cls.output_widths = [w for _ in range(min(new_rounds, 2 * kappa // w))]

        @classmethod
        def eval(cls, *master_key):
            k = master_key

            num_rk = min(cls.rounds, 2 * kappa // w)
            rk = [None for _ in range(num_rk)]

            for i in range(kappa // w):
                if i < num_rk:
                    rk[i] = k[i] ^ RotateLeft(k[i], 1) ^ RotateLeft(k[i], 8)
                if ((i + (kappa // w)) ^ 1) < num_rk:
                    rk[(i + (kappa // w)) ^ 1] = k[i] ^ RotateLeft(k[i], 1) ^ RotateLeft(k[i], 11)

            return rk

    class ChamEncryption(Encryption):
        """Encryption function."""

        rounds = default_rounds
        input_widths = [w for _ in range(n // w)]
        output_widths = [w for _ in range(n // w)]
        round_keys = None

        @classmethod
        def set_rounds(cls, new_rounds):
            cls.rounds = new_rounds

        @classmethod
        def eval(cls, p0, p1, p2, p3):
            x = [p0, p1, p2, p3]

            for i in range(cls.rounds):
                y = [None for _ in range(4)]

                rk = cls.round_keys[i % (2 * kappa // w)]

                if i % 2 == 0:
                    y[3] = RotateLeft((x[0] ^ i) + (RotateLeft(x[1], 1) ^ rk), 8)
                else:
                    y[3] = RotateLeft((x[0] ^ i) + (RotateLeft(x[1], 8) ^ rk), 1)

                for j in range(2 + 1):
                    y[j] = x[j + 1]

                x = y

            return x

    class ChamCipher(Cipher):
        key_schedule = ChamKeySchedule
        encryption = ChamEncryption
        rounds = default_rounds
        _cham_instance = cham_instance

        @classmethod
        def set_rounds(cls, new_rounds):
            cls.rounds = new_rounds
            cls.key_schedule.set_rounds(new_rounds)
            cls.encryption.set_rounds(new_rounds)

        @classmethod
        def test(cls):
            old_rounds = cls.rounds
            cls.set_rounds(default_rounds)

            if cls._cham_instance == ChamInstance.cham_64_128:
                plaintext = (0x1100, 0x3322, 0x5544, 0x7766)
                key = (0x0100, 0x0302, 0x0504, 0x0706,
                       0x0908, 0x0b0a, 0x0d0c, 0x0f0e)
                assert cls(plaintext, key) == (0x453c, 0x63bc, 0xdcfa, 0xbf4e)
            elif cls._cham_instance == ChamInstance.cham_128_128:
                plaintext = (0x33221100, 0x77665544, 0xbbaa9988, 0xffeeddcc)
                key = (0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c)
                assert cls(plaintext, key) == (0xc3746034, 0xb55700c5, 0x8d64ec32, 0x489332f7)
            elif cls._cham_instance == ChamInstance.cham_128_256:
                plaintext = (0x33221100, 0x77665544, 0xbbaa9988, 0xffeeddcc)
                key = (0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
                       0xf3f2f1f0, 0xf7f6f5f4, 0xfbfaf9f8, 0xfffefdfc)
                assert cls(plaintext, key) == (0xa899c8a0, 0xc929d55c, 0xab670d38, 0x0c4f7ac8)
            else:
                raise ValueError("invalid instance of Cham")

            cls.set_rounds(old_rounds)

    return ChamCipher
