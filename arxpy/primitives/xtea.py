"""Xtea cipher."""
from arxpy.bitvector.core import Constant

from arxpy.primitives.primitives import KeySchedule, Encryption, Cipher


class XteaKeySchedule(KeySchedule):
    """Key schedule function."""

    rounds = 64
    input_widths = [32, 32, 32, 32]
    output_widths = [32 for i in range(64)]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.output_widths = [32 for _ in range(new_rounds)]

    @classmethod
    def eval(cls, *master_key):
        mk = list(master_key)
        s = Constant(0, 32)
        delta = Constant(0x9E3779B9, 32)
        k = []
        for i in range(cls.rounds):
            if hasattr(cls, "skip_rounds") and i in cls.skip_rounds:
                k.append(mk[0])  # cte outputs not supported
            else:
                if i % 2 == 0:
                    k.append(s + mk[int(s & Constant(3, 32))])
                    # s += delta
                else:
                    k.append(s + mk[int((s >> Constant(11, 32)) & Constant(3, 32))])
            if i % 2 == 0:
                s += delta
        return k


class XteaEncryption(Encryption):
    """Encryption function."""

    rounds = 64
    input_widths = [32, 32]
    output_widths = [32, 32]
    round_keys = None

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds

    @classmethod
    def eval(cls, x, y):
        v0 = x
        v1 = y
        k = cls.round_keys
        cls.round_inputs = []
        for i in range(cls.rounds):
            cls.round_inputs.append([v0, v1])
            if hasattr(cls, "skip_rounds") and i in cls.skip_rounds:
                continue
            v0, v1 = v1, v0 + ((((v1 << Constant(4, 32)) ^ (v1 >> Constant(5, 32))) + v1) ^ k[i])
        cls.round_inputs.append([v0, v1])

        return v0, v1


class XteaCipher(Cipher):
    key_schedule = XteaKeySchedule
    encryption = XteaEncryption
    rounds = 64

    @classmethod
    def set_rounds(cls, new_rounds):
        # assert new_rounds >= 2
        cls.rounds = new_rounds
        cls.encryption.set_rounds(new_rounds)
        cls.key_schedule.set_rounds(new_rounds)

    @classmethod
    def set_skip_rounds(cls, skip_rounds):
        assert isinstance(skip_rounds, (list, tuple))
        cls.encryption.skip_rounds = skip_rounds
        cls.key_schedule.skip_rounds = skip_rounds

    @classmethod
    def test(cls):
        """Test Xtea with official test vectors."""
        # https://go.googlesource.com/crypto/+/master/xtea/xtea_test.go
        plaintext = (0x41424344, 0x45464748)
        key = (0, 0, 0, 0)
        assert cls(plaintext, key) == (0xa0390589, 0xf8b8efa5)

        plaintext = (0x41424344, 0x45464748)
        key = (0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F)
        assert cls(plaintext, key) == (0x497df3d0, 0x72612cb5)
