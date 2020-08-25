"""Permutation :math:`\pi` of Chaskey."""
from arxpy.bitvector.core import Constant
from arxpy.bitvector.operation import RotateLeft

from arxpy.primitives.primitives import BvFunction


class ChaskeyPi(BvFunction):
    rounds = 8
    input_widths = [32, 32, 32, 32]
    output_widths = [32, 32, 32, 32]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds

    @classmethod
    def eval(cls, v0, v1, v2, v3):
        for i in range(cls.rounds):
            v0 += v1
            v1 = RotateLeft(v1, 5)
            v1 ^= v0
            v0 = RotateLeft(v0, 16)

            v2 += v3
            v3 = RotateLeft(v3, 8)
            v3 ^= v2

            v0 += v3
            v3 = RotateLeft(v3, 13)
            v3 ^= v0

            v2 += v1
            v1 = RotateLeft(v1, 7)
            v1 ^= v2
            v2 = RotateLeft(v2, 16)

        return v0, v1, v2, v3

    @classmethod
    def test(cls):
        old_rounds = cls.rounds
        cls.set_rounds(8)

        pt = [
            Constant(0x00000000, 32)
        ] * 4
        ct = [
            Constant(0x00000000, 32)
        ] * 4
        assert cls(*pt) == tuple(ct)

        cls.set_rounds(old_rounds)
