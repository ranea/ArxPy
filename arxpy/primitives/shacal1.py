"""Shacal-1 cipher."""
from arxpy.bitvector.core import Constant
from arxpy.bitvector.operation import Operation, BvComp
from arxpy.bitvector.extraop import PopCount
from arxpy.bitvector.operation import RotateLeft as ROL

from arxpy.differential.difference import XorDiff
from arxpy.differential.derivative import Derivative

from arxpy.primitives.primitives import KeySchedule, Encryption, Cipher


N = 4  # number of key words, at least 4 for 128-bit, at most 16
assert N >= 4
# N == 16, optimized for default single-key
# N == 4, ref for linear related-key characteristics
# N == 4, optimized for non-linear related-key characteristics

REFERENCE_VERSION = False  # ctes added in the encryption


def get_constant(i):
    if i < 20:
        return Constant(0x5A827999, 32)
    elif i < 40:
        return Constant(0x6ED9EBA1, 32)
    elif i < 60:
        return Constant(0x8F1BBCDC, 32)
    else:
        return Constant(0xCA62C1D6, 32)


class Shacal1KeySchedule(KeySchedule):
    """Key schedule function."""

    rounds = 80
    input_widths = [32 for _ in range(N)]
    output_widths = [32 for _ in range(80 - (16 - N))]

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.output_widths = [32 for _ in range(N + max(new_rounds - 16, 0))]

    @classmethod
    def eval(cls, *W):  # w0, w1, ...
        rk = list(W)

        for i in range(N, 16):
            rk.append(Constant(0, 32))

        for i in range(16, cls.rounds):
            rk.append(ROL(rk[i-3] ^ rk[i-8] ^ rk[i-14] ^ rk[i-16], 1))

        if REFERENCE_VERSION:
            pass
        else:
            for i in range(cls.rounds):
                rk[i] += get_constant(i)

        return rk[:N] + rk[16:cls.rounds]


# noinspection PyPep8Naming
class Shacal1Encryption(Encryption):
    """Encryption function."""

    rounds = 80
    input_widths = [32 for _ in range(5)]
    output_widths = [32 for _ in range(5)]
    round_keys = None

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds

    @classmethod
    def round_function(cls, A, B, C, D, E, i):
        if i < N:
            W = cls.round_keys[i]
        elif N <= i < 16:
            if REFERENCE_VERSION:
                W = Constant(0, 32)
            else:
                W = get_constant(i)
        else:
            W = cls.round_keys[i - (16 - N)]

        if i < 20:
            F = BvIf
        elif i < 40:
            F = lambda x, y, z: x ^ y ^ z
        elif i < 60:
            F = BvMaj
        else:
            F = lambda x, y, z: x ^ y ^ z
        return [
            W + ROL(A, 5) + F(B, C, D) + E + get_constant(i) if REFERENCE_VERSION else W + ROL(A, 5) + F(B, C, D) + E,
            A,
            ROL(B, 30),
            C,
            D
        ]

    @classmethod
    def eval(cls, A, B, C, D, E):
        for i in range(cls.rounds):
            A, B, C, D, E = cls.round_function(A, B, C, D, E, i)

        return A, B, C, D, E


class Shacal1Cipher(Cipher):
    key_schedule = Shacal1KeySchedule
    encryption = Shacal1Encryption
    rounds = 80

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.key_schedule.set_rounds(new_rounds)
        cls.encryption.set_rounds(new_rounds)

    # noinspection SpellCheckingInspection
    @classmethod
    def test(cls):
        # https://www.cosic.esat.kuleuven.be/nessie/testvectors/
        # key = 80000000000000000000000000000000
        # 00000000000000000000000000000000
        # 00000000000000000000000000000000
        # 00000000000000000000000000000000
        # plain = 0000000000000000000000000000000000000000
        # cipher = 0FFD8D43 B4E33C7C 53461BD1 0F27A546 1050D90D

        old_rounds = cls.rounds
        cls.set_rounds(80)

        global REFERENCE_VERSION

        for ref_v in [True, False]:
            REFERENCE_VERSION = ref_v

            key = [Constant(0x80000000, 32)]
            key.extend([Constant(0, 32) for _ in range(1, N)])
            pt = [Constant(0, 32) for _ in range(5)]
            ct = [
                Constant(0x0FFD8D43, 32),
                Constant(0xB4E33C7C, 32),
                Constant(0x53461BD1, 32),
                Constant(0x0F27A546, 32),
                Constant(0x01050D90D, 32)
            ]
            assert cls(pt, key) == tuple(ct)

            if N == 16:
                # key =    00010203 04050607 08090A0B 0C0D0E0F
                #          10111213 14151617 18191A1B 1C1D1E1F
                #          20212223 24252627 28292A2B 2C2D2E2F
                #          30313233 34353637 38393A3B 3C3D3E3F
                # plain =  00112233 44556677 8899AABB CCDDEEFF 10213243
                # cipher = 213A4657 59CE3572 CA2A86D5 A680E948 45BEAA6B

                key = [
                    Constant(0x00010203, 32), Constant(0x04050607, 32), Constant(0x08090A0B, 32), Constant(0x0C0D0E0F, 32),
                    Constant(0x10111213, 32), Constant(0x14151617, 32), Constant(0x18191A1B, 32), Constant(0x1C1D1E1F, 32),
                    Constant(0x20212223, 32), Constant(0x24252627, 32), Constant(0x28292A2B, 32), Constant(0x2C2D2E2F, 32),
                    Constant(0x30313233, 32), Constant(0x34353637, 32), Constant(0x38393A3B, 32), Constant(0x3C3D3E3F, 32),
                ]
                pt = [
                    Constant(0x00112233, 32),
                    Constant(0x44556677, 32),
                    Constant(0x8899AABB, 32),
                    Constant(0xCCDDEEFF, 32),
                    Constant(0x10213243, 32),
                ]
                ct = [
                    Constant(0x213A4657, 32),
                    Constant(0x59CE3572, 32),
                    Constant(0xCA2A86D5, 32),
                    Constant(0xA680E948, 32),
                    Constant(0x45BEAA6B, 32)
                ]
                assert cls(pt, key) == tuple(ct)

        cls.set_rounds(old_rounds)


class BvIf(Operation):
    """The function If of SHACAL-1."""

    arity = [3, 0]
    is_symmetric = False

    @classmethod
    def output_width(cls, x, y, z):
        return x.width

    @classmethod
    def eval(cls, x, y, z):
        if all(isinstance(x_i, Constant) for x_i in [x, y, z]):
            return (x & y) | ((~x) & z)

    @classmethod
    def xor_derivative(cls, input_diff):
        return XDIf(input_diff)


class XDIf(Derivative):
    """Represent the derivative of the function If w.r.t XOR differences."""

    diff_type = XorDiff
    op = BvIf

    def is_possible(self, output_diff):
        x, y, z = [d.val for d in self.input_diff]
        w = output_diff.val
        n = x.width

        def eq(a, b):
            return ~a ^ b

        return BvComp(~x & eq(y, z) & eq(~y, w), Constant(0, n))

    def has_probability_one(self, output_diff):
        x, y, z = [d.val for d in self.input_diff]
        w = output_diff.val
        n = x.width

        def eq(x, y, z):
            return (~x ^ y) & (~x ^ z)

        return BvComp(~x & eq(y, z, w), ~Constant(0, n))

    def weight(self, output_diff):
        x, y, z = [d.val for d in self.input_diff]

        def eq(a, b):
            return ~a ^ b

        pr1 = ~x & eq(y, z)

        return PopCount(~pr1)

    def max_weight(self):
        x, y, z = [d.val for d in self.input_diff]
        return x.width

    def error(self):
        return 0

    def exact_weight(self, output_diff):
        return int(self.weight(output_diff))

    def num_frac_bits(self):
        return 0


class BvMaj(Operation):
    """The function majority of SHACAL-1."""

    arity = [3, 0]
    is_symmetric = False

    @classmethod
    def output_width(cls, x, y, z):
        return x.width

    @classmethod
    def eval(cls, x, y, z):
        if all(isinstance(x_i, Constant) for x_i in [x, y, z]):
            return (x & y) | (x & z) | (y & z)

    @classmethod
    def xor_derivative(cls, input_diff):
        return XDMaj(input_diff)


class XDMaj(Derivative):
    """Represent the derivative of the majority function w.r.t XOR differences."""

    diff_type = XorDiff
    op = BvMaj

    def is_possible(self, output_diff):
        x, y, z = [d.val for d in self.input_diff]
        w = output_diff.val
        n = x.width

        def eq(x, y, z):
            return (~x ^ y) & (~x ^ z)

        return BvComp(eq(x, y, z) & (x ^ w), Constant(0, n))

    def has_probability_one(self, output_diff):
        x, y, z = [d.val for d in self.input_diff]
        w = output_diff.val
        n = x.width

        def eq(x, y, z, w):
            return (~x ^ y) & (~x ^ z) & (~x ^ w)

        return BvComp(eq(x, y, z, w), ~Constant(0, n))

    def weight(self, output_diff):
        x, y, z = [d.val for d in self.input_diff]

        def eq(x, y, z):
            return (~x ^ y) & (~x ^ z)

        pr1 = eq(x, y, z)

        return PopCount(~pr1)

    def max_weight(self):
        x, y, z = [d.val for d in self.input_diff]
        return x.width

    def error(self):
        return 0

    def exact_weight(self, output_diff):
        return int(self.weight(output_diff))

    def num_frac_bits(self):
        return 0
