"""MULTI2 cipher.

Source: Cryptanalysis of the ISDB Scrambling Algorithm (MULTI2)
"""
from arxpy.bitvector.core import Constant
from arxpy.bitvector.operation import RotateLeft

from arxpy.primitives.primitives import KeySchedule, Encryption, Cipher

# for BvOR and XDOr
from arxpy.bitvector.operation import BvComp, Operation
from arxpy.bitvector.extraop import PopCount
from arxpy.differential.difference import XorDiff
from arxpy.differential.derivative import Derivative


REFERENCE_VERSION = False  # if True, ctes are added in the encryption


class BvOr(Operation):
    """The OR function."""

    arity = [2, 0]
    is_symmetric = True

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        if all(isinstance(x_i, Constant) for x_i in [x, y]):
            return x | y

    @classmethod
    def xor_derivative(cls, input_diff):
        return XDOr(input_diff)


class XDOr(Derivative):
    """Represent the derivative of the function If w.r.t XOR differences."""

    diff_type = XorDiff
    op = BvOr

    def is_possible(self, output_diff):
        dx, dy = [d.val for d in self.input_diff]
        dz = output_diff.val
        n = dx.width

        # (dx, dy) = (0, 0) -> dz =(1)
        bad_case = (~dx) & (~dy) & dz

        return BvComp(bad_case, Constant(0, n))

    def has_probability_one(self, output_diff):
        dx, dy = [d.val for d in self.input_diff]
        dz = output_diff.val
        n = dx.width

        # (dx, dy) = (0, 0) -> dz =(0)
        pr1_case = (~dx) & (~dy) & (~dz)

        return BvComp(pr1_case, ~Constant(0, n))

    def weight(self, output_diff):
        dx, dy = [d.val for d in self.input_diff]

        return PopCount(dx | dy)

    def max_weight(self):
        dx, dy = [d.val for d in self.input_diff]
        return dx.width

    def error(self):
        return 0

    def exact_weight(self, output_diff):
        return int(self.weight(output_diff))

    def num_frac_bits(self):
        return 0


def pi1(L):
    return L


def pi2(R, k_i):
    if REFERENCE_VERSION:
        x = RotateLeft(R + k_i, 1) + (R + k_i) + (-Constant(1, 32))
    else:
        assert isinstance(k_i, list)
        x = RotateLeft(R + k_i[0], 1) + (R + k_i[1])
    return RotateLeft(x, 4) ^ x


def pi3(L, k_i, k_j):
    if REFERENCE_VERSION:
        y = RotateLeft(L + k_i, 2) + (L + k_i) + Constant(1, 32)
    else:
        assert isinstance(k_i, list)
        y = RotateLeft(L + k_i[0], 2) + (L + k_i[1])

    x = RotateLeft(RotateLeft(y, 8) ^ (y + k_j), 1) - (RotateLeft(y, 8) ^ (y + k_j))

    return RotateLeft(x, 16) ^ (BvOr(x, L))


def pi4(R, k_i):
    if REFERENCE_VERSION:
        x = RotateLeft(R + k_i, 2) + (R + k_i) + Constant(1, 32)
    else:
        assert isinstance(k_i, list)
        x = RotateLeft(R + k_i[0], 2) + (R + k_i[1])
    return x


class Multi2KeySchedule(KeySchedule):
    """Key schedule function."""

    rounds = 32
    input_widths = [32, 32] + [32 for _ in range(8)]
    output_widths = [32 for _ in range(8)]

    @classmethod
    def get_num_keys(cls):
        num_system_keys = 0
        num_round_keys = 0
        for i in range(min(8, cls.rounds)):
            if i == 0:
                # s[1] |
                num_system_keys += 1
            elif i == 1:
                # s[2], s[3] | k[1]
                num_system_keys += 2
                num_round_keys += 1
            elif i == 2:
                # s[4] |  k[2], k[3]
                num_system_keys += 1
                num_round_keys += 2
            elif i == 3:
                # | k[4]
                num_round_keys += 1
            elif i == 4:
                # s[5] |
                num_system_keys += 1
            elif i == 5:
                # s[6], s[7] | k[5]
                num_system_keys += 2
                num_round_keys += 1
            elif i == 6:
                # s[8] | k[6], k[7]
                num_system_keys += 1
                num_round_keys += 2
            elif i == 7:
                # | k[8]
                num_round_keys += 1

        num_expanded_round_keys = num_round_keys
        if not REFERENCE_VERSION:
            for i in range(min(8, cls.rounds)):
                if i == 0:
                    pass
                elif i == 1:
                    num_expanded_round_keys += 1
                elif i == 2:
                    num_expanded_round_keys += 1
                elif i == 3:
                    num_expanded_round_keys += 1
                elif i == 4:
                    pass
                elif i == 5:
                    num_expanded_round_keys += 1
                elif i == 6:
                    num_expanded_round_keys += 1
                elif i == 7:
                    num_expanded_round_keys += 1

        return num_system_keys, num_round_keys, num_expanded_round_keys

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        num_system_keys, _, num_expanded_round_keys = cls.get_num_keys()
        cls.input_widths = [32, 32] + [32 for _ in range(num_system_keys)]
        cls.output_widths = [32 for _ in range(num_expanded_round_keys)]

    @classmethod
    def eval(cls, d1, d2, *system_key):
        s = [None] + list(system_key)

        num_system_keys, num_round_keys, num_expanded_round_keys = cls.get_num_keys()

        global REFERENCE_VERSION
        old_ref_v = REFERENCE_VERSION
        REFERENCE_VERSION = True

        k = [None]
        for i in range(num_round_keys):
            if i == 0:
                k.append(d1 ^ pi2(d1 ^ d2, s[1]))
            elif i == 1:
                k.append(d1 ^ d2 ^ pi3(k[1], s[2], s[3]))
            elif i == 2:
                k.append(k[1] ^ pi4(k[2], s[4]))
            elif i == 3:
                k.append(k[2] ^ k[3])
            elif i == 4:
                k.append(k[3] ^ pi2(k[4], s[5]))
            elif i == 5:
                k.append(k[4] ^ pi3(k[5], s[6], s[7]))
            elif i == 6:
                k.append(k[5] ^ pi4(k[6], s[8]))
            elif i == 7:
                k.append(k[6] ^ k[7])

        assert len(k) == num_round_keys + 1

        REFERENCE_VERSION = old_ref_v

        if not REFERENCE_VERSION:
            if len(k) > 8:
                # pi4(R, k[8]) | pi4 : k_i + Constant(1, 32)
                k.append(k[8] + Constant(1, 32))

            if len(k) > 6:
                # pi3(L, k[6], *) | pi3 : k_i + Constant(1, 32)
                k.append(k[6] + Constant(1, 32))

            if len(k) > 5:
                # pi2(R, k[5]) | pi2 : k_i + (-Constant(1, 32))
                k.append(k[5] + (-Constant(1, 32)))

            if len(k) > 4:
                # pi4(R, k[4]) | pi4 : k_i + Constant(1, 32)
                k.append(k[4] + Constant(1, 32))

            if len(k) > 2:
                # pi3(L, k[2], *) | pi3 : k_i + Constant(1, 32)
                k.append(k[2] + Constant(1, 32))

            if len(k) > 1:
                # pi2(R, k[1]) | pi2 : k_i + (-Constant(1, 32))
                k.append(k[1] + (-Constant(1, 32)))

        assert len(k) == num_expanded_round_keys + 1

        return k[1: 1+num_expanded_round_keys]


class Multi2Encryption(Encryption):
    """Encryption function."""

    rounds = 32
    input_widths = [32, 32]
    output_widths = [32, 32]
    round_keys = None

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds

    @classmethod
    def eval(cls, L, R):
        k = [None] + list(cls.round_keys)

        for i in range(cls.rounds):
            if REFERENCE_VERSION:
                if i % 8 == 0:
                    R ^= pi1(L)
                elif i % 8 == 1:
                    L ^= pi2(R, k[1])
                elif i % 8 == 2:
                    R ^= pi3(L, k[2], k[3])
                elif i % 8 == 3:
                    L ^= pi4(R, k[4])
                elif i % 8 == 4:
                    R ^= pi1(L)
                elif i % 8 == 5:
                    L ^= pi2(R, k[5])
                elif i % 8 == 6:
                    R ^= pi3(L, k[6], k[7])
                elif i % 8 == 7:
                    L ^= pi4(R, k[8])
            else:
                if i % 8 == 0:
                    R ^= pi1(L)
                elif i % 8 == 1:
                    L ^= pi2(R, [k[1], k[-1]])
                elif i % 8 == 2:
                    R ^= pi3(L, [k[2], k[-2]], k[3])
                elif i % 8 == 3:
                    L ^= pi4(R, [k[4], k[-3]])
                elif i % 8 == 4:
                    R ^= pi1(L)
                elif i % 8 == 5:
                    L ^= pi2(R, [k[5], k[-4]])
                elif i % 8 == 6:
                    R ^= pi3(L, [k[6], k[-5]], k[7])
                elif i % 8 == 7:
                    L ^= pi4(R, [k[8], k[-6]])

        return L, R


class Multi2Cipher(Cipher):
    key_schedule = Multi2KeySchedule
    encryption = Multi2Encryption
    rounds = 32

    @classmethod
    def set_rounds(cls, new_rounds):
        cls.rounds = new_rounds
        cls.encryption.set_rounds(new_rounds)
        cls.key_schedule.set_rounds(new_rounds)

    @classmethod
    def test(cls):
        old_rounds = cls.rounds

        global REFERENCE_VERSION

        for ref_v in [True, False]:
            REFERENCE_VERSION = ref_v

            cls.set_rounds(32)

            plaintext = [0, 0]
            key = [0 for _ in range(len(cls.key_schedule.input_widths))]
            ciphertext = (0x1d9dfa1e, 0x4d64bc67)
            assert cls(plaintext, key) == ciphertext

            plaintext = [0x01, 0x23]
            key = [0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23]
            ciphertext = (0xd241e7c8, 0x74166979)
            assert cls(plaintext, key) == ciphertext

        cls.set_rounds(old_rounds)
