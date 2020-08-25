"""Tests for the Characteristic module."""
import doctest
import unittest

from hypothesis import given
from hypothesis.strategies import integers

from arxpy.bitvector.core import Variable, Constant
from arxpy.bitvector.operation import RotateLeft, RotateRight

from arxpy.primitives.primitives import BvFunction, KeySchedule, Encryption, Cipher

from arxpy.differential.difference import XorDiff, RXDiff

from arxpy.differential import characteristic
from arxpy.differential.characteristic import BvCharacteristic, SingleKeyCh, RelatedKeyCh

VERBOSE = False


# 1st cipher: linear key schedule and encryption with modular addition (no ctes)

class MyFunction(BvFunction):  # noqa: D101
    input_widths = [8, 8]
    output_widths = [8]
    rounds = 1

    @classmethod
    def eval(cls, x, y):
        x = RotateLeft(x ^ Constant(1, 8), 1)
        y = ~y
        return tuple([x + y])


class KeySchedule1(KeySchedule):  # noqa: D101
    input_widths = [8]
    output_widths = [8, 8]

    @classmethod
    def eval(cls, mk):
        return tuple([mk, mk ^ Constant(1, 8)])


class Encryption1(Encryption):  # noqa: D101
    input_widths = [8, 8]
    output_widths = [8, 8]
    round_keys = None

    @classmethod
    def eval(cls, x, y):
        x ^= cls.round_keys[0]
        x = (x + y)
        y = x ^ RotateLeft(cls.round_keys[1], 1)
        return x, y


class Cipher1(Cipher):  # noqa: D101
    key_schedule = KeySchedule1
    encryption = Encryption1
    rounds = 1

    @classmethod
    def set_rounds(cls, new_rounds):
        assert new_rounds == 1


# 2nd cipher: non-linear key schedule (without mixing) with complex encryption
#             (mixing + and +1)


class KeySchedule2(KeySchedule):  # noqa: D101
    input_widths = [8]
    output_widths = [8]

    @classmethod
    def eval(cls, mk):
        return tuple([RotateRight(mk, 1) + Constant(3, 8)])


class Encryption2(Encryption):  # noqa: D101
    input_widths = [8, 8]
    output_widths = [8, 8]
    round_keys = None

    @classmethod
    def eval(cls, x, y):
        x = (x + y) ^ cls.round_keys[0]
        y = RotateLeft(x, 1) + Constant(1, 8)
        return x, y


class Cipher2(Cipher):  # noqa: D101
    key_schedule = KeySchedule2
    encryption = Encryption2
    rounds = 1

    @classmethod
    def set_rounds(cls, new_rounds):
        assert new_rounds == 1


def first_value(my_dict):  # noqa: D103
    return next(iter(my_dict.values()))


class TestCh(unittest.TestCase):
    """Tests of the Ch classes."""

    @given(
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
        integers(min_value=0, max_value=2 ** 8 - 1),
    )
    def test_bv_ch(self, x, y, x_, y_):
        input_diff_names = ["dx", "dy"]
        prefix = "d"

        x = Constant(x, 8)
        x_ = Constant(x_, 8)
        y = Constant(y, 8)
        y_ = Constant(y_, 8)

        for diff_type in [XorDiff, RXDiff]:
            ch = BvCharacteristic(MyFunction, diff_type, input_diff_names, prefix)
            input_diff = [
                diff_type.from_pair(x, x_),
                diff_type.from_pair(y, y_)
            ]
            # noinspection PyUnresolvedReferences
            output_diff = diff_type.from_pair(
                MyFunction(x, y)[0],
                MyFunction(x_, y_)[0]
            )

            der = first_value(ch.nonlinear_diffs)
            der.input_diff[0].val = der.input_diff[0].val.xreplace(
                {diff_type(Variable("dx", 8)): input_diff[0]}
            )
            der.input_diff[1].val = der.input_diff[1].val.xreplace(
                {diff_type(Variable("dx", 8)): input_diff[1]}
            )
            self.assertTrue(der.is_possible(output_diff))

    def test_singlekey_ch(self):
        diff_type = XorDiff
        for cipher in [Cipher1, Cipher2]:
            skch = SingleKeyCh(cipher, diff_type)

            if VERBOSE:
                num_inputs = len(skch.func.input_widths)
                input_diff_names = ["dp" + str(i) for i in range(num_inputs)]
                prefix = "d"
                print(diff_type.__name__, type(skch).__name__)
                print("ssa:", skch.func.ssa(input_diff_names, id_prefix=prefix))
                print("input:", skch.input_diff)
                print("output:", skch.output_diff)
                print("nonlinear:", skch.nonlinear_diffs)
                print()

            if cipher == Cipher1:
                test_vectors = ["XDA(XorDiff(dp1), XorDiff(dp0))"]
            elif cipher == Cipher2:
                test_vectors = ["XDA(XorDiff(dp0), XorDiff(dp1))", "XDCA_0x01(XorDiff(dx0 <<< 1))"]
            else:
                raise ValueError("invalid cipher: " + str(cipher))
            for der, tv in zip(skch.nonlinear_diffs.values(), test_vectors):
                self.assertEqual(str(der), tv)

    def test_relatedkey_ch(self):
        for diff_type in [XorDiff, RXDiff]:
            for cipher in [Cipher1, Cipher2]:
                if diff_type == RXDiff and cipher == Cipher2:
                    continue

                rkch = RelatedKeyCh(cipher, diff_type)

                if VERBOSE:
                    print(diff_type.__name__, type(rkch).__name__)
                    ch = rkch.key_schedule_ch
                    num_inputs = len(ch.func.input_widths)
                    input_diff_names = ["dmk" + str(i) for i in range(num_inputs)]
                    prefix = "dk"
                    print("ssa:", ch.func.ssa(input_diff_names, id_prefix=prefix))
                    print("input:", ch.input_diff)
                    print("output:", ch.output_diff)
                    print("nonlinear:", ch.nonlinear_diffs)
                    ch = rkch.encryption_ch
                    num_inputs = len(ch.func.input_widths)
                    input_diff_names = ["dp" + str(i) for i in range(num_inputs)]
                    prefix = "dx"
                    print("ssa:", ch.func.ssa(input_diff_names, id_prefix=prefix))
                    print("input:", ch.input_diff)
                    print("output:", ch.output_diff)
                    print("nonlinear:", ch.nonlinear_diffs)
                    print()

                if cipher == Cipher1:
                    if diff_type == XorDiff:
                        key_test_vectors = ["XDIdentity(XorDiff(dmk0),)"]
                        enc_test_vectors = ["XDA(XorDiff(dp1), XorDiff(dmk0 ^ dp0))"]
                    elif diff_type == RXDiff:
                        key_test_vectors = ["RXDA(RXDiff(dmk0 <<< 1), RXDiff(dmk0))"]
                        enc_test_vectors = ["RXDA(RXDiff(dp1), RXDiff(dmk0 ^ dp0))"]
                elif cipher == Cipher2 and diff_type == XorDiff:
                        key_test_vectors = ["XDCA_0x03(XorDiff(dmk0 >>> 1))"]
                        enc_test_vectors = ["XDA(XorDiff(dp0), XorDiff(dp1))", "XDCA_0x01(XorDiff((dk1 ^ dx0) <<< 1))"]
                else:
                    raise ValueError("invalid cipher: " + str(cipher))

                for der, tv in zip(rkch.key_schedule_ch.nonlinear_diffs.values(), key_test_vectors):
                    self.assertEqual(str(der), tv)
                for der, tv in zip(rkch.encryption_ch.nonlinear_diffs.values(), enc_test_vectors):
                    self.assertEqual(str(der), tv)


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(characteristic))
    return tests
