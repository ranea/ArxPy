"""Tests for the Cipher module."""
import doctest
import unittest

from arxpy.bitvector.core import Variable

from arxpy.primitives import primitives
from arxpy.primitives.primitives import BvFunction, KeySchedule, Encryption, Cipher


class TestBvFunction(unittest.TestCase):
    """Tests of the BvFunction class."""

    def setUp(self):
        class MyFunction(BvFunction):
            input_widths = [8, 8]
            output_widths = [8, 8]
            rounds = 1

            @classmethod
            def eval(cls, x, y):
                return x ^ y, x

        class MyIterFunction(BvFunction):
            input_widths = [8, 8]
            output_widths = [8]
            rounds = 2

            @classmethod
            def eval(cls, *args):
                result = args[0]
                for arg in args[1:]:
                    result += arg
                return tuple([result])

            @classmethod
            def set_rounds(cls, new_rounds):
                cls.rounds = new_rounds
                cls.input_widths = [8 for _ in range(new_rounds)]

        self.func = MyFunction
        self.iter_func = MyIterFunction

    def test_creation(self):
        self.assertEqual(self.func(0, 0), (0x00, 0x00))

        with self.assertRaises(TypeError):
            self.func(Variable("x", 8), Variable("y", 8))

    def test_iterated(self):
        self.assertEqual(self.iter_func(0, 1), (1,))
        self.iter_func.set_rounds(3)
        self.assertEqual(self.iter_func(0, 1, 2), (3,))

    def test_ssa(self):
        self.assertEqual(
            str(self.func.ssa(["x", "y"], "r")),
            "{'input_vars': (x, y), 'output_vars': (r0, x), 'assignments': ((r0, x ^ y),)}"
        )


class TestCipher(unittest.TestCase):
    """Tests of the Cipher class."""

    def setUp(self):
        class MyKeySchedule(KeySchedule):
            input_widths = [8]
            output_widths = [8 for i in range(2)]
            rounds = 2

            @classmethod
            def eval(cls, k):
                return [k + i for i in range(cls.rounds)]

            @classmethod
            def set_rounds(cls, new_rounds):
                cls.rounds = new_rounds
                cls.output_widths = [8 for _ in range(new_rounds)]

        class MyEncryption(Encryption):
            input_widths = [8]
            output_widths = [8]
            rounds = 2
            round_keys = None

            @classmethod
            def eval(cls, x):
                for k_i in cls.round_keys:
                    x ^= k_i
                return tuple([x])

            @classmethod
            def set_rounds(cls, new_rounds):
                cls.rounds = new_rounds

        class MyCipher(Cipher):
            key_schedule = MyKeySchedule
            encryption = MyEncryption
            rounds = 2

            @classmethod
            def set_rounds(cls, new_rounds):
                cls.rounds = new_rounds
                cls.key_schedule.set_rounds(new_rounds)
                cls.encryption.set_rounds(new_rounds)

        self.cipher = MyCipher

    def test_creation(self):
        self.assertEqual(self.cipher([0], [0]), (0x01, ))

        with self.assertRaises(TypeError):
            self.cipher([Variable("x", 8)], [Variable("y", 8)])

    def test_iterated(self):
        self.cipher.set_rounds(3)
        self.assertEqual(self.cipher([0], [0]), (0x03, ))


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(primitives))
    return tests
