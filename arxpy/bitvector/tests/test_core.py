"""Tests for the core module."""
import doctest
import unittest

from arxpy.bitvector.core import bitvectify, Constant, Term, Variable


class TestTerm(unittest.TestCase):
    """Tests of the Term class."""

    def test_invalid_width(self):  
        with self.assertRaises(AssertionError):
            Term(width=-1)
        with self.assertRaises(AssertionError):
            Term(width=0)
        with self.assertRaises(AssertionError):
            Term(width="8")
        with self.assertRaises(AssertionError):
            Term(width=[1, 1])

    def test_initialization(self):  
        t = Term(width=8)

        with self.assertRaises(AttributeError):
            t.width += 1


class TestVariable(unittest.TestCase):
    """Tests of the Variable class."""

    def test_invalid_args(self):  
        with self.assertRaises(AssertionError):
            Variable(["v", "a", "r"], 8)
        with self.assertRaises(AssertionError):
            Variable(000, 8)

    def test_initialization(self):  
        s = Variable("x", 8)
        self.assertTrue(s.is_Atom)
        self.assertEqual(len(s.atoms()), 1)
        self.assertEqual(s.atoms(), {s})

        with self.assertRaises(AttributeError):
            s.name = 0

    def test_comparisons(self):  
        x, y = Variable("x", 8), Variable("y", 8)
        x9, z = Variable("x", 9), Variable("z", 9)

        self.assertTrue((x != y) & (x != x9) & (x != z))
        self.assertEqual(x, Variable("x", 8))


class TestConstant(unittest.TestCase):
    """Tests of the Constant class."""

    def test_invalid_args(self):  
        with self.assertRaises(AssertionError):
            Constant("1", 8)
        with self.assertRaises(AssertionError):
            Constant(0.5, 8)
        with self.assertRaises(AssertionError):
            Constant(-1, 8)
        with self.assertRaises(AssertionError):
            Constant(9, 2)

    def test_initialization(self):  
        x = Constant(0, 8)

        self.assertTrue(x.is_Atom)
        self.assertEqual(x.atoms(), {x})

        with self.assertRaises(AttributeError):
            x.val = 0


class Testbitvectify(unittest.TestCase):
    """Tests of the bitvectify function."""

    def test_invalid_args(self):  
        with self.assertRaises(AssertionError):
            bitvectify(1, -1)
        with self.assertRaises(AssertionError):
            bitvectify(Constant(0, 8), 8 + 1)
        with self.assertRaises(AssertionError):
            bitvectify(Variable("x", 8), 8 + 1)
        with self.assertRaises(AssertionError):
            bitvectify(Term(width=8), 8 + 1)

    def test_initialization(self):  
        self.assertEqual(bitvectify(2, 8), Constant(2, 8))
        self.assertEqual(bitvectify("x", 8), Variable("x", 8))
        self.assertEqual(bitvectify(Term(width=8), 8), Term(width=8))


# noinspection PyUnusedLocal,PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    import arxpy.bitvector.core
    tests.addTests(doctest.DocTestSuite(arxpy.bitvector.core))
    return tests
