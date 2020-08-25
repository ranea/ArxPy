"""Tests for the context and printing module."""
import unittest
import doctest

import arxpy.bitvector.context
import arxpy.bitvector.printing


class EmptyTest(unittest.TestCase):
    pass


# noinspection PyUnusedLocal,PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(arxpy.bitvector.printing))
    tests.addTests(doctest.DocTestSuite(arxpy.bitvector.context))
    return tests
