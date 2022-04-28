"""Tests for the search_imp module."""
import unittest
import doctest

import arxpy.smt.search_impossible


class EmptyTest(unittest.TestCase):
    pass


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(arxpy.smt.search_impossible))
    return tests


