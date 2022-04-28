"""Tests for the verification module."""
import unittest
import doctest

import arxpy.smt.verification_impossible


class EmptyTest(unittest.TestCase):
    pass


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(arxpy.smt.verification_impossible))
    return tests


