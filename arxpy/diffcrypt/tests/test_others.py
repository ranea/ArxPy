"""Tests for the Characteristic and Smt module."""
import doctest

import arxpy.diffcrypt.characteristic
import arxpy.diffcrypt.itercipher


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(arxpy.diffcrypt.characteristic))
    tests.addTests(doctest.DocTestSuite(arxpy.diffcrypt.itercipher))
    return tests
