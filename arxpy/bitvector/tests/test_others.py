"""Tests for the Printing and State module."""
import doctest

import arxpy.bitvector.context
import arxpy.bitvector.function
import arxpy.bitvector.printing


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(arxpy.bitvector.printing))
    tests.addTests(doctest.DocTestSuite(arxpy.bitvector.context))
    tests.addTests(doctest.DocTestSuite(arxpy.bitvector.function))
    return tests
