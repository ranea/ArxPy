"""Tests for the Smt module."""
import doctest
import unittest

from arxpy.bitvector.core import Constant, Variable
from arxpy.bitvector.operation import (
    BvNot, BvComp, RotateLeft, RotateRight, Concat, Repeat)

import arxpy.diffcrypt.smt
from arxpy.diffcrypt.smt import bv2pysmt

from pysmt import shortcuts as sc
from pysmt import typing


class TestConversionpsySMT(unittest.TestCase):
    """Tests of the bv2pysmt and psySMT2bv functions."""

    def test_bv2pysmt(self):
        bvx, bvy = Variable("x", 8), Variable("y", 8)
        psx, psy = bv2pysmt(bvx), bv2pysmt(bvy)

        self.assertEqual(bv2pysmt(Constant(0, 8)), sc.BV(0, 8))
        self.assertEqual(psx, sc.Symbol("x", typing.BVType(8)))

        self.assertEqual(bv2pysmt(~bvx), sc.BVNot(psx))
        self.assertEqual(bv2pysmt(bvx & bvy), sc.BVAnd(psx, psy))
        self.assertEqual(bv2pysmt(bvx | bvy), sc.BVOr(psx, psy))
        self.assertEqual(bv2pysmt(bvx ^ bvy), sc.BVXor(psx, psy))

        self.assertEqual(bv2pysmt(BvComp(bvx, bvy)), sc.Equals(psx, psy))
        self.assertEqual(bv2pysmt(BvNot(BvComp(bvx, bvy))),
                         sc.Not(sc.Equals(psx, psy)))

        self.assertEqual(bv2pysmt(bvx < bvy), sc.BVULT(psx, psy))
        self.assertEqual(bv2pysmt(bvx <= bvy), sc.BVULE(psx, psy))
        self.assertEqual(bv2pysmt(bvx > bvy), sc.BVUGT(psx, psy))
        self.assertEqual(bv2pysmt(bvx >= bvy), sc.BVUGE(psx, psy))

        self.assertEqual(bv2pysmt(bvx << bvy), sc.BVLShl(psx, psy))
        self.assertEqual(bv2pysmt(bvx >> bvy), sc.BVLShr(psx, psy))
        self.assertEqual(bv2pysmt(RotateLeft(bvx, 1)), sc.BVRol(psx, 1))
        self.assertEqual(bv2pysmt(RotateRight(bvx, 1)), sc.BVRor(psx, 1))

        self.assertEqual(bv2pysmt(bvx[4:2]), sc.BVExtract(psx, 2, 4))
        self.assertEqual(bv2pysmt(Concat(bvx, bvy)), sc.BVConcat(psx, psy))
        # zeroextend reduces to Concat
        # self.assertEqual(bv2pysmt(ZeroExtend(bvx, 2)), sc.BVZExt(psx, 2))
        self.assertEqual(bv2pysmt(Repeat(bvx, 2)), psx.BVRepeat(2))

        self.assertEqual(bv2pysmt(-bvx), sc.BVNeg(psx))
        self.assertEqual(bv2pysmt(bvx + bvy), sc.BVAdd(psx, psy))
        # bvsum reduces to add
        # self.assertEqual(bv2pysmt(bvx - bvy), sc.BVSub(psx, psy))
        self.assertEqual(bv2pysmt(bvx * bvy), sc.BVMul(psx, psy))
        self.assertEqual(bv2pysmt(bvx / bvy), sc.BVUDiv(psx, psy))
        self.assertEqual(bv2pysmt(bvx % bvy), sc.BVURem(psx, psy))


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(arxpy.diffcrypt.smt))
    return tests
