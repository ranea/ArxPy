"""Tests for the Types module."""
import doctest
import functools
import unittest

from arxpy.bitvector.core import Constant, Variable
from arxpy.bitvector.operation import (
    BvNot, BvComp, RotateLeft, RotateRight, Concat, Repeat, ZeroExtend, Ite)

import arxpy.smt.types
from arxpy.smt.types import bv2pysmt as _bv2pysmt

from pysmt import environment


class TestConversion(unittest.TestCase):
    """Tests of the bv2pysmt function."""

    def setUp(self):
        self.env = environment.reset_env()

    def test_bv2pysmt(self):
        bv2pysmt = functools.partial(_bv2pysmt, env=self.env)
        fm = self.env.formula_manager
        tm = self.env.type_manager

        bx, by = Variable("x", 8), Variable("y", 8)
        b1x, b1y = Variable("x1", 1), Variable("y1", 1)
        b6x, b6y = Variable("x6", 6), Variable("y6", 6)
        px, py = bv2pysmt(bx), bv2pysmt(by)
        p1x, p1y = bv2pysmt(b1x, True), bv2pysmt(b1y, True)
        p6x, p6y = bv2pysmt(b6x), bv2pysmt(b6y)

        self.assertEqual(bv2pysmt(Constant(0, 8)), fm.BV(0, 8))
        self.assertEqual(px, fm.Symbol("x", tm.BVType(8)))
        self.assertEqual(p1x, fm.Symbol("x1", tm.BOOL()))

        self.assertEqual(bv2pysmt(~bx), fm.BVNot(px))
        self.assertEqual(bv2pysmt(~b1x, True), fm.Not(p1x))
        self.assertEqual(bv2pysmt(bx & by), fm.BVAnd(px, py))
        self.assertEqual(bv2pysmt(b1x & b1y, True), fm.And(p1x, p1y))
        self.assertEqual(bv2pysmt(bx | by), fm.BVOr(px, py))
        self.assertEqual(bv2pysmt(b1x | b1y, True), fm.Or(p1x, p1y))
        self.assertEqual(bv2pysmt(bx ^ by), fm.BVXor(px, py))
        self.assertEqual(bv2pysmt(b1x ^ b1y, True), fm.Xor(p1x, p1y))

        self.assertEqual(bv2pysmt(BvComp(bx, by)), fm.BVComp(px, py))
        self.assertEqual(bv2pysmt(BvComp(bx, by), True), fm.Equals(px, py))
        self.assertEqual(bv2pysmt(BvNot(BvComp(bx, by))),
                         fm.BVNot(fm.BVComp(px, py)))
        self.assertEqual(bv2pysmt(BvNot(BvComp(bx, by)), True),
                         fm.Not(fm.Equals(px, py)))

        self.assertEqual(bv2pysmt(bx < by), fm.BVULT(px, py))
        self.assertEqual(bv2pysmt(bx <= by), fm.BVULE(px, py))
        self.assertEqual(bv2pysmt(bx > by), fm.BVUGT(px, py))
        self.assertEqual(bv2pysmt(bx >= by), fm.BVUGE(px, py))

        self.assertEqual(bv2pysmt(bx << by), fm.BVLShl(px, py))
        self.assertEqual(bv2pysmt(bx >> by), fm.BVLShr(px, py))
        self.assertEqual(bv2pysmt(RotateLeft(bx, 1)), fm.BVRol(px, 1))
        self.assertEqual(bv2pysmt(RotateRight(bx, 1)), fm.BVRor(px, 1))

        def zext(pysmt_type, offset):
            # zero_extend reduces to Concat
            return fm.BVConcat(fm.BV(0, offset), pysmt_type)

        self.assertEqual(bv2pysmt(b6x << b6y, strict_shift=True),
                         fm.BVExtract(fm.BVLShl(zext(p6x, 2), zext(p6y, 2)), 0, 5))
        self.assertEqual(bv2pysmt(RotateRight(b6x, 1), strict_shift=True),
                         fm.BVConcat(fm.BVExtract(p6x, 0, 0), fm.BVExtract(p6x, 1, 5)))

        self.assertEqual(bv2pysmt(bx[4:2]), fm.BVExtract(px, 2, 4))
        self.assertEqual(bv2pysmt(Concat(bx, by)), fm.BVConcat(px, py))

        self.assertEqual(bv2pysmt(ZeroExtend(bx, 2)), zext(px, 2))
        self.assertEqual(bv2pysmt(Repeat(bx, 2)), px.BVRepeat(2))
        self.assertEqual(bv2pysmt(-bx), fm.BVNeg(px))
        self.assertEqual(bv2pysmt(bx + by), fm.BVAdd(px, py))
        # bv_sum reduces to add
        self.assertEqual(bv2pysmt(bx - by), fm.BVSub(px, py))
        self.assertEqual(bv2pysmt(bx * by), fm.BVMul(px, py))
        self.assertEqual(bv2pysmt(bx / by), fm.BVUDiv(px, py))
        self.assertEqual(bv2pysmt(bx % by), fm.BVURem(px, py))

        # cannot reuse Bool and BV{1} variable with the same name
        bxx, byy = Variable("xx", 8), Variable("yy", 8)
        b1xx, b1yy, b1zz = Variable("xx1", 1), Variable("yy1", 1), Variable("zz1", 1)
        pxx, pyy = bv2pysmt(bxx), bv2pysmt(byy)
        p1xx, p1yy, p1zz = bv2pysmt(b1xx, False), bv2pysmt(b1yy, True), bv2pysmt(b1zz, True)
        self.assertEqual(bv2pysmt(Ite(b1xx, bxx, byy)), fm.Ite(fm.Equals(p1xx, fm.BV(1, 1)), pxx, pyy))
        self.assertEqual(bv2pysmt(Ite(b1xx, b1yy, b1zz), True), fm.Ite(fm.Equals(p1xx, fm.BV(1, 1)), p1yy, p1zz))


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(arxpy.smt.types))
    return tests
