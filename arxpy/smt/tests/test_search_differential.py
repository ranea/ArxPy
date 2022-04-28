"""Tests for the Optimization module."""
import doctest
import collections
import functools
import math
import unittest

from arxpy.bitvector.operation import Concat, BvComp

from arxpy.primitives.primitives import BvFunction, Cipher

from arxpy.differential.difference import XorDiff, RXDiff
from arxpy.differential.characteristic import (
    BvCharacteristic, SingleKeyCh, RelatedKeyCh
)

import arxpy.smt.search_differential
from arxpy.smt.search_differential import (
    SearchCh, SearchSkCh, SearchRkCh, DerMode, ChSearchMode, SkChSearchMode, RkChSearchMode, _get_smart_print
)

from arxpy.differential.tests.test_characteristic import (
    MyFunction, Cipher1, Cipher2
)


VERBOSE_LEVEL = 0
# 0: quiet
# 1: basic info
# 2: + ssa, model
# 3: + hrepr
# 4: + full hrepr


ChOption = collections.namedtuple('ChOptions', ['der_mode', 'search_mode'])
SkChOption = collections.namedtuple('SingleKeyOptions', ['der_mode', 'search_mode'])
RkChOption = collections.namedtuple('RelatedKeyOptions', ['key_der_mode', 'enc_der_mode',
                                                          'search_mode', 'initial_ew', 'initial_kw'])


CH_OPTIONS = [
    ChOption(DerMode.ProbabilityOne, ChSearchMode.Optimal),
    ChOption(DerMode.XDCA_Approx, ChSearchMode.FirstCh),
    ChOption(DerMode.Default, ChSearchMode.Optimal),
    ChOption(DerMode.Default, ChSearchMode.OptimalDifferential),
]

SKCH_OPTIONS = [
    SkChOption(DerMode.ProbabilityOne, SkChSearchMode.Optimal),
    SkChOption(DerMode.XDCA_Approx, SkChSearchMode.FirstCh),
    SkChOption(DerMode.Default, SkChSearchMode.Optimal),
    SkChOption(DerMode.Default, SkChSearchMode.OptimalDifferential),
]

RKCH_OPTIONS = [
    RkChOption(DerMode.ProbabilityOne, DerMode.ProbabilityOne, RkChSearchMode.OptimalMinSum, 0, 0),
    RkChOption(DerMode.ProbabilityOne, DerMode.ProbabilityOne, RkChSearchMode.OptimalMinSumDifferential, 0, 0),
    RkChOption(DerMode.Default, DerMode.Default, RkChSearchMode.FirstMinSum, 0, 0),
    RkChOption(DerMode.ProbabilityOne, DerMode.Default, RkChSearchMode.FirstValidKeyMinEnc, 0, 0),
    RkChOption(DerMode.Default, DerMode.Default, RkChSearchMode.OptimalFixEncMinKey, 1, 0),
]


def test_search_ch_skch(bvf_cipher, diff_type, initial_weight, solver_name, rounds, der_mode, search_mode, check,
                        verbose_level, filename):
    smart_print = _get_smart_print(filename)

    if rounds is not None:
        bvf_cipher.set_rounds(rounds)

    if issubclass(bvf_cipher, BvFunction):
        num_inputs = len(bvf_cipher.input_widths)
        input_diff_names = ["dp" + str(i) for i in range(num_inputs)]
        ch = BvCharacteristic(bvf_cipher, diff_type, input_diff_names)
    else:
        assert issubclass(bvf_cipher, Cipher)
        ch = SingleKeyCh(bvf_cipher, diff_type)

    if verbose_level >= 1:
        str_rounds = "" if rounds is None else "{} rounds".format(rounds)
        smart_print(str_rounds, bvf_cipher.__name__, diff_type.__name__, type(ch).__name__)
        if verbose_level >= 2:
            smart_print("Characteristic:")
            smart_print(ch)

    if issubclass(bvf_cipher, BvFunction):
        problem = SearchCh(ch, der_mode=der_mode)
    else:
        problem = SearchSkCh(ch, der_mode=der_mode)

    if verbose_level >= 1:
        smart_print(type(problem).__name__, der_mode, search_mode, solver_name,
                    "size:", problem.formula_size())
        if verbose_level >= 2:
            smart_print(problem.hrepr(verbose_level >= 3))

    sol = problem.solve(initial_weight,
                        solver_name=solver_name,
                        search_mode=search_mode,
                        check=check,
                        verbose_level=verbose_level,
                        filename=filename)

    if verbose_level >= 1:
        if sol is None:
            smart_print("\nUnsatisfiable")
        else:
            smart_print("\nSolution:")
            smart_print(sol)
            if verbose_level >= 2:
                if isinstance(sol, collections.abc.Sequence):
                    # for search_mode TopDifferentials
                    smart_print(sol[0].vrepr())
                else:
                    smart_print(sol.vrepr())
        smart_print()

    return sol


def test_search_related_key_ch(cipher, diff_type, initial_ew, initial_kw, solver_name,
                               rounds, key_der_mode, enc_der_mode, allow_zero_enc_input_diff,
                               search_mode, check, verbose_level, filename):
    # assert search_mode != RkChSearchMode.AllValid

    smart_print = _get_smart_print(filename)

    if rounds is not None:
        cipher.set_rounds(rounds)

    ch = RelatedKeyCh(cipher, diff_type)

    if verbose_level >= 1:
        smart_print(rounds, "round(s)", cipher.__name__, diff_type.__name__, type(ch).__name__)
        if verbose_level >= 2:
            smart_print("Characteristic:")
            smart_print(ch)

    problem = SearchRkCh(rkch=ch, key_der_mode=key_der_mode, enc_der_mode=enc_der_mode,
                         allow_zero_enc_input_diff=allow_zero_enc_input_diff)

    if verbose_level >= 1:
        smart_print(type(problem).__name__, "key/enc mode:", key_der_mode, enc_der_mode,
                    search_mode, solver_name, "size:", problem.formula_size())
        if verbose_level >= 2:
            smart_print(problem.hrepr(verbose_level >= 3))

    sol = problem.solve(
        initial_ew=initial_ew,
        initial_kw=initial_kw,
        solver_name=solver_name,
        search_mode=search_mode,
        check=check,
        verbose_level=verbose_level,
        filename=filename)

    if verbose_level >= 1:
        if sol is None:
            smart_print("\nUnsatisfiable")
        else:
            smart_print("\nSolution:")
            smart_print(sol)
            if verbose_level >= 2:
                smart_print(sol.vrepr())
        smart_print()

    return sol


class TestSMT(unittest.TestCase):
    """Tests of the SMT classes."""

    def test_SearchCh(self):
        for diff_type in [XorDiff, RXDiff]:
            for bv_function in [MyFunction]:
                for option in CH_OPTIONS:
                    btor_ch_found = test_search_ch_skch(
                        bvf_cipher=bv_function,
                        diff_type=diff_type,
                        initial_weight=0,
                        solver_name="btor",
                        rounds=None,
                        der_mode=option.der_mode,
                        search_mode=option.search_mode,
                        check=True,
                        verbose_level=VERBOSE_LEVEL,
                        filename=None
                    )

                    if btor_ch_found is not None:
                        btor_weight = int(btor_ch_found.ch_weight)
                        btor_id = [value.val for var, value in btor_ch_found.input_diff]
                        self.assertFalse(BvComp(functools.reduce(Concat, btor_id), 0))
                    else:
                        btor_weight = math.inf

                    if diff_type == XorDiff:
                        self.assertEqual(0, btor_weight)
                    elif diff_type == RXDiff:
                        if option.der_mode == DerMode.ProbabilityOne:
                            self.assertEqual(math.inf, btor_weight)
                        else:
                            self.assertLessEqual(1, btor_weight)
                            self.assertLessEqual(btor_weight, 3)

                    yices_ch_found = test_search_ch_skch(
                        bvf_cipher=bv_function,
                        diff_type=diff_type,
                        initial_weight=0,
                        solver_name="yices",
                        rounds=None,
                        der_mode=option.der_mode,
                        search_mode=option.search_mode,
                        check=True,
                        verbose_level=1 if VERBOSE_LEVEL >= 2 else 0,
                        filename=None
                    )

                    if yices_ch_found is not None:
                        yices_weight = int(yices_ch_found.ch_weight)
                        yices_id = [value.val for var, value in yices_ch_found.input_diff]
                        self.assertFalse(BvComp(functools.reduce(Concat, yices_id), 0))
                    else:
                        yices_weight = math.inf

                    self.assertEqual(btor_weight, yices_weight)

    def test_SearchSkCh(self):
        diff_type = XorDiff

        for cipher in [Cipher1, Cipher2]:
            cipher.set_rounds(1)
            for option in SKCH_OPTIONS:
                btor_ch_found = test_search_ch_skch(
                    bvf_cipher=cipher,
                    diff_type=diff_type,
                    initial_weight=0,
                    solver_name="btor",
                    rounds=1,
                    der_mode=option.der_mode,
                    search_mode=option.search_mode,
                    check=True,
                    verbose_level=VERBOSE_LEVEL,
                    filename=None
                )

                if btor_ch_found is not None:
                    btor_weight = int(btor_ch_found.ch_weight)
                    btor_id = [value.val for var, value in btor_ch_found.input_diff]
                    self.assertFalse(BvComp(functools.reduce(Concat, btor_id), 0))
                else:
                    btor_weight = math.inf

                self.assertEqual(0, btor_weight)

                yices_ch_found = test_search_ch_skch(
                    bvf_cipher=cipher,
                    diff_type=diff_type,
                    initial_weight=0,
                    solver_name="yices",
                    rounds=1,
                    der_mode=option.der_mode,
                    search_mode=option.search_mode,
                    check=True,
                    verbose_level=1 if VERBOSE_LEVEL >= 2 else 0,
                    filename=None
                )

                if yices_ch_found is not None:
                    yices_weight = int(yices_ch_found.ch_weight)
                    yices_id = [value.val for var, value in yices_ch_found.input_diff]
                    self.assertFalse(BvComp(functools.reduce(Concat, yices_id), 0))
                else:
                    yices_weight = math.inf

                self.assertEqual(btor_weight, yices_weight)

    def test_SearchRkCh(self):
        for diff_type in [XorDiff, RXDiff]:
            for cipher in [Cipher1, Cipher2]:
                if diff_type == RXDiff and cipher == Cipher2:
                    continue

                cipher.set_rounds(1)
                for option in RKCH_OPTIONS:
                    if diff_type == RXDiff and option.enc_der_mode == RkChSearchMode.OptimalFixEncMinKey:
                        continue

                    btor_rkch_found = test_search_related_key_ch(
                        cipher=cipher,
                        diff_type=diff_type,
                        initial_ew=option.initial_ew,
                        initial_kw=option.initial_kw,
                        solver_name="btor",
                        rounds=1,
                        key_der_mode=option.key_der_mode,
                        enc_der_mode=option.enc_der_mode,
                        allow_zero_enc_input_diff=True,
                        search_mode=option.search_mode,
                        check=True,
                        verbose_level=VERBOSE_LEVEL,
                        filename=None
                    )

                    if btor_rkch_found is not None:
                        btor_weight = [
                            int(btor_rkch_found.key_ch_found.ch_weight),
                            int(btor_rkch_found.enc_ch_found.ch_weight)
                        ]
                        btor_id = [value.val for var, value in btor_rkch_found.key_ch_found.input_diff]
                        self.assertFalse(BvComp(functools.reduce(Concat, btor_id), 0))
                    else:
                        btor_weight = [math.inf, math.inf]

                    if diff_type == XorDiff:
                        self.assertEqual([option.initial_kw, option.initial_ew], btor_weight)
                    elif diff_type == RXDiff:
                        if option.enc_der_mode == DerMode.ProbabilityOne:
                            self.assertEqual([math.inf, math.inf], btor_weight)
                        elif option.search_mode in [RkChSearchMode.FirstMinSum, RkChSearchMode.FirstValidKeyMinEnc]:
                            self.assertEqual([0, 1], btor_weight)

                    yices_rkch_found = test_search_related_key_ch(
                        cipher=cipher,
                        diff_type=diff_type,
                        initial_ew=option.initial_ew,
                        initial_kw=option.initial_kw,
                        solver_name="yices",
                        rounds=1,
                        key_der_mode=option.key_der_mode,
                        enc_der_mode=option.enc_der_mode,
                        allow_zero_enc_input_diff=True,
                        search_mode=option.search_mode,
                        check=True,
                        verbose_level=1 if VERBOSE_LEVEL >= 2 else 0,
                        filename=None
                    )

                    if yices_rkch_found is not None:
                        yices_weight = [
                            int(yices_rkch_found.key_ch_found.ch_weight),
                            int(yices_rkch_found.enc_ch_found.ch_weight)
                        ]
                        yices_id = [value.val for var, value in yices_rkch_found.key_ch_found.input_diff]
                        self.assertFalse(BvComp(functools.reduce(Concat, yices_id), 0))
                    else:
                        yices_weight = [math.inf, math.inf]

                    if option.search_mode in [RkChSearchMode.FirstValidKeyMinEnc, RkChSearchMode.OptimalValidKeyMinEnc]:
                        # ignore key weights
                        self.assertEqual(btor_weight[1], yices_weight[1])
                    else:
                        self.assertEqual(btor_weight, yices_weight)


def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(arxpy.smt.search_differential))
    return tests
