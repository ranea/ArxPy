"""Tests for cryptographic primitives."""
import math
from datetime import datetime
import collections
import functools
import os
import unittest

from arxpy.bitvector.operation import Concat, BvComp

from arxpy.differential.difference import XorDiff

from arxpy.smt.search import (
    DerMode, SkChSearchMode, RkChSearchMode,
    round_based_search_SkCh, round_based_search_RkCh,
    _get_smart_print
)
from arxpy.smt.tests.test_search import test_search_ch_skch, test_search_related_key_ch

from arxpy.primitives import speck
from arxpy.primitives import simon
from arxpy.primitives import simeck
from arxpy.primitives.hight import HightCipher
from arxpy.primitives.lea import LeaCipher
from arxpy.primitives.shacal1 import Shacal1Cipher
from arxpy.primitives.shacal2 import Shacal2Cipher
from arxpy.primitives.feal import FealCipher
from arxpy.primitives.tea import TeaCipher
from arxpy.primitives.xtea import XteaCipher
from arxpy.primitives.multi2 import Multi2Cipher
from arxpy.primitives import cham
# from arxpy.primitives.threefish import ThreefishCipher


OUTPUT_FILE = False
VERBOSE_LEVEL = 0
# 0: quiet
# 1: basic info
# 2: + ssa, model
# 3: + hrepr
# 4: + full hrepr
CHECK = True


# *_rounds should have trivial characteristics but *_rounds + 1 not (with ProbabilityOne)
BlockCipher = collections.namedtuple('BlockCipher', ['cipher', 'sk_rounds', 'rk_rounds'])


Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
Simon32 = simon.get_Simon_instance(simon.SimonInstance.simon_32_64)
Simeck32 = simeck.get_Simeck_instance(simeck.SimeckInstance.simeck_32_64)
Cham64 = cham.get_Cham_instance(cham.ChamInstance.cham_64_128)



BLOCK_CIPHERS = [
    BlockCipher(Speck32, 1, 4),
    BlockCipher(Simon32, 1, 4),
    BlockCipher(Simeck32, 1, 4),
    BlockCipher(Cham64, 4, 6),
    BlockCipher(HightCipher, 2, None),  # for rk_rounds=7, ExactWeightError is raised
    BlockCipher(LeaCipher, 2, 1),
    BlockCipher(Shacal1Cipher, 1, 16),
    BlockCipher(Shacal2Cipher, 1, None),  # for rk_rounds=18, ExactWeightError is raised
    BlockCipher(FealCipher, 3, 1),
    BlockCipher(TeaCipher, 1, TeaCipher.rounds - 1),
    BlockCipher(XteaCipher, 1, 8),
    BlockCipher(Multi2Cipher, 2, 2),
    # BlockCipher(ThreefishCipher, ?, ?),  # huge block size
]

SKOption = collections.namedtuple('SingleKeyOptions', ['der_mode', 'search_mode'])
RKOption = collections.namedtuple('RelatedKeyOptions', ['key_der_mode', 'enc_der_mode',
                                                        'search_mode', 'initial_ew', 'initial_kw'])

NoCheckModes = [SkChSearchMode.FirstCh, RkChSearchMode.FirstMinSum]

SK_OPTIONS = [
    SKOption(DerMode.ProbabilityOne, SkChSearchMode.Optimal),
    SKOption(DerMode.XDCA_Approx, SkChSearchMode.FirstCh),  # XDCA_Approx + Optimal not efficient
    # SKOption(DerMode.Default, SkChSearchMode.Optimal),  # Optimal + Default not efficient
    SKOption(DerMode.Default, SkChSearchMode.OptimalDifferential),
]

RK_OPTIONS = [
    # RKOption(DerMode.ProbabilityOne, DerMode.ProbabilityOne, RkChSearchMode.OptimalMinSum, 0, 0),
    RKOption(DerMode.ProbabilityOne, DerMode.ProbabilityOne, RkChSearchMode.OptimalMinSumDifferential, 0, 0),
    RKOption(DerMode.XDCA_Approx, DerMode.XDCA_Approx, RkChSearchMode.FirstMinSum, 0, 0),
    RKOption(DerMode.Default, DerMode.Default, RkChSearchMode.OptimalValidKeyMinEnc, 0, 0),
    RKOption(DerMode.Default, DerMode.Default, RkChSearchMode.OptimalFixEncMinKey, 1, 0),
]


class TestBlockCiphers(unittest.TestCase):
    """Test the block ciphers implemented."""

    @classmethod
    def setUpClass(cls):
        if OUTPUT_FILE:
            date_string = datetime.strftime(datetime.now(), '%d-%H-%M')
            filename = "test_ciphers" + date_string + ".txt"
            assert not os.path.isfile(filename)
            cls.filename = filename
        else:
            cls.filename = None

    def setUp(self):
        self.default_rounds = {}
        for bc in BLOCK_CIPHERS:
            self.default_rounds[bc] = bc.cipher.rounds

    def tearDown(self):
        for bc in BLOCK_CIPHERS:
            bc.cipher.set_rounds(self.default_rounds[bc])
            bc.cipher.encryption.round_keys = None

    def test_testvectors(self):
        for bc in BLOCK_CIPHERS:
            bc.cipher.test()

    def test_zero_input(self):
        for bc in BLOCK_CIPHERS:
            for r in range(self.default_rounds[bc] // 2, self.default_rounds[bc]):
                key = [0 for _ in bc.cipher.key_schedule.input_widths]
                pt = [0 for _ in bc.cipher.encryption.input_widths]
                bc.cipher(pt, key)

    @unittest.skip("test_SearchSkCh")
    def test_search_SkCh(self):
        diff_type = XorDiff

        for bc in BLOCK_CIPHERS:
            if bc.sk_rounds is None:
                continue
            for rounds in range(bc.sk_rounds, bc.sk_rounds + 2):
                bc.cipher.set_rounds(rounds)
                for option in SK_OPTIONS:
                    btor_ch_found = test_search_ch_skch(
                        bvf_cipher=bc.cipher,
                        diff_type=diff_type,
                        initial_weight=0,
                        solver_name="btor",
                        rounds=rounds,
                        der_mode=option.der_mode,
                        search_mode=option.search_mode,
                        check=False if option.search_mode in NoCheckModes else CHECK,
                        verbose_level=VERBOSE_LEVEL,
                        filename=self.filename
                    )

                    if btor_ch_found is not None:
                        btor_weight = int(btor_ch_found.ch_weight)
                        btor_id = [value.val for var, value in btor_ch_found.input_diff]
                        self.assertFalse(BvComp(functools.reduce(Concat, btor_id), 0))
                    else:
                        btor_weight = math.inf

                    yices_ch_found = test_search_ch_skch(
                        bvf_cipher=bc.cipher,
                        diff_type=diff_type,
                        initial_weight=0,
                        solver_name="yices",
                        rounds=rounds,
                        der_mode=option.der_mode,
                        search_mode=option.search_mode,
                        check=False,
                        verbose_level=1 if VERBOSE_LEVEL >=2 else 0,
                        filename=self.filename
                    )

                    if yices_ch_found is not None:
                        yices_weight = int(yices_ch_found.ch_weight)
                        yices_id = [value.val for var, value in yices_ch_found.input_diff]
                        self.assertFalse(BvComp(functools.reduce(Concat, yices_id), 0))
                    else:
                        yices_weight = math.inf

                    self.assertEqual(btor_weight, yices_weight)

    @unittest.skip("test_SearchRkCh")
    def test_search_RkCh(self):
        diff_type = XorDiff

        for bc in BLOCK_CIPHERS:
            if bc.rk_rounds is None:
                continue
            for rounds in range(bc.rk_rounds, bc.rk_rounds + 2):
                bc.cipher.set_rounds(rounds)
                for option in RK_OPTIONS:
                    btor_rkch_found = test_search_related_key_ch(
                        cipher=bc.cipher,
                        diff_type=diff_type,
                        initial_ew=option.initial_ew,
                        initial_kw=option.initial_kw,
                        solver_name="btor",
                        rounds=rounds,
                        key_der_mode=option.key_der_mode,
                        enc_der_mode=option.enc_der_mode,
                        allow_zero_enc_input_diff=True,
                        search_mode=option.search_mode,
                        check=False if option.search_mode in NoCheckModes else CHECK,
                        verbose_level=VERBOSE_LEVEL,
                        filename=self.filename
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

                    yices_rkch_found = test_search_related_key_ch(
                        cipher=bc.cipher,
                        diff_type=diff_type,
                        initial_ew=option.initial_ew,
                        initial_kw=option.initial_kw,
                        solver_name="yices",
                        rounds=rounds,
                        key_der_mode=option.key_der_mode,
                        enc_der_mode=option.enc_der_mode,
                        allow_zero_enc_input_diff=True,
                        search_mode=option.search_mode,
                        check=False,
                        verbose_level=1 if VERBOSE_LEVEL >=2 else 0,
                        filename=self.filename
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
                        self.assertAlmostEqual(sum(btor_weight), sum(yices_weight))

    @unittest.skip("test_round_search_Sk")
    def test_round_search_Sk(self):
        diff_type = XorDiff

        for bc in BLOCK_CIPHERS:
            if bc.sk_rounds is None:
                continue
            for option in SK_OPTIONS:
                round_based_search_SkCh(cipher=bc.cipher,
                                        diff_type=diff_type,
                                        initial_weight=0,
                                        solver_name="btor",
                                        start_round=bc.sk_rounds,
                                        end_round=bc.sk_rounds + 1,
                                        der_mode=option.der_mode,
                                        search_mode=option.search_mode,
                                        check=False if option.search_mode in NoCheckModes else CHECK,
                                        verbose_level=VERBOSE_LEVEL,
                                        filename=self.filename)

    @unittest.skip("test_round_search_Rk")
    def test_round_search_Rk(self):
        diff_type = XorDiff

        for bc in BLOCK_CIPHERS:
            if bc.rk_rounds is None:
                continue
            for option in RK_OPTIONS:
                round_based_search_RkCh(cipher=bc.cipher,
                                        diff_type=diff_type,
                                        initial_ew=option.initial_ew,
                                        initial_kw=option.initial_kw,
                                        solver_name="btor",
                                        start_round=bc.rk_rounds,
                                        end_round=bc.rk_rounds + 1,
                                        key_der_mode=option.key_der_mode,
                                        enc_der_mode=option.enc_der_mode,
                                        allow_zero_enc_input_diff=True,
                                        search_mode=option.search_mode,
                                        check=False if option.search_mode in NoCheckModes else CHECK,
                                        verbose_level=VERBOSE_LEVEL,
                                        filename=self.filename)

    @unittest.skip("test_verbose_search_Ch")
    def test_verbose_search_RkCh(self):
        smart_print = _get_smart_print(self.filename)

        diff_type = XorDiff

        for verbose_level in range(1, 3):
            smart_print("VERBOSE LEVEL:", verbose_level)

            bc = BlockCipher(LeaCipher, None, 1)
            for option in [RkChSearchMode.FirstMinSum, RkChSearchMode.OptimalMinSumDifferential]:
                 test_search_related_key_ch(
                    cipher=bc.cipher,
                    diff_type=diff_type,
                    initial_ew=3,
                    initial_kw=0,
                    solver_name="btor",
                    rounds=bc.rk_rounds,
                    key_der_mode=DerMode.XDCA_Approx,
                    enc_der_mode=DerMode.Default,
                    allow_zero_enc_input_diff=True,
                    search_mode=option,
                    check=False if option in NoCheckModes else True,
                    verbose_level=verbose_level,
                    filename=self.filename
                 )
                 smart_print("\n~~\n")

            smart_print("\n\n-----\n\n")