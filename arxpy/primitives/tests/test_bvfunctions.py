"""Tests for cryptographic primitives."""
import math
from datetime import datetime
import collections
import functools
import os
import unittest

from arxpy.bitvector.operation import Concat, BvComp

from arxpy.differential.difference import XorDiff

from arxpy.smt.search import DerMode, ChSearchMode, _get_smart_print
from arxpy.smt.tests.test_search import test_search_ch_skch

from arxpy.primitives.chaskey import ChaskeyPi
from arxpy.primitives.picipher import PiPermutation

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
BvFunction = collections.namedtuple('BVF', ['function', 'rounds'])


Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
Simon32 = simon.get_Simon_instance(simon.SimonInstance.simon_32_64)
Simeck32 = simeck.get_Simeck_instance(simeck.SimeckInstance.simeck_32_64)
Cham64 = cham.get_Cham_instance(cham.ChamInstance.cham_64_128)


BV_FUNCTIONS = [
    BvFunction(ChaskeyPi, 1),
    BvFunction(PiPermutation, 1),  # optimal finds too many ch
    BvFunction(Speck32.key_schedule, 5),
    BvFunction(Simon32.key_schedule, 10),  # linear ks
    BvFunction(Simeck32.key_schedule, 4),
    BvFunction(Cham64.key_schedule, 10),  # linear ks
    BvFunction(HightCipher.key_schedule, HightCipher.rounds - 1),
    BvFunction(LeaCipher.key_schedule, 2),
    BvFunction(Shacal1Cipher.key_schedule, 19),
    BvFunction(Shacal2Cipher.key_schedule, 18),  # r=19, m=Optimal takes long time
    BvFunction(FealCipher.key_schedule, 1),
    BvFunction(TeaCipher.key_schedule, 1),
    BvFunction(XteaCipher.key_schedule, XteaCipher.key_schedule.rounds - 1),
    # BvFunction(ThreefishCipher.key_schedule, ?),  # huge block size
]

Option = collections.namedtuple('Option', ['der_mode', 'search_mode'])

NoCheckModes = [ChSearchMode.FirstCh, ChSearchMode.TopDifferentials]

OPTIONS = [
    Option(DerMode.ProbabilityOne, ChSearchMode.Optimal),
    Option(DerMode.XDCA_Approx, ChSearchMode.FirstCh),  # XDCA_Approx + Optimal not efficient
    # Option(DerMode.Default, ChSearchMode.Optimal),  # Optimal + Default not efficient
    Option(DerMode.Default, ChSearchMode.OptimalDifferential),
]


class TestBlockCiphers(unittest.TestCase):
    """Test the block ciphers implemented."""

    @classmethod
    def setUpClass(cls):
        if OUTPUT_FILE:
            date_string = datetime.strftime(datetime.now(), '%d-%H-%M')
            filename = "test_bvfunctions" + date_string + ".txt"
            assert not os.path.isfile(filename)
            cls.filename = filename
        else:
            cls.filename = None

    def setUp(self):
        self.default_rounds = {}
        for bvf in BV_FUNCTIONS:
            self.default_rounds[bvf] = bvf.function.rounds

    def tearDown(self):
        for bvf in BV_FUNCTIONS:
            bvf.function.set_rounds(self.default_rounds[bvf])

    def test_testvectors(self):
        for bvf in BV_FUNCTIONS:
            if hasattr(bvf.function, 'test'):
                bvf.function.test()

    def test_zero_input(self):
        for bvf in BV_FUNCTIONS:
            for r in range(self.default_rounds[bvf] // 2, self.default_rounds[bvf]):
                bvf.function(*[0 for _ in bvf.function.input_widths])

    @unittest.skip("test_search_Ch")
    def test_search_Ch(self):
        diff_type = XorDiff

        for bvf in BV_FUNCTIONS:
            if bvf.rounds is None:
                continue
            for rounds in range(bvf.rounds, bvf.rounds + 2):
                bvf.function.set_rounds(rounds)
                for option in OPTIONS:
                    if bvf.function == PiPermutation and option == ChSearchMode.Optimal:
                        continue

                    btor_ch_found = test_search_ch_skch(
                        bvf_cipher=bvf.function,
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
                        bvf_cipher=bvf.function,
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

    @unittest.skip("test_verbose_search_Ch")
    def test_verbose_search_Ch(self):
        smart_print = _get_smart_print(self.filename)

        diff_type = XorDiff

        for verbose_level in range(1, 4):
            smart_print("VERBOSE LEVEL:", verbose_level)

            bvf = BvFunction(LeaCipher.key_schedule, 1)
            for option in [ChSearchMode.FirstCh, ChSearchMode.OptimalDifferential, ChSearchMode.TopDifferentials]:
                test_search_ch_skch(
                    bvf_cipher=bvf.function,
                    diff_type=diff_type,
                    initial_weight=3,
                    solver_name="btor",
                    rounds=bvf.rounds,
                    der_mode=DerMode.XDCA_Approx,
                    search_mode=option,
                    check=False if option in NoCheckModes else True,
                    verbose_level=verbose_level,
                    filename=self.filename
                )

                smart_print("\n~~\n")

            smart_print("\n\n-----\n\n")