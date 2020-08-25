"""Tests for the derivative module."""
import collections
import doctest
import importlib
import itertools
import math
import tempfile
import unittest

import cffi
from hypothesis import given, example, settings  # unlimited, HealthCheck
# useful hypothesis decorator
# @example(arg0=..., arg1=..., ...)
# @settings(max_examples=10000, deadline=None)
from hypothesis.strategies import integers

from arxpy.bitvector.core import Constant
from arxpy.bitvector.context import Validation, Simplification

from arxpy.differential.difference import XorDiff, RXDiff, Difference
from arxpy.differential import derivative
from arxpy.differential.derivative import XDA, XDS, RXDA

from arxpy.differential.tests import preimageXDA
from arxpy.differential.tests import preimageRXDA
from arxpy.differential.tests import preimageXDAC
from arxpy.differential.tests import preimageXDS


DP_WIDTH = 8
VERBOSE = False
PRINT_DISTR_ERROR = False
PRINT_MAX_ERROR = False
ERROR_DIGITS = 5  # num fraction digits used when printing error values


class TestDerivativeBvAdd(unittest.TestCase):
    """Tests for the Derivative of BvAdd."""

    @classmethod
    def setUpClass(cls):
        module_name = "_preimageXDA"
        ffibuilderXOR = cffi.FFI()
        ffibuilderXOR.cdef(preimageXDA.header)
        ffibuilderXOR.set_source(module_name, preimageXDA.source)

        cls.tmpdirnameXOR = tempfile.TemporaryDirectory()
        lib_path = ffibuilderXOR.compile(tmpdir=cls.tmpdirnameXOR.name, verbose=VERBOSE)
        spec = importlib.util.spec_from_file_location(module_name, lib_path)
        lib_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lib_module)
        cls.libXDA = lib_module.lib

        module_name = "_preimageRXDA"
        ffibuilderRX = cffi.FFI()
        ffibuilderRX.cdef(preimageRXDA.header)
        ffibuilderRX.set_source(module_name, preimageRXDA.source)

        cls.tmpdirnameRX = tempfile.TemporaryDirectory()
        lib_path = ffibuilderRX.compile(tmpdir=cls.tmpdirnameRX.name, verbose=VERBOSE)
        spec = importlib.util.spec_from_file_location(module_name, lib_path)
        lib_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lib_module)
        cls.libRXDA = lib_module.lib

    @classmethod
    def tearDownClass(cls):
        cls.tmpdirnameXOR.cleanup()
        cls.tmpdirnameRX.cleanup()

    @classmethod
    def _find_preimage(cls, f, beta):
        width = f.input_diff[0].val.width

        if width == 8 and f.diff_type == XorDiff:
            foo = cls.libXDA.find_XOR_preimage_8bit
        elif width == 8 and f.diff_type == RXDiff:
            foo = cls.libRXDA.find_RX_preimage_8bit
        elif width == 16 and f.diff_type == RXDiff:
            foo = cls.libRXDA.find_RX_preimage_16bit
        else:
            foo = None

        if foo is not None:
            result = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            if result.found:
                return result.i, result.j
            else:
                return None
        else:
            for i, j in itertools.product(range(2 ** width), range(2 ** width)):
                output_diff = f.eval(Constant(i, width), Constant(j, width))
                assert isinstance(output_diff, Difference) and isinstance(beta, Difference)
                if output_diff == beta:
                    return i, j
            else:
                return None

    @unittest.skip("skipping test_find_preimage")
    @given(
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
    )
    def test_find_preimage(self, d1, d2, d3):
        assert DP_WIDTH == 8

        d1 = Constant(d1, DP_WIDTH)
        d2 = Constant(d2, DP_WIDTH)
        d3 = Constant(d3, DP_WIDTH)

        for diff_type, der_type in zip([XorDiff, RXDiff], [XDA, RXDA]):
            alpha = diff_type(d1), diff_type(d2)
            beta = diff_type(d3)
            f = der_type(alpha)

            msg = "{}({} -> {})\n".format(der_type.__name__, alpha, beta)

            if diff_type == XorDiff:
                foo = self.__class__.libXDA.find_XOR_preimage_8bit
            elif diff_type == RXDiff:
                foo = self.__class__.libRXDA.find_RX_preimage_8bit

            result_lib = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            result_lib = result_lib.found

            result_loop = 0
            for i, j in itertools.product(range(2 ** DP_WIDTH), range(2 ** DP_WIDTH)):
                output_diff = f.eval(Constant(i, DP_WIDTH), Constant(j, DP_WIDTH))
                assert isinstance(output_diff, Difference) and isinstance(beta, Difference)
                if output_diff == beta:
                    result_loop = 1
                    break

            self.assertEqual(result_lib, result_loop, msg=msg)

    @classmethod
    def _count_preimages(cls, f, beta):
        width = f.input_diff[0].val.width

        if width == 8 and f.diff_type == XorDiff:
            foo = cls.libXDA.count_XOR_preimage_8bit
        elif width == 8 and f.diff_type == RXDiff:
            foo = cls.libRXDA.count_RX_preimage_8bit
        elif width == 16 and f.diff_type == RXDiff:
            foo = cls.libRXDA.count_RX_preimage_16bit
        else:
            foo = None

        if foo is not None:
            result = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            return int(result)
        else:
            num_preimages = 0
            # total_preimages = 0
            for i, j in itertools.product(range(2 ** width), range(2 ** width)):
                # total_preimages += 1
                output_diff = f.eval(Constant(i, width), Constant(j, width))
                assert isinstance(output_diff, Difference) and isinstance(beta, Difference)
                if output_diff == beta:
                    num_preimages += 1
            return num_preimages

    @unittest.skip("skipping test_count_preimage")
    @given(
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
    )
    def test_count_preimage(self, d1, d2, d3):
        assert DP_WIDTH == 8

        d1 = Constant(d1, DP_WIDTH)
        d2 = Constant(d2, DP_WIDTH)
        d3 = Constant(d3, DP_WIDTH)

        for diff_type, der_type in zip([XorDiff, RXDiff], [XDA, RXDA]):
            alpha = diff_type(d1), diff_type(d2)
            beta = diff_type(d3)
            f = der_type(alpha)

            msg = "{}({} -> {})\n".format(der_type.__name__, alpha, beta)

            if diff_type == XorDiff:
                foo = self.__class__.libXDA.count_XOR_preimage_8bit
            elif diff_type == RXDiff:
                foo = self.__class__.libRXDA.count_RX_preimage_8bit
            else:
                raise ValueError("invalid diff_type")

            result_lib = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            result_lib = int(result_lib)

            result_loop = 0
            for i, j in itertools.product(range(2 ** DP_WIDTH), range(2 ** DP_WIDTH)):
                output_diff = f.eval(Constant(i, DP_WIDTH), Constant(j, DP_WIDTH))
                assert isinstance(output_diff, Difference) and isinstance(beta, Difference)
                if output_diff == beta:
                    result_loop += 1

            self.assertEqual(result_lib, result_loop, msg=msg)

    @given(
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
    )
    def test_XDA(self, d1, d2, d3):
        # d1, d2 input differences; d3 output difference
        d1 = Constant(d1, DP_WIDTH)
        d2 = Constant(d2, DP_WIDTH)
        d3 = Constant(d3, DP_WIDTH)

        for diff_type, der_type in zip([XorDiff], [XDA]):
            alpha = diff_type(d1), diff_type(d2)
            beta = diff_type(d3)
            f = der_type(alpha)

            if VERBOSE:
                der_name = der_type.__name__
                print("{}({} -> {})".format(der_name, alpha, beta))

            is_valid = f.is_possible(beta)

            if VERBOSE:
                print("\tis_valid:", bool(is_valid))

            if is_valid:
                with Simplification(False):
                    num_preimages = self._count_preimages(f, beta)

                if VERBOSE:
                    print("\tpreimages found:", num_preimages)

                self.assertNotEqual(num_preimages, 0)

                total_preimages = 2**(2 * DP_WIDTH)

                emp_weight = - math.log(num_preimages / total_preimages, 2)

                theo_weight = f.weight(beta)  # .doit()
                self.assertLessEqual(theo_weight, f.max_weight())

                theo_weight = int(theo_weight)

                if VERBOSE:
                    print("\tempirical weight:\t", emp_weight)
                    print("\ttheoretical weight:\t", theo_weight)

                if diff_type == XorDiff:
                    error = 0
                else:
                    raise ValueError("invalid difference: {}".format(diff_type))

                self.assertGreaterEqual(emp_weight, theo_weight - error)
                self.assertLessEqual(emp_weight, theo_weight + error)
            else:
                with Simplification(False):
                    preimage = self._find_preimage(f, beta)

                if VERBOSE:
                    print("\tpreimage found:", preimage)

                self.assertIsNone(preimage)

    @unittest.skip("skipping test_RXDA")
    @given(
        integers(min_value=0, max_value=2 ** (2*DP_WIDTH) - 1),
        integers(min_value=0, max_value=2 ** (2*DP_WIDTH) - 1),
        integers(min_value=0, max_value=2 ** (2*DP_WIDTH) - 1),
        integers(min_value=0, max_value=10),
    )
    # @settings(deadline=None, max_examples=10000)
    def test_RXDA(self, d1, d2, d3, precision):
        width = 2*DP_WIDTH

        d1 = Constant(d1, width)
        d2 = Constant(d2, width)
        d3 = Constant(d3, width)

        diff_type, der_type = RXDiff, RXDA
        old_precision = der_type.precision
        der_type.precision = precision

        alpha = diff_type(d1), diff_type(d2)
        beta = diff_type(d3)
        f = der_type(alpha)

        der_repr = "{}_prec={}({} -> {})\n".format(der_type.__name__, precision, alpha, beta)
        msg = der_repr

        is_valid = f.is_possible(beta)

        msg += "\tis_valid: {}\n".format(is_valid)

        if is_valid:
            exact_weight = f.exact_weight(beta)

            msg += "\texact weight: {}\n".format(exact_weight)

            theo_weight = int(f.weight(beta))

            self.assertLessEqual(theo_weight, f.max_weight(), msg=msg)

            theo_weight /= 2**(f.num_frac_bits())

            msg += "\ttheo weight: {}\n".format(theo_weight)

            with Simplification(False):
                num_preimages = self._count_preimages(f, beta)

            self.assertNotEqual(num_preimages, 0, msg=msg)

            total_preimages = 2 ** (2 * width)
            emp_weight = - math.log(num_preimages / total_preimages, 2)

            msg += "\temp weight: {}\n".format(emp_weight)

            if width < 16:
                # delta found empirically for width = 8
                self.assertAlmostEqual(emp_weight, exact_weight, delta=1.02, msg=msg)
            else:
                self.assertAlmostEqual(emp_weight, exact_weight, msg=msg)

            real_error = abs(emp_weight - theo_weight)
            theo_error = f.error() + 10**(-8)

            msg += "\treal_error: {}\n".format(round(real_error, ERROR_DIGITS))
            msg += "\ttheo_error: {}\n".format(round(theo_error, ERROR_DIGITS))

            self.assertLessEqual(real_error, theo_error, msg=msg)
        else:
            with Simplification(False):
                preimage = self._find_preimage(f, beta)

            msg += "\tpreimage found: {}\n".format(preimage)

            self.assertIsNone(preimage, msg=msg)

        der_type.precision = old_precision


class TestDerivativeBvSub(unittest.TestCase):
    """Tests for the Derivative of BvSub."""

    @classmethod
    def setUpClass(cls):
        module_name = "_preimageXDS"

        ffibuilder = cffi.FFI()
        ffibuilder.cdef(preimageXDS.header)
        ffibuilder.set_source(module_name, preimageXDS.source)

        cls.tmpdirname = tempfile.TemporaryDirectory()

        libxds_path = ffibuilder.compile(tmpdir=cls.tmpdirname.name, verbose=VERBOSE)

        spec = importlib.util.spec_from_file_location(module_name, libxds_path)
        libxds = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(libxds)
        cls.libXDS = libxds.lib

    @classmethod
    def tearDownClass(cls):
        cls.tmpdirname.cleanup()

    @classmethod
    def _find_preimage(cls, f, beta):
        if DP_WIDTH == 8:
            if f.diff_type == XorDiff:
                foo = cls.libXDS.find_XOR_preimage_8bit
            else:
                assert False

            result = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            if result.found:
                return result.i, result.j
            else:
                return None
        else:
            width = f.input_diff[0].val.width
            for i, j in itertools.product(range(2 ** width), range(2 ** width)):
                output_diff = f.eval(Constant(i, width), Constant(j, width))
                assert isinstance(output_diff, XorDiff) and isinstance(beta, XorDiff), "{} {}".format(output_diff, beta)
                if output_diff == beta:
                    return i, j
            else:
                return None

    @classmethod
    def _count_preimages(cls, f, beta):
        if DP_WIDTH == 8:
            if f.diff_type == XorDiff:
                foo = cls.libXDS.count_XOR_preimage_8bit
            else:
                assert False

            result = foo(f.input_diff[0].val, f.input_diff[1].val, beta.val)
            return int(result)
        else:
            width = f.input_diff[0].val.width
            num_preimages = 0
            total_preimages = 0
            for i, j in itertools.product(range(2 ** width), range(2 ** width)):
                total_preimages += 1
                output_diff = f.eval(Constant(i, width), Constant(j, width))
                assert isinstance(output_diff, XorDiff) and isinstance(beta, XorDiff)
                if output_diff == beta:
                    num_preimages += 1
            return num_preimages

    @given(
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
    )
    def test_case(self, d1, d2, d3):
        # d1, d2 input differences; d3 output difference
        d1 = Constant(d1, DP_WIDTH)
        d2 = Constant(d2, DP_WIDTH)
        d3 = Constant(d3, DP_WIDTH)

        for diff_type, der_type in zip([XorDiff], [XDS]):
            alpha = diff_type(d1), diff_type(d2)
            beta = diff_type(d3)
            f = der_type(alpha)

            if VERBOSE:
                der_name = der_type.__name__
                print("{}({} -> {})".format(der_name, alpha, beta))

            is_valid = f.is_possible(beta)

            if VERBOSE:
                print("\tis_valid:", bool(is_valid))

            if is_valid:
                with Simplification(False):
                    num_preimages = self._count_preimages(f, beta)

                if VERBOSE:
                    print("\tpreimages found:", num_preimages)

                self.assertNotEqual(num_preimages, 0)

                total_preimages = 2**(2 * DP_WIDTH)

                emp_weight = - math.log(num_preimages / total_preimages, 2)

                theo_weight = f.weight(beta)  # .doit()
                self.assertLessEqual(theo_weight, f.max_weight())

                theo_weight = int(theo_weight)

                if VERBOSE:
                    print("\tempirical weight:\t", emp_weight)
                    print("\ttheoretical weight:\t", theo_weight)

                if diff_type == XorDiff:
                    error = 0
                else:
                    raise ValueError("invalid difference: {}".format(diff_type))

                self.assertGreaterEqual(emp_weight, theo_weight - error)
                self.assertLessEqual(emp_weight, theo_weight + error)
            else:
                with Simplification(False):
                    preimage = self._find_preimage(f, beta)

                if VERBOSE:
                    print("\tpreimage found:", preimage)

                self.assertIsNone(preimage)


class TestFormulaBvCteAdd(unittest.TestCase):
    """Tests for the Derivative of BvCteAdd."""

    @classmethod
    def setUpClass(cls):
        module_name = "_preimageXDAC"

        ffibuilder = cffi.FFI()
        ffibuilder.cdef(preimageXDAC.header)
        ffibuilder.set_source(module_name, preimageXDAC.source)

        cls.tmpdirname = tempfile.TemporaryDirectory()

        libxdac_path = ffibuilder.compile(tmpdir=cls.tmpdirname.name, verbose=VERBOSE)

        spec = importlib.util.spec_from_file_location(module_name, libxdac_path)
        libxdac = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(libxdac)
        cls.libXDAC = libxdac.lib

    @classmethod
    def tearDownClass(cls):
        cls.tmpdirname.cleanup()

    @classmethod
    def _find_preimage(cls, f, beta):
        if DP_WIDTH == 8:
            foo = cls.libXDAC.find_XOR_preimage_8bit

            result = foo(f.input_diff[0].val, beta.val, f.op.constant)
            if result.found:
                return result.i
            else:
                return None
        else:
            width = f.input_diff[0].val.width
            for i in range(2 ** width):
                i = Constant(i, width)
                output_diff = f.eval(i)
                assert isinstance(output_diff, XorDiff) and isinstance(beta, XorDiff)
                if output_diff == beta:
                    return i
            else:
                return None

    @classmethod
    def _count_preimages(cls, f, beta):
        if DP_WIDTH == 8:
            foo = cls.libXDAC.count_XOR_preimage_8bit

            result = foo(f.input_diff[0].val, beta.val, f.op.constant)
            return int(result)
        else:
            width = f.input_diff[0].val.width
            num_preimages = 0
            total_preimages = 0
            for i in range(2 ** width):
                total_preimages += 1
                output_diff = f.eval(Constant(i, width))
                assert isinstance(output_diff, XorDiff) and isinstance(beta, XorDiff)
                if output_diff == beta:
                    num_preimages += 1
            return num_preimages

    def _test_case(self, d1, d2, a):
        d1 = Constant(d1, DP_WIDTH)
        d2 = Constant(d2, DP_WIDTH)
        a = Constant(a, DP_WIDTH)

        diff_type = XorDiff
        der_type = derivative.XDCA

        alpha = diff_type(d1)
        beta = diff_type(d2)
        f = der_type(alpha, a)

        msg = "{}_{}_({} -> {})\n".format(der_type.__name__, a, alpha, beta)

        formula_weight = f.exact_weight(beta)
        is_valid = formula_weight != math.inf

        msg += "\tis_valid: {}\n".format(is_valid)

        if is_valid:
            with Simplification(False):
                num_preimages = self._count_preimages(f, beta)

            msg += "\tpreimage found: {}\n".format(num_preimages)

            self.assertNotEqual(num_preimages, 0, msg=msg)

            total_preimages = 2**DP_WIDTH
            empirical_weight = abs(- math.log(num_preimages / total_preimages, 2))

            msg += "\tempirical weight: {}".format(empirical_weight)
            msg += "\tformula weight  : {}".format(formula_weight)

            self.assertAlmostEqual(empirical_weight, formula_weight, msg=msg)
            self.assertLessEqual(empirical_weight, f.max_weight(), msg=msg)

        else:
            with Simplification(False):
                preimage = self._find_preimage(f, beta)

            msg += "\tpreimage found: {}\n".format(preimage)

            self.assertIsNone(preimage, msg=msg)

    @given(
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=1, max_value=2 ** DP_WIDTH - 1),
    )
    def test_case(self, d1, d2, a):
        self._test_case(d1, d2, a)

    @unittest.skip("skipping testing all cases of CteAdd real formula")
    def test_all_cases(self):
        with Simplification(False):
            for a in range(1, 2**DP_WIDTH):
                print("a:", a)
                for d1, d2 in itertools.product(range(2**DP_WIDTH), range(2**DP_WIDTH)):
                    self._test_case(d1, d2, a)


class TestDerivativeBvCteAdd(unittest.TestCase):
    """Test for the derivative of the BvCteAdd."""
    distr_error = None
    max_error = None

    @classmethod
    def setUpClass(cls):
        cls.distr_error = collections.Counter()
        cls.max_error = [-1, None]

    @classmethod
    def tearDownClass(cls):
        if PRINT_DISTR_ERROR:
            # print("counter_weight:", [[round(k, ERROR_DIGITS), v] for k, v in cls.counter_weight.most_common()])
            print("distribution errors:", [(round(e[0], ERROR_DIGITS), e[1]) for e in cls.distr_error.most_common()])
        if PRINT_MAX_ERROR:
            print("max error:", round(cls.max_error[0], ERROR_DIGITS), cls.max_error[1])

    def _test_case(self, d1, d2, a, precision, version):
        d1 = Constant(d1, DP_WIDTH)
        d2 = Constant(d2, DP_WIDTH)
        a = Constant(a, DP_WIDTH)

        diff_type = XorDiff
        der_type = derivative.XDCA
        old_precision = der_type.precision
        der_type.precision = precision

        alpha = diff_type(d1)
        beta = diff_type(d2)
        f = der_type(alpha, a)

        der_repr = "{}_cte={}_prec={}_v={}({} -> {})\n".format(der_type.__name__, a, precision, version, alpha, beta)
        msg = der_repr

        is_valid = f.is_possible(beta)

        msg += "\tis_valid: {}\n".format(is_valid)

        real_weight = f.exact_weight(beta)

        msg += "\texact weight: {}\n".format(real_weight)

        self.assertEqual(is_valid, real_weight != math.inf, msg=msg)

        if is_valid:
            theo_weight = int(f.weight(beta, version=version))  # .doit())

            self.assertLessEqual(theo_weight, f.max_weight(), msg=msg)

            theo_weight /= 2**(f.num_frac_bits())

            msg += "\ttheo weight: {}\n".format(theo_weight)

            real_error = theo_weight - real_weight
            theo_error = f.error()

            if PRINT_MAX_ERROR and real_error > self.__class__.max_error[0]:
                self.__class__.max_error = [real_error, der_repr]

            msg += "\treal_error: {}\n".format(round(real_error, ERROR_DIGITS))
            msg += "\ttheo_error: {}\n".format(round(theo_error, ERROR_DIGITS))

            if PRINT_DISTR_ERROR:
                self.distr_error[round(real_error, ERROR_DIGITS)] += 1

            self.assertGreaterEqual(real_error, 0, msg=msg)
            self.assertLessEqual(real_error, theo_error, msg=msg)

        der_type.precision = old_precision

    @given(
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=1, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=4),
        integers(min_value=0, max_value=2),
    )
    @example(d1=0x04, d2=0x04, a=0x01, k=3, v=2)
    @example(d1=137, d2=137, a=138, k=3, v=2)
    @example(d1=208, d2=208, a=1, k=3, v=2)
    @example(d1=206, d2=206, a=1, k=3, v=2)
    @example(d1=96, d2=96, a=97, k=3, v=2)
    @example(d1=32, d2=32, a=31, k=3, v=2)
    @example(d1=0x08, d2=0x08, a=0x7b, k=3, v=2)
    @example(d1=8, d2=8, a=14, k=3, v=2)
    @example(d1=64, d2=64, a=21, k=3, v=2)
    @settings(deadline=None)  # , max_examples=10000)
    def test_case(self, d1, d2, a, k, v):
        self._test_case(d1, d2, a, k, v)

    @unittest.skip("skipping testing all cases of CteAdd BV-formula")
    def test_all_cases(self):
        precision = 3
        print("precision:", precision)
        with Simplification(False):
            for a in range(1, 2**DP_WIDTH):
                import random
                a = random.randint(1, 2**DP_WIDTH - 1)
                print("a:", a)
                for d1, d2 in itertools.product(range(2**DP_WIDTH), range(2**DP_WIDTH)):
                    self._test_case(d1, d2, a, precision)

    # -----

    def _test_has_probability_one(self, d1, d2, a):
        d1 = Constant(d1, DP_WIDTH)
        d2 = Constant(d2, DP_WIDTH)
        a = Constant(a, DP_WIDTH)

        diff_type = XorDiff
        der_type = derivative.XDCA

        alpha = diff_type(d1)
        beta = diff_type(d2)
        f = der_type(alpha, a)

        msg = "{}_{}_({} -> {})\n".format(der_type.__name__, a, alpha, beta)

        real_weight = f.exact_weight(beta)

        msg += "\texact weight: {}\n".format(real_weight)

        has_probability_one = f.has_probability_one(beta)

        msg += "\thas_probability_one: {}\n".format(has_probability_one)

        self.assertEqual(has_probability_one, real_weight == 0, msg=msg)

    @given(
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=0, max_value=2 ** DP_WIDTH - 1),
        integers(min_value=1, max_value=2 ** DP_WIDTH - 1),
    )
    def test_has_probability_one(self, d1, d2, a):
        self._test_has_probability_one(d1, d2, a)


# noinspection PyUnusedLocal
def load_tests(loader, tests, ignore):
    """Add doctests."""
    tests.addTests(doctest.DocTestSuite(derivative))
    return tests
