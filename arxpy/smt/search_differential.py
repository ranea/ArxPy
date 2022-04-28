"""Search for characteristics by modeling the search as a SMT problem."""
import collections
import enum
import functools
import itertools
import datetime
import math
import pprint
import warnings

from pysmt import environment
from pysmt import logics

from arxpy.bitvector import core
from arxpy.bitvector import context
from arxpy.bitvector import operation
from arxpy.bitvector import extraop
from arxpy.differential import difference
from arxpy.differential import derivative
from arxpy.differential import characteristic
from arxpy.smt import types
from arxpy.smt import verification_differential


# default arguments for checking empirical weights
MIN_PAIRS = 128
MAX_PAIRS = 1024
KEY_SAMPLES = 64
MAX_CH_FOUND = 2**10


def _get_smart_print(filename=None):
    def smart_print(*msg, **kwargs):
        if filename is not None:
            with open(filename, "a") as fh:
                print(*msg, file=fh, flush=True, **kwargs)
        else:
            print(*msg, flush=True, **kwargs)
    return smart_print


def _get_time():
    now = datetime.datetime.now()
    return "{}-{}:{}".format(now.day, now.hour, now.minute)


class DerMode(enum.Enum):
    """Represent the different constraints available for the derivative weights of a characteristic.

    Attributes:
        Default: no additional constraint is added
        ProbabilityOne: fix all derivative weights to zero
        Valid: only the validity constraint is added (weight is ignored)
        XDCA_Approx: for the weights of `XDCA`, the ``precision`` is set to ``0``

    """
    Default = enum.auto()
    ProbabilityOne = enum.auto()
    Valid = enum.auto()
    XDCA_Approx = enum.auto()


class ChSearchMode(enum.Enum):
    """Represent the different options for searching `BvCharacteristic`.

    A characteristic is said to be *valid* if its weight
    :math:`w = - \log_2(p)`, where *p* is the probability of the characteristic,
    is lower than the bit-size of the input difference.

    A derivative weight is said to be *non-exact* if its weight
    does not match its exact weight.

    Attributes:
        FirstCh: returns the 1st characteristic found by iteratively searching with increasing weight.
            The characteristic found is returned as a `ChFound` object, but it may not have
            optimal theoretical weight if it contains a non-exact derivative weight
            or it may be empirically invalid. If no characteristic is found,
            ``None`` is returned.
        FirstChValid: similar to `FirstCh`, but if the characteristic found is
            empirically invalid (after computing its empirical weight), the search continues.
        Optimal: similar to `FirstCh`, but several characteristics are found to ensure
            that the one returned has the optimal theoretical weight.
            (only differs from `FirstCh` if there is a non-exact derivative weight).
        OptimalDifferential: similar to `Optimal`, but once a characteristic is found,
            only characteristics with different input/output differences are searched.
        AllOptimal: similar to `Optimal`, but the search never stops.
        TopDifferentials: searches for `MAX_CH_FOUND` characteristics
            (as in `FirstCh`) and return the differentials obtained
            by gathering those characteristics.
        AllValid: a non-stop search of valid characteristics with no
            additional constraints.

    """
    FirstCh = enum.auto()
    FirstChValid = enum.auto()
    Optimal = enum.auto()
    OptimalDifferential = enum.auto()
    AllOptimal = enum.auto()
    AllValid = enum.auto()
    TopDifferentials = enum.auto()


class SkChSearchMode(enum.Enum):
    """Represent the different options for searching `SingleKeyCh`.

    The options are the same as `ChSearchMode`, but `SkChFound` objects are returned instead of `ChFound`.
    """
    FirstCh = enum.auto()
    FirstChValid = enum.auto()
    Optimal = enum.auto()
    OptimalDifferential = enum.auto()
    AllOptimal = enum.auto()
    AllValid = enum.auto()
    TopDifferentials = enum.auto()


class RkChSearchMode(enum.Enum):
    """Represent the different options for searching `RelatedKeyCh`.

    A related-key characteristic is said to be *valid* if its encryption weight
    is lower than the bit-size of the plaintext difference and
    its key schedule weight is lower than the bit-size of the master-key difference.

    Attributes:
        FirstMinSum: returns the 1st related-key characteristic found
            by iteratively searching with increasing *weight*, where the *weight*
            is taken as the sum of the key schedule weight and the encryption weight.
            The characteristic found is returned as a `RkChFound` object,
            but it may not have optimal theoretical weight if it it contains
            a non-exact derivative weight or it may be empirically invalid.
            If no characteristic is found, ``None`` is returned.
        FirstMinSumValid: similar to `FirstMinSum`, but if the characteristic found is
            empirically invalid (after computing its empirical weight), the search continues.
        OptimalMinSum: similar to `FirstMinSum`, but several characteristics are found to ensure
            that the one returned has the optimal theoretical weight.
            (only differs from `FirstMinSum` if there is a non-exact derivative weight).
        OptimalMinSumDifferential: similar to `OptimalMinSum`, but once a characteristic is found,
            only characteristics with different input/output differences are searched.
            (to ensure optimality).
        FirstValidKeyMinEnc: similar to `FirstMinSum`,
            but the *weight* is taken as the encryption weight and the
            key schedule weight is restricted to be smaller than the key size.
        FirstValidKeyMinEncValid: similar to `FirstMinSumValid`,
            but the *weight* is defined as `FirstValidKeyMinEnc`.
        OptimalValidKeyMinEnc: similar to `OptimalMinSum`,
            but the *weight* is defined as `FirstValidKeyMinEnc`.
        OptimalValidKeyMinEncDifferential: similar to `OptimalMinSumDifferential`,
            but the *weight* is  defined as `FirstValidKeyMinEnc`.
        FirstFixEncMinKey: similar to `FirstMinSum`,
            but the *weight* is taken as the key schedule weight and
            the encryption weight is fixed to a given value.
        FirstFixEncMinKeyValid: similar to `FirstMinSumValid`,
            but the *weight* is defined as `FirstFixEncMinKey`.
        OptimalFixEncMinKey: similar to `OptimalMinSum`,
            but the *weight* is defined as `FirstFixEncMinKey`.
        OptimalFixEncMinKeyDifferential: similar to `OptimalMinSumDifferential`,
            but the *weight* is defined as `FirstFixEncMinKey`.
        AllOptimalMinSum: similar to `OptimalMinSum`, but the search never stops.
        AllValid: a non-stop search of valid characteristics with no
            additional constraints.

    """
    FirstMinSum = enum.auto()
    FirstMinSumValid = enum.auto()
    OptimalMinSum = enum.auto()
    OptimalMinSumDifferential = enum.auto()
    AllOptimalMinSum = enum.auto()
    FirstValidKeyMinEnc = enum.auto()
    FirstValidKeyMinEncValid = enum.auto()
    OptimalValidKeyMinEnc = enum.auto()
    OptimalValidKeyMinEncDifferential = enum.auto()
    FirstFixEncMinKey = enum.auto()
    FirstFixEncMinKeyValid = enum.auto()
    OptimalFixEncMinKey = enum.auto()
    OptimalFixEncMinKeyDifferential = enum.auto()
    AllValid = enum.auto()


class ExactWeightError(Exception):
    """The exception raised in `ChFound.check_empirical_weight` and similar functions."""
    pass


class ChFound(object):
    """Represent (non-symbolic) characteristics found.

    Attributes:
        ch: the associated symbolic characteristic
        ch_weight: a `Constant` denoting the the characteristic weight
        der_weights: a list, where the i-th element is a pair containing
            the i-th derivative symbolic weight and its value
        input_diff: a list, where the i-th element is a pair containing
            the i-th input symbolic difference and its value
        nonlinear_diffs: aa list, where the i-th element is a pair containing
            the i-th non-linear symbolic difference and its value
        output_diff: a list, where the i-th element is a pair containing
            the i-th output symbolic difference and its value

    ::

        >>> from arxpy.bitvector.operation import BvComp
        >>> from arxpy.differential.difference import XorDiff, RXDiff
        >>> from arxpy.differential.characteristic import BvCharacteristic
        >>> from arxpy.primitives.chaskey import ChaskeyPi
        >>> from arxpy.smt.search_differential import SearchCh
        >>> ChaskeyPi.set_rounds(1)
        >>> ch = BvCharacteristic(ChaskeyPi, XorDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> search_problem = SearchCh(ch)
        >>> ch_found = search_problem.solve(0)
        >>> ch_found.ch_weight
        0b0000000
        >>> ch_found.der_weights
        [[w0w1w2, 0b0000000], [w3, 0b00000]]
        >>> ch_found.input_diff  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
        [[XorDiff(dv0), XorDiff(...)], [XorDiff(dv1), XorDiff(...)],
        [XorDiff(dv2), XorDiff(...)], [XorDiff(dv3), XorDiff(...)]]
        >>> ch_found.nonlinear_diffs  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
        [[XorDiff(d0), XorDiff(...)], [XorDiff(d4), XorDiff(...)],
        [XorDiff(d7), XorDiff(...)], [XorDiff(d10), XorDiff(...)]]
        >>> ch_found.output_diff  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
        [[XorDiff(d7), XorDiff(...)], [XorDiff(d12), XorDiff(...)],
        [XorDiff(d13), XorDiff(...)], [XorDiff(d9), XorDiff(...)]]
        >>> print(ch_found)  # doctest: +ELLIPSIS
        {'ch_weight': 0,
         'der_weights': [[w0w1w2, 0], [w3, 0]],
         'exact_weight': 0,
         'input_diff': [[dv0, ...], [dv1, ...], [dv2, ...], [dv3, ...]],
         'nonlinear_diffs': [[d0, ...], [d4, ...], [d7, ...], [d10, ...]],
         'output_diff': [[d7, ...], [d12, ...], [d13, ...], [d9, ...]]}
        >>> ch = BvCharacteristic(ChaskeyPi, RXDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> ic = [BvComp(0, d.val) for d in ch.input_diff]
        >>> ic += [BvComp(0, d[1].val) for d in ch.output_diff]
        >>> search_problem = SearchCh(ch, allow_zero_input_diff=True, initial_constraints=ic)
        >>> ch_found = search_problem.solve(5)
        >>> ch_found.ch_weight
        0x05
        >>> ch_found.der_weights
        [[w0, 0b000001011], [w1, 0b000001011], [w2, 0b000001011], [w3, 0b000001011]]
        >>> ch_found.input_diff  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
        [[RXDiff(dv0), RXDiff(0x00000000)], [RXDiff(dv1), RXDiff(0x00000000)],
        [RXDiff(dv2), RXDiff(0x00000000)], [RXDiff(dv3), RXDiff(0x00000000)]]
        >>> ch_found.nonlinear_diffs  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
        [[RXDiff(d0), RXDiff(0x00000000)], [RXDiff(d4), RXDiff(0x00000000)],
        [RXDiff(d7), RXDiff(0x00000000)], [RXDiff(d10), RXDiff(0x00000000)]]
        >>> ch_found.output_diff  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
        [[RXDiff(d7), RXDiff(0x00000000)], [RXDiff(d12), RXDiff(0x00000000)],
        [RXDiff(d13), RXDiff(0x00000000)], [RXDiff(d9), RXDiff(0x00000000)]]
        >>> print(ch_found)  # doctest: +ELLIPSIS
        {'ch_weight': 5,
         'der_weights': [[w0, 1.375], [w1, 1.375], [w2, 1.375], [w3, 1.375]],
         'exact_weight': 5.66...,
         'input_diff': [[dv0, 0x00000000], [dv1, 0x00000000], [dv2, 0x00000000], [dv3, 0x00000000]],
         'nonlinear_diffs': [[d0, 0x00000000], [d4, 0x00000000], [d7, 0x00000000], [d10, 0x00000000]],
         'output_diff': [[d7, 0x00000000], [d12, 0x00000000], [d13, 0x00000000], [d9, 0x00000000]]}

    """
    def __init__(self, ch, ch_weight, der_weights, model, aux_model=None):
        self.ch = ch

        assert not isinstance(ch_weight, core.Constant)
        self.ch_weight = model[ch_weight]

        if aux_model is None:
            aux_model = {}
        model = {**model, **aux_model}
        diff_model = {d: model[d] for d in model if isinstance(d, difference.Difference)}

        self.free_diffs = []
        for d in itertools.chain(self.ch.input_diff, self.ch.nonlinear_diffs):
            if d not in diff_model:
                self.free_diffs.append(d)
                diff_model[d] = type(d)(core.Constant(0, d.val.width))

        self.input_diff = [[d, diff_model[d]] for d in self.ch.input_diff]
        self.output_diff = [[v, d.xreplace(diff_model)] for v, d in self.ch.output_diff]
        self.nonlinear_diffs = [[d, diff_model[d]] for d in self.ch.nonlinear_diffs]
        self.der_weights = [[w, model[w]] for w in der_weights if "tmp" not in w.name]

        if hasattr(self.ch.func, "round_inputs"):
            self.round_inputs = []
            for round_input in self.ch.func.round_inputs:
                self.round_inputs.append([self.ch._var2diff[d].xreplace(diff_model) for d in round_input])

        self.emp_weight = None
        self._diff_model = diff_model
        self._exact_weight = None

    def get_exact_weight(self):
        """Return the exact weight of the characteristic found."""
        if self._exact_weight is not None:
            return self._exact_weight

        exact_weight = 0
        for i, (diff, der) in enumerate(self.ch.nonlinear_diffs.items()):
            # diff = diff.xreplace(self._diff_model)
            diff = self.nonlinear_diffs[i][1]
            new_input_diff = [(d.xreplace(self._diff_model)) for d in der.input_diff]
            exact_der_weight = der._replace_input_diff(new_input_diff).exact_weight(diff)
            assert not isinstance(exact_der_weight, core.Term)
            exact_weight += exact_der_weight

        # if self.ch.diff_type == difference.XorDiff and int(exact_weight) > int(self.ch_weight):
        #     raise ValueError("exact weight {} should be lower than ch.weight {}".format(exact_weight, self.ch_weight))
        # elif self.ch.diff_type == difference.RXDiff and int(exact_weight) < int(self.ch_weight):
        #     raise ValueError("exact weight {} should be higher than ch.weight {}".format(exact_weight, self.ch_weight))

        self._exact_weight = exact_weight

        return exact_weight

    def signature(self, ch_signature_type, return_str=False):
        symbolic_sig = self.ch.signature(ch_signature_type)
        sig = [self.ch.diff_type(d).xreplace(self._diff_model).val for d in symbolic_sig]
        if return_str:
            return ''.join([str(s) for s in sig])
        else:
            return sig

    def check_empirical_weight(self, verbose_lvl=0, filename=None):
        """Check the exact weight of the model matches its empirical weight.

        If the exact weight doesn't match, the exception `ExactWeightError` is raised.

        If ``filename`` is not ``None``, the output will be printed
        to the given file rather than the to stdout.

        The argument ``verbose_lvl`` can also take the values ``1`` and ``2`` for a
        more detailed output.
        """
        smart_print = _get_smart_print(filename)

        input_diff = [value for var, value in self.input_diff]
        output_diff = [value for var, value in self.output_diff]

        exact_weight = self.get_exact_weight()

        current_max_pairs = max(
            int(MAX_PAIRS / (len(self.ch.ssa["assignments"]) + 1)),  # + 1 to avoid division by 0
            MIN_PAIRS + 1
        )

        pair_samples = max(5 * 2**(math.ceil(exact_weight + 1)), MIN_PAIRS)

        if hasattr(self.ch, "_cipher"):
            weak_check = False if not hasattr(self.ch._cipher, "weak_check") else self.ch._cipher.weak_check
        else:
            weak_check = False if not hasattr(self.ch.func, "weak_check") else self.ch.func.weak_check

        if pair_samples > current_max_pairs and not weak_check:
            assert len(self.ch.nonlinear_diffs) > 0
            emp_weight = verification_differential.fast_empirical_weight(self, verbose_lvl=verbose_lvl, filename=filename)
        elif not weak_check or (weak_check and exact_weight == 0):
            if verbose_lvl >= 2:
                smart_print("- checking {} -> {} with {} pair samples".format(
                    '|'.join([str(d.val) for d in input_diff]),
                    '|'.join([str(d.val) for d in output_diff]),
                    pair_samples))
            emp_weight = self.ch.empirical_weight(input_diff, output_diff, pair_samples)
            if verbose_lvl >= 2:
                smart_print("- exact/empirical weight: {}, {}".format(exact_weight, emp_weight))
        else:
            return

        error_weight = sum(self.ch.func.input_widths) * 0.03125  # every 32-bit, 1 unit of error

        # if not (emp_weight - error_weight <= exact_weight <= emp_weight + error_weight):
        if emp_weight is math.inf or not(exact_weight <= emp_weight + error_weight):
            msg = "The exact and empirical weights do not match\n"
            msg += " - exact weight:     {}\n".format(exact_weight)
            msg += " - empirical weight: {}\n".format(emp_weight)
            msg += str(self._to_dict(vrepr=verbose_lvl >= 3))
            raise ExactWeightError(msg)
        else:
            self.emp_weight = emp_weight

    def _check_empirical_distribution_weight(self, cipher, verbose_lvl=0, filename=None, rk_dict_diffs=None):
        # similar to _empirical_distribution_weight of characteristic module
        smart_print = _get_smart_print(filename)

        input_diff = [value for var, value in self.input_diff]
        output_diff = [value for var, value in self.output_diff]

        exact_weight = self.get_exact_weight()

        current_max_pairs = max(
            int(MAX_PAIRS / (len(self.ch.ssa["assignments"]) + 1)),  # + 1 to avoid division by 0
            MIN_PAIRS + 1
        )

        key_samples = KEY_SAMPLES
        pair_samples = max(4 * 2 ** math.ceil(exact_weight + 1), MIN_PAIRS)

        weak_check = False if not hasattr(cipher, "weak_check") else cipher.weak_check

        if key_samples * pair_samples > current_max_pairs and not weak_check:
            assert len(self.ch.nonlinear_diffs) != 0
            emp_weight_dist = verification_differential._fast_empirical_weight_distribution(self, cipher, rk_dict_diffs,
                                                                               verbose_lvl=verbose_lvl, filename=filename)
        elif not weak_check or (weak_check and exact_weight == 0):
            if verbose_lvl >= 2:
                smart_print("- checking {} -> {} with {} pair samples and {} key samples".format(
                    '|'.join([str(d.val) for d in input_diff]),
                    '|'.join([str(d.val) for d in output_diff]),
                    pair_samples, key_samples))
            if rk_dict_diffs is None:
                rk_output_diff = None
            else:
                assert len(rk_dict_diffs) > 0
                rk_output_diff = [value for var, value in rk_dict_diffs["output_diff"]]
            emp_weight_dist = self.ch._empirical_weight_distribution(cipher, input_diff, output_diff, pair_samples,
                                                                     key_samples, rk_output_diff=rk_output_diff)
            if verbose_lvl >= 2:
                smart_print("- exact/empirical weight: {}, {}".format(exact_weight, emp_weight_dist))
        else:
            return

        error_weight = sum(self.ch.func.input_widths) * 0.03125

        max_emp_weight_dist = max([ew for ew in emp_weight_dist if ew != math.inf], default=-math.inf)
        # if not(min(emp_weight_dist) - error_weight <= exact_weight <= max_emp_weight_dist + error_weight):
        if max_emp_weight_dist in [-math.inf, math.inf] or not(exact_weight <= max_emp_weight_dist + error_weight):
            msg = "The exact and empirical weights do not match\n"
            msg += " - exact weight:      {}\n".format(round(exact_weight, 1))
            msg += " - empirical weights: {}\n".format(emp_weight_dist)
            msg += str(self._to_dict(vrepr=verbose_lvl >= 3))
            raise ExactWeightError(msg)
        else:
            self.emp_weight = emp_weight_dist

    def _to_dict(self, vrepr=False):
        class DictItem(object):
            verbose_repr = vrepr

            def __init__(self, item, str_item=None):
                self.item = item
                self.str_item = str_item

            def __str__(self):
                vr = self.__class__.verbose_repr

                if not vr and self.str_item is not None:
                    return str(self.str_item)
                elif isinstance(self.item, (int, float, collections.Counter)):
                    return str(self.item)
                elif isinstance(self.item, core.Term):
                    if not vr:
                        return str(self.item)
                    else:
                        return self.item.vrepr()
                elif isinstance(self.item, difference.Difference):
                    if not vr:
                        return str(self.item.val)
                    else:
                        return self.item.vrepr()
                elif isinstance(self.item, list):
                    new_args = []
                    for a in self.item:
                        new_args.append(str(DictItem(a)))
                    return "[{}]".format(', '.join(new_args))
                else:
                    raise ValueError("invalid argument {}".format(self.item))

            __repr__ = __str__

        ch_weight = DictItem(self.ch_weight, str_item=int(self.ch_weight))
        exact_weight = DictItem(self.get_exact_weight())
        input_diff = DictItem(self.input_diff)
        output_diff = DictItem(self.output_diff)
        nonlinear_diffs = DictItem(self.nonlinear_diffs)

        der_weights_str = self.der_weights[:]
        max_num_frac_bits = max((der.num_frac_bits() for der in self.ch.nonlinear_diffs.values()), default=0)
        for i in range(len(der_weights_str)):
            diff_val = int(der_weights_str[i][1])
            if max_num_frac_bits > 0:
                diff_val = diff_val * 1.0 / 2**(max_num_frac_bits)
            der_weights_str[i] = [der_weights_str[i][0], diff_val]
        der_weights = DictItem(self.der_weights, str_item=der_weights_str)

        dict_ch = {
            'ch_weight': ch_weight,
            'exact_weight': exact_weight,
            'input_diff': input_diff,
            'output_diff': output_diff,
            'nonlinear_diffs': nonlinear_diffs,
            'der_weights': der_weights,
        }
        if self.emp_weight is not None:
            dict_ch['emp_weight'] = DictItem(self.emp_weight)
        if len(self.free_diffs) > 0:
            dict_ch['free_diffs'] = DictItem(self.free_diffs)
        if hasattr(self.ch.func, "round_inputs"):
            dict_ch['round_inputs'] = DictItem(self.round_inputs)

        return dict_ch

    def __str__(self):
        return pprint.pformat(self._to_dict(), width=100, compact=True)

    def vrepr(self):
        """Return a verbose dictionary-like representation of the characteristic.

            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.characteristic import BvCharacteristic
            >>> from arxpy.primitives.chaskey import ChaskeyPi
            >>> from arxpy.smt.search_differential import SearchCh
            >>> ChaskeyPi.set_rounds(1)
            >>> ch = BvCharacteristic(ChaskeyPi, XorDiff, ["dv0", "dv1", "dv2", "dv3"])
            >>> search_problem = SearchCh(ch)
            >>> ch_found = search_problem.solve(0)
            >>> print(ch_found.vrepr())  # doctest:+ELLIPSIS,+NORMALIZE_WHITESPACE
            {'ch_weight': Constant(0b0000000, width=7),
            'exact_weight': 0,
            'input_diff': [[XorDiff(Variable('dv0', width=32)), XorDiff(Constant(..., width=32))],
                [XorDiff(Variable('dv1', width=32)), XorDiff(Constant(..., width=32))],
                [XorDiff(Variable('dv2', width=32)), XorDiff(Constant(..., width=32))],
                [XorDiff(Variable('dv3', width=32)), XorDiff(Constant(..., width=32))]],
            'output_diff': [[XorDiff(Variable('d7', width=32)), XorDiff(Constant(..., width=32))],
                [XorDiff(Variable('d12', width=32)), XorDiff(Constant(..., width=32))],
                [XorDiff(Variable('d13', width=32)), XorDiff(Constant(..., width=32))],
                [XorDiff(Variable('d9', width=32)), XorDiff(Constant(..., width=32))]],
            'nonlinear_diffs': [[XorDiff(Variable('d0', width=32)), XorDiff(Constant(..., width=32))],
                [XorDiff(Variable('d4', width=32)), XorDiff(Constant(..., width=32))],
                [XorDiff(Variable('d7', width=32)), XorDiff(Constant(..., width=32))],
                [XorDiff(Variable('d10', width=32)), XorDiff(Constant(..., width=32))]],
            'der_weights': [[Variable('w0w1w2', width=7), Constant(0b0000000, width=7)],
                [Variable('w3', width=5), Constant(0b00000, width=5)]]}

        """
        return str(self._to_dict(vrepr=True))

    def srepr(self):
        """Return a short representation of the characteristic."""
        assert len(self.free_diffs) == 0
        input_diff = ' '.join([x.val.hex()[2:] if x.val.width >= 8 else x.val.bin()[2:] for _, x in self.input_diff])
        output_diff = ' '.join([x.val.hex()[2:] if x.val.width >= 8 else x.val.bin()[2:] for _, x in self.output_diff])
        return "(weight {}) {} -> {}".format(int(self.ch_weight), input_diff, output_diff)


class SkChFound(ChFound):
    """Represent (non-symbolic) single-key characteristics found.

    See also `ChFound`.

    ::

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import SingleKeyCh
        >>> from arxpy.smt.search_differential import SearchSkCh
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_rounds(1)
        >>> ch = SingleKeyCh(Speck32, XorDiff)
        >>> search_problem = SearchSkCh(ch)
        >>> ch_found = search_problem.solve(0)
        >>> ch_found.ch_weight
        0x0
        >>> ch_found.der_weights
        [[w0, 0x0]]
        >>> ch_found.input_diff  # doctest:+ELLIPSIS
        [[XorDiff(dp0), XorDiff(...)], [XorDiff(dp1), XorDiff(...)]]
        >>> ch_found.nonlinear_diffs  # doctest:+ELLIPSIS
        [[XorDiff(dx1), XorDiff(...)]]
        >>> ch_found.output_diff  # doctest:+ELLIPSIS
        [[XorDiff(dx2), XorDiff(...)], [XorDiff(dx4), XorDiff(...)]]
        >>> print(ch_found)  # doctest:+ELLIPSIS
        {'ch_weight': 0,
         'der_weights': [[w0, 0]],
         'exact_weight': 0,
         'input_diff': [[dp0, ...], [dp1, ...]],
         'nonlinear_diffs': [[dx1, ...]],
         'output_diff': [[dx2, ...], [dx4, ...]]}

    """

    def __init__(self, ch, ch_weight, der_weights, model):
        super().__init__(ch, ch_weight, der_weights, model)

        self._cipher = ch._cipher

    def check_empirical_weight(self, verbose_lvl=0, filename=None):
        """Check the exact weight of the model matches one weight of the empirical weight distribution.

        If the exact weight doesn't match, the exception `ExactWeightError` is raised.

        If ``filename`` is not ``None``, the output will be printed
        to the given file rather than the to stdout.

        The argument ``verbose_lvl`` can also take the values ``1`` and ``2`` for a
        more detailed output.
        """
        return self._check_empirical_distribution_weight(self._cipher, verbose_lvl, filename, rk_dict_diffs=None)


class RkChFound(object):
    """Represent (non-symbolic) related-key characteristics found.

    Attributes:
        rkch: the associated symbolic `RelatedKeyCh`
        key_ch_found: a `ChFound` denoting the key schedule characteristic found
        enc_ch_found: a `ChFound` denoting the encryption characteristic found

    ::

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import RelatedKeyCh
        >>> from arxpy.smt.search_differential import SearchRkCh
        >>> from arxpy.primitives.lea import LeaCipher
        >>> LeaCipher.set_rounds(1)
        >>> rkch = RelatedKeyCh(LeaCipher, XorDiff)
        >>> search_problem = SearchRkCh(rkch)
        >>> rkch_found = search_problem.solve(0, 0, solver_name="btor")
        >>> rkch_found.key_ch_found.ch_weight
        0b0000000
        >>> rkch_found.key_ch_found.input_diff  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
        [[XorDiff(dmk0), XorDiff(...)], [XorDiff(dmk1), XorDiff(...)],
        [XorDiff(dmk2), XorDiff(...)], [XorDiff(dmk3), XorDiff(...)]]
        >>> rkch_found.key_ch_found.output_diff  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
        [[XorDiff(dk1), XorDiff(...)], [XorDiff(dk3), XorDiff(...)],
        [XorDiff(dk5), XorDiff(...)], [XorDiff(dk3), XorDiff(...)],
        [XorDiff(dk7), XorDiff(...)], [XorDiff(dk3), XorDiff(...)]]
        >>> rkch_found.enc_ch_found.ch_weight
        0b0000000
        >>> rkch_found.enc_ch_found.input_diff  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
        [[XorDiff(dp0), XorDiff(...)], [XorDiff(dp1), XorDiff(...)],
        [XorDiff(dp2), XorDiff(...)], [XorDiff(dp3), XorDiff(...)]]
        >>> rkch_found.enc_ch_found.output_diff  # doctest: +ELLIPSIS,+NORMALIZE_WHITESPACE
        [[XorDiff(dx3), XorDiff(...)], [XorDiff(dx7), XorDiff(...)],
        [XorDiff(dx11), XorDiff(...)], [XorDiff(dp0), XorDiff(...)]]
        >>> print(rkch_found)  # doctest: +ELLIPSIS
        {'enc_ch_found': {'ch_weight': 0,
                          'der_weights': [[w0w1w2, 0]],
                          'exact_weight': 0,
                          'input_diff': [[dp0, ...], [dp1, ...], [dp2, ...], [dp3, ...]],
                          'nonlinear_diffs': [[dx2, ...], [dx6, ...], [dx10, ...]],
                          'output_diff': [[dx3, ...], [dx7, ...], [dx11, ...], [dp0, ...]]},
         'key_ch_found': {'ch_weight': 0,
                          'der_weights': [[wk0, ...], [wk1, ...], [wk2, ...], [wk3, ...]],
                          'exact_weight': ...,
                          'input_diff': [[dmk0, ...], [dmk1, ...], [dmk2, ...], [dmk3, ...]],
                          'nonlinear_diffs': [[dk0, ...], [dk2, ...], [dk4, ...], [dk6, ...]],
                          'output_diff': [[dk1, ...], [dk3, ...], [dk5, ...], [dk3, ...], [dk7, ...], [dk3, ...]]}}

    """
    def __init__(self, rkch, key_schedule_problem, encryption_problem, model):
        self.rkch = rkch
        kp = key_schedule_problem
        key_model = {var: model[var] for var in kp.der_weights + [kp.ch_weight]}  # always present
        for var in itertools.chain(kp.ch.input_diff, kp.ch.nonlinear_diffs.keys()):
            if var in model:
                key_model[var] = model[var]
        self.key_ch_found = ChFound(kp.ch, kp.ch_weight, kp.der_weights, key_model)

        ep = encryption_problem
        enc_model = {var: model[var] for var in ep.der_weights + [ep.ch_weight]}  # always present
        for var in itertools.chain(ep.ch.input_diff, ep.ch.nonlinear_diffs.keys()):
            if var in model:
                enc_model[var] = model[var]
        self.enc_ch_found = ChFound(ep.ch, ep.ch_weight, ep.der_weights, enc_model,
                                    aux_model=dict(self.key_ch_found._diff_model))

        self._cipher = rkch._cipher

    def signature(self, ch_signature_type, return_str=False):
        return self.key_ch_found.signature(ch_signature_type, return_str) + \
               self.enc_ch_found.signature(ch_signature_type, return_str)

    def check_empirical_weight(self, verbose_lvl=0, filename=None):
        """Check the exact weight of the model match its empirical weight.

        Check both the exact weight of the key schedule characteristic
        and the encryption characteristic found match their
        empirical weights.

        See also `ChFound.check_empirical_weight` and `SkChFound.check_empirical_weight`.
        """
        smart_print = _get_smart_print(filename)
        try:
            output_diff = []
            for i, (diff_name, diff_expr) in enumerate(self.rkch.key_schedule_ch.output_diff):
                output_diff.append([diff_expr, self.key_ch_found.output_diff[i][1]] )

            if verbose_lvl >= 2:
                smart_print("Checking the weight of the key schedule characteristic")
            self.key_ch_found.check_empirical_weight(verbose_lvl, filename)
            rk_dict_diffs = {
                "nonlinear_diffs": self.key_ch_found.nonlinear_diffs,
                "output_diff": output_diff
                # "output_diff": self.key_ch_found.output_diff
            }
            if verbose_lvl >= 2:
                smart_print("Checking the weight of the encryption characteristic")
            self.enc_ch_found._check_empirical_distribution_weight(self._cipher, verbose_lvl, filename, rk_dict_diffs=rk_dict_diffs)
        except ExactWeightError as e:
            raise e

    def _to_dict(self, vrepr=False):
        dict_ch = {
            "key_ch_found": self.key_ch_found._to_dict(vrepr=vrepr),
            "enc_ch_found": self.enc_ch_found._to_dict(vrepr=vrepr),
        }
        return dict_ch

    def __str__(self):
        return pprint.pformat(self._to_dict(), width=100, compact=True)

    def vrepr(self):
        """Return a verbose dictionary-like representation of the characteristic.

        See also `ChFound.vrepr`.
        """
        return str(self._to_dict(vrepr=True))

    def srepr(self):
        """Return a short representation of the characteristic."""
        return "K: {} | E: {}".format(self.key_ch_found.srepr(), self.enc_ch_found.srepr())


class SearchCh(object):
    """Represent the SMT problem of finding a `BvCharacteristic` with conditions on the weight.

    Args:
        ch (BvCharacteristic): a symbolic characteristic of a `BvFunction`
        der_mode (DerMode): one of the modes available for the derivative weights
        weight_prefix (str): the prefix to label weight variables
        allow_zero_input_diff(bool): if ``True``, allow the input difference to be zero
        initial_constraints (list): a list of constraints (given as `Term`) to add to the SMT problem
        env (pysmt.environment.Environment): if ``None``, a new pySMT environment is created

    ::

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import BvCharacteristic
        >>> from arxpy.primitives.chaskey import ChaskeyPi
        >>> ChaskeyPi.set_rounds(1)
        >>> ch = BvCharacteristic(ChaskeyPi, XorDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> search_problem = SearchCh(ch)
        >>> search_problem.formula_size()
        2804
        >>> search_problem.error()
        0
        >>> print(search_problem.hrepr(False))
        assert ~(0x00000000000000000000000000000000 == (dv0 :: dv1 :: dv2 :: dv3))
        assert 0x00000000 == ((~(... << ...) ^ (d0 << 0x00000001)) & (~(... << ...) ^ (dv1 << 0x00000001)) & ((dv0 << 0x00000001) ^ d0 ^ dv0 ^ dv1))
        assert 0x00000000 == ((~(... << ...) ^ (d4 << 0x00000001)) & (~(... << ...) ^ (dv3 << 0x00000001)) & ((dv2 << 0x00000001) ^ d4 ^ dv2 ^ dv3))
        assert 0x00000000 == ((~(... << ...) ^ (d7 << 0x00000001)) & (~(... << ...) ^ ((... ^ ...) << 0x00000001)) & (((d0 <<< 16) << 0x00000001) ^ d7 ^ ... ^ ... ^ (... <<< ...)))
        assert 0x00000000 == ((~(... << ...) ^ (d10 << 0x00000001)) & (~(... << ...) ^ (d4 << 0x00000001)) & (((d0 ^ (... <<< ...)) << 0x00000001) ^ d10 ^ d4 ^ ... ^ ...))
        assert w0w1w2 == (PopCountSum3(~((d0 ^ ~...) & (dv1 ^ ~...))[30:], ~((d4 ^ ~...) & (dv3 ^ ~...))[30:], ~((d7 ^ ~...) & (~... ^ ... ^ ...))[30:]))
        assert w3 == PopCount(~((d10 ^ ~...) & (d4 ^ ~...))[30:])
        assert w == (w0w1w2 + (0b00 :: w3))
        minimize w
        >>> search_problem.solve(0).ch_weight
        0b0000000

    """

    def __init__(self, ch, der_mode=DerMode.Default, weight_prefix="w",
                 allow_zero_input_diff=False, initial_constraints=None, env=None,):
        assert isinstance(ch, characteristic.BvCharacteristic)
        assert isinstance(der_mode, DerMode)

        if der_mode == DerMode.XDCA_Approx:
            class XDCA_Approx(derivative.XDCA):
                precision = 0

            for diff in ch.nonlinear_diffs:
                der = ch.nonlinear_diffs[diff]
                if isinstance(der, derivative.XDCA):
                    der = XDCA_Approx(der.input_diff, der.op.constant)
                    ch.nonlinear_diffs[diff] = der
                    assert der.precision == 0

        if initial_constraints is None:
            constraints = []
        else:
            constraints = initial_constraints[:]

        if not allow_zero_input_diff:
            input_diff = functools.reduce(operation.Concat, [d.val for d in ch.input_diff])
            constraints.append(operation.BvNot(operation.BvComp(input_diff, core.Constant(0, input_diff.width))))

        self.ch = ch
        self.constraints = constraints
        self.weight_prefix = weight_prefix
        self.der_mode = der_mode

        with context.NotEvaluation([extraop.Reverse, extraop.LeadingZeros,
                                    extraop.PopCount, extraop.PopCountDiff, extraop.PopCountSum2, extraop.PopCountSum3]):
            self._generate_constraints()

        if env is None:
            self._env = environment.reset_env()

    def _generate_constraints(self):
        """Generate the bit-vector constraints of the SMT problem."""
        if not self.ch.nonlinear_diffs:
            constraints = []
            ch_weight = core.Variable(self.weight_prefix, 1)
            constraints.append(operation.BvComp(ch_weight, core.Constant(0, ch_weight.width)))
            self.constraints.extend(constraints)
            self.ch_weight = ch_weight
            self.der_weights = []
            self.ch_max_weight = 0
            return

        if self.der_mode == DerMode.ProbabilityOne:
            constraints = []
            for (diff, der) in self.ch.nonlinear_diffs.items():
                constraints.append(der.has_probability_one(diff))

            ch_weight = core.Variable(self.weight_prefix, 1)
            constraints.append(operation.BvComp(ch_weight, core.Constant(0, ch_weight.width)))

            self.constraints.extend(constraints)
            self.ch_weight = ch_weight
            self.der_weights = []
            self.ch_max_weight = 0
            return

        if self.der_mode == DerMode.Valid:
            constraints = []
            for (diff, der) in self.ch.nonlinear_diffs.items():
                constraints.append(der.is_possible(diff))

            ch_weight = core.Variable(self.weight_prefix, 1)
            constraints.append(operation.BvComp(ch_weight, core.Constant(0, ch_weight.width)))

            self.constraints.extend(constraints)
            self.ch_weight = ch_weight
            self.der_weights = []
            self.ch_max_weight = 0
            return

        max_num_frac_bits = max(der.num_frac_bits() for der in self.ch.nonlinear_diffs.values())
        ch_max_weight = 0
        for der in self.ch.nonlinear_diffs.values():
            ch_max_weight += der.max_weight() << (max_num_frac_bits - der.num_frac_bits())
        ch_max_weight = ch_max_weight >> max_num_frac_bits   # ch_weight ignore fraction bits
        ch_width = max(ch_max_weight.bit_length(), 1)
        for diff, der in self.ch.nonlinear_diffs.items():
            if isinstance(der, derivative.XDCA):
                # the width of XDAC.weight is "greater" than its max_weight
                # (the MSB bits of XDAC.weight are not used)
                int_width = der.weight(diff)[0].width - der.num_frac_bits()
            else:
                int_width = der.weight(diff).width - der.num_frac_bits()
            ch_width = max(ch_width, int_width)

        constraints = []
        # fb2width2hw is a dictionary mapping frac_bits -> input_width -> (weight_var, PopCount)
        # note only PopCounts with the same number of frac_bits and similar args are grouped together
        fb2width2hw = {}
        original_der_weights = []
        der_weights = []

        def zero_ext_right(var, num_zeros):
            """Expand with zeros to the right."""
            if num_zeros == 0:
                return var
            else:
                return operation.Concat(var, core.Constant(0, num_zeros))

        for i, (diff, der) in enumerate(self.ch.nonlinear_diffs.items()):
            constraints.append(der.is_possible(diff))

            if isinstance(der, derivative.XDCA):
                prefix = self.weight_prefix + str(i) + "_tmp"
                # with context.NotEvaluation(extraop.PopCount):
                assert extraop.PopCount in context.NotEvaluation.current_context
                weight_value = der.weight(diff, prefix=prefix)
                if isinstance(weight_value, collections.abc.Sequence):
                    assert len(weight_value) == 2
                    weight_value, assertions = weight_value
                    weight_var = core.Variable(self.weight_prefix + str(i), weight_value.width)
                    constraints.extend(assertions)
            else:
                # with context.NotEvaluation(extraop.PopCount):
                assert extraop.PopCount in context.NotEvaluation.current_context
                weight_value = der.weight(diff)
                weight_var = core.Variable(self.weight_prefix + str(i), weight_value.width)

            if weight_value == 0:
                continue

            num_fb = der.num_frac_bits()
            if isinstance(weight_value, extraop.PopCount):
                if num_fb not in fb2width2hw:
                    fb2width2hw[num_fb] = {}
                input_hw_width = weight_value.args[0].width
                if input_hw_width not in fb2width2hw[num_fb]:
                    fb2width2hw[num_fb][input_hw_width] = []
                fb2width2hw[num_fb][input_hw_width] += [[weight_var, weight_value]]
            else:
                constraints.append(operation.BvComp(weight_var, weight_value))
                original_der_weights.append(weight_var)
                assert ch_width >= weight_var.width - der.num_frac_bits()  # der_int_width
                weight_var = operation.ZeroExtend(weight_var, ch_width - (weight_var.width - der.num_frac_bits()))
                weight_var = zero_ext_right(weight_var, max_num_frac_bits - der.num_frac_bits())
                der_weights.append(weight_var)

        def grouped(iterable, n):
            """Group iterable in packs of n elements.

            If len(iterable) % n != 0, the last elements are not included.
            Example: grouped([s0,s1,s2,s3,s4], 2) -> ((s0, s1), (s2, s3))
            """
            return zip(*[iter(iterable)] * n)

        for num_fb in fb2width2hw:
            for weightvar_hwvalue_pairs in fb2width2hw[num_fb].values():
                for ((w1, p1), (w2, p2), (w3, p3)) in grouped(weightvar_hwvalue_pairs, 3):
                    assert p1.width == p2.width == p3.width
                    weight_value = extraop.PopCountSum3(p1.args[0], p2.args[0], p3.args[0])
                    weight_var = core.Variable(w1.name + w2.name + w3.name, weight_value.width)
                    constraints.append(operation.BvComp(weight_var, weight_value))
                    original_der_weights.append(weight_var)
                    assert ch_width >= weight_var.width - num_fb
                    weight_var = operation.ZeroExtend(weight_var, ch_width - (weight_var.width - num_fb))
                    weight_var = zero_ext_right(weight_var, max_num_frac_bits - num_fb)
                    der_weights.append(weight_var)
                if len(weightvar_hwvalue_pairs) % 3 == 2:
                    w1, p1 = weightvar_hwvalue_pairs[-2]
                    w2, p2 = weightvar_hwvalue_pairs[-1]
                    assert p1.width == p2.width
                    weight_value = extraop.PopCountSum2(p1.args[0], p2.args[0])
                    weight_var = core.Variable(w1.name + w2.name, weight_value.width)
                    constraints.append(operation.BvComp(weight_var, weight_value))
                    original_der_weights.append(weight_var)
                    assert ch_width >= weight_var.width - num_fb
                    weight_var = operation.ZeroExtend(weight_var, ch_width - (weight_var.width - num_fb))
                    weight_var = zero_ext_right(weight_var, max_num_frac_bits - num_fb)
                    der_weights.append(weight_var)
                elif len(weightvar_hwvalue_pairs) % 3 == 1:
                    w1, p1 = weightvar_hwvalue_pairs[-1]
                    weight_var = w1
                    weight_value = p1
                    constraints.append(operation.BvComp(weight_var, weight_value))
                    original_der_weights.append(weight_var)
                    assert ch_width >= weight_var.width - num_fb
                    weight_var = operation.ZeroExtend(weight_var, ch_width - (weight_var.width - num_fb))
                    weight_var = zero_ext_right(weight_var, max_num_frac_bits - num_fb)
                    der_weights.append(weight_var)

        # value of the characteristic weight
        sum_w = sum(der_weights)[:max_num_frac_bits]
        ch_weight = core.Variable(self.weight_prefix, sum_w.width)
        constraints.append(operation.BvComp(ch_weight, sum_w))

        self.constraints.extend(constraints)
        self.ch_weight = ch_weight
        self.ch_max_weight = ch_max_weight
        self.der_weights = original_der_weights

    def solve(self, initial_weight, solver_name="btor", search_mode=ChSearchMode.Optimal, check=False,
              return_generator=False, verbose_level=0, filename=None):
        """Solve the SMT problem associated to the search of a valid `BvCharacteristic`.

        Args:
             initial_weight(int): the initial weight for starting the iterative search
             solver_name(str): the name of the solver (according to pySMT) to be used
             search_mode(ChSearchMode): one of the search modes available
             check(bool): if ``True``, `ChFound.check_empirical_weight` will be called
                after a characteristic is found. If it is not valid, the search will continue.
             return_generator(bool): if ``True``, return a Python generator with the
                characteristics found (only valid in AllOptimal)
             verbose_level(int): an integer between ``0`` (no verbose) and ``3`` (full verbose).
             filename(str): if not ``None``, the output will be  printed to the given file
                rather than the to stdout.

        """
        assert initial_weight < sum(self.ch.func.input_widths) + math.floor(self.error())
        assert type(self) == SearchCh
        assert isinstance(search_mode, ChSearchMode)

        # don't merge incremental_solve() in solve() to allow different
        # ways of solving in the future
        return self._incremental_solve(initial_weight, solver_name, search_mode, check, return_generator,
                                       verbose_level, filename)

    def _incremental_solve(self, initial_weight, solver_name, search_mode, check, return_generator,
                           verbose_level, filename):
        strict_shift = True if solver_name == "btor" else False  # e.g., btor_rol: width must be a power of 2
        bv2pysmt = functools.partial(types.bv2pysmt, env=self._env, strict_shift=strict_shift)

        smart_print = _get_smart_print(filename)

        differences_in_model = []
        for d in itertools.chain(self.ch.input_diff, self.ch.nonlinear_diffs.keys()):
            differences_in_model.append(d)

        if type(self) == SearchCh:
            search_mode_class = ChSearchMode
            ch_found_class = ChFound
        elif type(self) == SearchSkCh:
            search_mode_class = SkChSearchMode
            ch_found_class = SkChFound
        else:
            raise ValueError("invalid subclass of SearchCh")

        if return_generator and not search_mode_class.AllOptimal:
            raise ValueError("return_generator can only be enabled with the search mode AllOptimal")

        # with self._env.factory.Solver(name=solver_name, logic=logics.QF_BV) as solver:
        solver = self._env.factory.Solver(name=solver_name, logic=logics.QF_BV)
        for c in self.constraints:
            solver.add_assertion(bv2pysmt(c, boolean=True))

        max_error = self.error()
        orig_upper_bound = sum(self.ch.func.input_widths) + math.ceil(max_error)  # strict upper bound
        upper_bound = min(orig_upper_bound, self.ch_max_weight + 1)

        if verbose_level >= 1:
            smart_print("Upper bound: {}, max error: {}\n".format(upper_bound, max_error))

        if search_mode == search_mode_class.FirstCh:
            target_weight = initial_weight

            while target_weight < upper_bound:
                eq_weight = operation.BvComp(self.ch_weight, core.Constant(target_weight, self.ch_weight.width))
                if verbose_level >= 1:
                    smart_print(_get_time(), "| Solving", eq_weight)
                satisfiable = solver.solve([bv2pysmt(eq_weight, boolean=True)])

                if satisfiable:
                    model = types.pysmt_model2bv_model(solver.get_model(), differences_in_model)
                    assert model[self.ch_weight] == target_weight

                    ch_found = ch_found_class(self.ch, self.ch_weight, self.der_weights, model)

                    if check:
                        try:
                            ch_found.check_empirical_weight(verbose_level, filename)
                        except ExactWeightError:
                            ch_found.emp_weight = math.inf

                    solver.exit()
                    return ch_found
                else:
                    target_weight += 1
            else:
                if verbose_level >= 1:
                    smart_print(_get_time(), "| Unsatisfiable")

                solver.exit()
                return None

        if search_mode == search_mode_class.AllValid:
            last_ch_found = None
            ch_sig = self.ch.signature(characteristic.ChSignatureType.Full)

            if orig_upper_bound <= self.ch_max_weight:
                solver.add_assertion(bv2pysmt(self.ch_weight < orig_upper_bound))
                if verbose_level >= 1:
                    smart_print("Added assertion:", self.ch_weight < orig_upper_bound)

            for _ in range(MAX_CH_FOUND):
                if last_ch_found is not None:
                    candidate_sig = last_ch_found.signature(characteristic.ChSignatureType.Full)
                    # disable simplification due to recursion error
                    with context.Simplification(False):
                        c = ~operation.BvComp(ch_sig[0], candidate_sig[0])
                        for i in range(1, len(ch_sig)):
                            c |= ~operation.BvComp(ch_sig[i], candidate_sig[i])
                    solver.add_assertion(bv2pysmt(c, boolean=True))
                    if verbose_level >= 3:
                        smart_print("Added assertion:", c)
                if verbose_level >= 1:
                    smart_print(_get_time(), "| Solving")
                satisfiable = solver.solve()

                if satisfiable:
                    model = types.pysmt_model2bv_model(solver.get_model(), differences_in_model)
                    last_ch_found = ch_found_class(self.ch, self.ch_weight, self.der_weights, model)

                    valid_ch = True
                    if check:
                        try:
                            last_ch_found.check_empirical_weight(verbose_level, filename)
                        except ExactWeightError:
                            valid_ch = False
                            if verbose_level >= 1:
                                smart_print(_get_time(), "| Found invalid characteristic")
                                smart_print(last_ch_found)
                                if verbose_level >= 2:
                                    smart_print(last_ch_found.vrepr())

                    if valid_ch:
                        smart_print(_get_time(), "| Found characteristic")
                        smart_print(last_ch_found)
                        if verbose_level >= 2:
                            smart_print(last_ch_found.vrepr())
                else:
                    if verbose_level >= 1:
                        if last_ch_found is not None:
                            smart_print(_get_time(), "| No more characteristics found")
                        else:
                            smart_print(_get_time(), "| No characteristic found")
                        smart_print()
                    solver.exit()
                    return

        if search_mode in [search_mode_class.Optimal, search_mode_class.OptimalDifferential,
                           search_mode_class.FirstChValid, search_mode_class.AllOptimal]:
            solver.push()

            target_weight = initial_weight

            if search_mode in [search_mode_class.Optimal, search_mode_class.AllOptimal]:
                signature_type = characteristic.ChSignatureType.Full
            elif search_mode in [search_mode_class.FirstChValid, search_mode_class.OptimalDifferential]:
                signature_type = characteristic.ChSignatureType.InputOutput

            ch_sig = self.ch.signature(signature_type)

            solver.add_assertion(bv2pysmt(
                operation.BvComp(self.ch_weight, core.Constant(target_weight, self.ch_weight.width)), boolean=True))

            # yield must be hidden in closure so that the method is only a generator
            # when return_generator is True
            def iterate_results():
                nonlocal target_weight, upper_bound
                last_ch_found = None
                best_ch_found = None
                min_exact_weight = math.inf
                while target_weight < upper_bound:
                    if last_ch_found is not None:
                        candidate_sig = last_ch_found.signature(signature_type)
                        assert len(candidate_sig) > 0
                        c = ~operation.BvComp(ch_sig[0], candidate_sig[0])
                        for i in range(1, len(ch_sig)):
                            c |= ~operation.BvComp(ch_sig[i], candidate_sig[i])
                        solver.add_assertion(bv2pysmt(c, boolean=True))
                        if verbose_level >= 3:
                            smart_print("Added assertion:", c)
                    if verbose_level >= 1:
                        smart_print(_get_time(), "| Solving", operation.BvComp(self.ch_weight, core.Constant(target_weight, self.ch_weight.width)))

                    satisfiable = solver.solve()

                    if satisfiable:
                        model = types.pysmt_model2bv_model(solver.get_model(), differences_in_model)
                        assert model[self.ch_weight] == target_weight
                        last_ch_found = ch_found_class(self.ch, self.ch_weight, self.der_weights, model)

                        exact_weight = last_ch_found.get_exact_weight()

                        valid_ch = True

                        if exact_weight < min_exact_weight:
                            if check:
                                try:
                                    last_ch_found.check_empirical_weight(verbose_level, filename)
                                except ExactWeightError:
                                    valid_ch = False
                                    if verbose_level >= 1:
                                        smart_print(_get_time(), "| Found invalid characteristic")
                                        if verbose_level >= 2:
                                            smart_print(last_ch_found.vrepr())

                            if valid_ch:
                                best_ch_found = last_ch_found
                                min_exact_weight = exact_weight
                                if verbose_level >= 1:
                                    smart_print(_get_time(), "| Found better characteristic")
                                    smart_print(last_ch_found)
                                    if verbose_level >= 2:
                                        smart_print(last_ch_found.vrepr())
                        else:
                            if verbose_level >= 1:
                                smart_print(_get_time(), "| Found worse characteristic (not checked)", last_ch_found.srepr())
                                if verbose_level >= 2:
                                    smart_print(last_ch_found.vrepr())

                        if valid_ch:
                            if return_generator:
                                yield last_ch_found
                            elif search_mode == search_mode_class.FirstChValid:
                                return last_ch_found
                            elif search_mode in [search_mode_class.Optimal, search_mode_class.OptimalDifferential]:
                                new_upper_bound = self._new_upper_bound(last_ch_found, upper_bound)
                                if int(exact_weight) == 0 or target_weight >= new_upper_bound:
                                    break
                                if new_upper_bound != upper_bound and verbose_level >= 1:
                                    smart_print("New upper bound:", new_upper_bound)
                                upper_bound = new_upper_bound
                    else:
                        if verbose_level >= 1:
                            if last_ch_found is not None:
                                smart_print(_get_time(), "| No more characteristics found")
                            else:
                                smart_print(_get_time(), "| No characteristic found")

                            if verbose_level >= 3 and last_ch_found is not None:
                                smart_print("Removed assertions")

                            smart_print()

                        solver.pop()

                        target_weight += 1
                        last_ch_found = None
                        solver.push()
                        solver.add_assertion(bv2pysmt(
                            operation.BvComp(self.ch_weight, core.Constant(target_weight, self.ch_weight.width)), boolean=True))

                solver.exit()

                if best_ch_found is not None:
                    return best_ch_found
                else:
                    if verbose_level >= 1:
                        smart_print(_get_time(), "| No valid characteristic found")
                    return

            if return_generator:
                return iterate_results()
            else:
                try:
                    next(iterate_results())
                except StopIteration as result:
                    solver.exit()
                    return result.value

        if search_mode == search_mode_class.TopDifferentials:
            index_ch_found = 0
            diff2weight = dict()
            topdiff = tuple(sorted(diff2weight.items(), key=lambda x: x[1])[:10])

            def sum_weights(w0, w1):
                if w0 == w1:
                    if w0 < 1:
                        return 0
                    else:
                        return w0 - 1

                # return - math.log2(2**(-w0) + 2**(-w1))

                # a < b
                # 2^(-a) + 2^(-b) = 1/2^a + 1/2^b = 2^(b-a)/2^(b) + 1/2^b
                #                 = (2^(b-a) + 1) / 2^b
                # -log2(2^(-a) + 2^(-b)) = (-1) * (log2(2^(b-a) + 1) - b)
                if w0 < w1:
                    a, b = w0, w1
                else:
                    a, b = w1, w0
                try:
                    return -(math.log2(2**(b - a) + 1) - b)
                except OverflowError:
                    from decimal import Decimal
                    a, b = Decimal(a), Decimal(b)
                    return -(Decimal(2**(b-a) + 1).ln()/Decimal(2).ln() - b)

            solver.push()

            min_exact_weight = math.inf
            last_ch_found = None
            best_ch_found = None
            target_weight = initial_weight

            signature_type = characteristic.ChSignatureType.InputOutput

            ch_sig = self.ch.signature(signature_type)

            solver.add_assertion(bv2pysmt(
                operation.BvComp(self.ch_weight, core.Constant(target_weight, self.ch_weight.width)), boolean=True))

            while target_weight < upper_bound:
                if last_ch_found is not None:
                    candidate_sig = last_ch_found.signature(signature_type)
                    assert len(candidate_sig) > 0
                    c = ~operation.BvComp(ch_sig[0], candidate_sig[0])
                    for i in range(1, len(ch_sig)):
                        c |= ~operation.BvComp(ch_sig[i], candidate_sig[i])
                    solver.add_assertion(bv2pysmt(c, boolean=True))
                    if verbose_level >= 3:
                        smart_print("Added assertion:", c)
                if verbose_level >= 1 and last_ch_found is None:
                    # only print the target_weight for the 1st characteristic
                    smart_print(_get_time(), "| Solving", operation.BvComp(self.ch_weight, core.Constant(target_weight, self.ch_weight.width)))

                satisfiable = solver.solve()

                if satisfiable:
                    model = types.pysmt_model2bv_model(solver.get_model(), differences_in_model)
                    assert model[self.ch_weight] == target_weight
                    last_ch_found = ch_found_class(self.ch, self.ch_weight, self.der_weights, model)

                    index_ch_found += 1

                    exact_weight = last_ch_found.get_exact_weight()

                    valid_ch = True
                    if check:
                        try:
                            last_ch_found.check_empirical_weight(verbose_level, filename)
                        except ExactWeightError:
                            valid_ch = False
                            if verbose_level >= 1:
                                smart_print(_get_time(), "| Found invalid characteristic")
                                if verbose_level >= 2:
                                    smart_print(last_ch_found.vrepr())
                    if not valid_ch:
                        continue

                    if exact_weight < min_exact_weight:
                        best_ch_found = last_ch_found
                        min_exact_weight = exact_weight
                        if verbose_level >= 1:
                            smart_print(_get_time(), "| Found better characteristic")
                            smart_print(last_ch_found)
                            if verbose_level >= 2:
                                smart_print(last_ch_found.vrepr())

                    input_diff = [value.val for var, value in last_ch_found.input_diff]
                    output_diff = [value.val for var, value in last_ch_found.output_diff]
                    in_out_diff = tuple(input_diff + output_diff)

                    if in_out_diff not in diff2weight:
                        diff2weight[in_out_diff] = exact_weight
                    else:
                        diff2weight[in_out_diff] = sum_weights(diff2weight[in_out_diff], exact_weight)
                        if verbose_level >= 2:
                            smart_print("diff {} | sum_weights({}, {}) = {}".format(
                                in_out_diff,
                                diff2weight[in_out_diff],
                                exact_weight,
                                sum_weights(diff2weight[in_out_diff], exact_weight)
                            ))


                    newtopdiff = tuple(sorted(diff2weight.items(), key=lambda x: x[1])[:10])

                    if verbose_level >= 1 and (index_ch_found & (index_ch_found - 1)) == 0:
                        smart_print(_get_time(), "| Found 2^{} characteristics".format(int(math.log2(index_ch_found))))

                    if len(topdiff) == 9 and len(newtopdiff) == 10:
                        if verbose_level >= 1:
                            smart_print(_get_time(), "| First top 10 differentials")
                            smart_print(newtopdiff)
                    elif len(newtopdiff) == 10 and newtopdiff != topdiff:
                        if verbose_level >= 1:
                            differing = [i_d for i_d, (old_d, new_d) in enumerate(zip(topdiff, newtopdiff)) if old_d != new_d]
                            smart_print(_get_time(), "| Found new top 10 differentials differing in", differing)
                            smart_print(newtopdiff)
                            # smart_print(diff2weight)
                    topdiff = newtopdiff

                    if index_ch_found > MAX_CH_FOUND:
                        return best_ch_found, diff2weight

                else: # if not satisfiable
                    if verbose_level >= 1:
                        if last_ch_found is not None:
                            smart_print(_get_time(), "| No more characteristics found")
                        else:
                            smart_print(_get_time(), "| No characteristic found")

                        if verbose_level >= 3 and last_ch_found is not None:
                            smart_print("Removed assertions")

                        smart_print()

                    solver.pop()

                    target_weight += 1
                    last_ch_found = None
                    solver.push()
                    solver.add_assertion(bv2pysmt(
                        operation.BvComp(self.ch_weight, core.Constant(target_weight, self.ch_weight.width)), boolean=True))

            solver.exit()

            if best_ch_found is not None:
                return best_ch_found, diff2weight
            else:
                if verbose_level >= 1:
                    smart_print(_get_time(), "| No valid characteristic found")
                return

    def formula_size(self, measure=None):
        """Return the size of the underlying SMT problem.

        See `pysmt.oracles.SizeOracle` for choosing the ``measure``.
        """
        env = environment.reset_env()
        bv2pysmt = functools.partial(types.bv2pysmt, env=env)
        assertions = [bv2pysmt(c, boolean=True) for c in self.constraints]
        formula = env.formula_manager.And(*assertions)
        return env.sizeo.get_size(formula, measure)

    def hrepr(self, full_repr=False, minimize_constraint=True):
        """Return a human readable representing of the SMT problem.

        If ``full_repr`` is False, the short string representation `srepr` is used.
        """
        representation = []
        for c in self.constraints:
            if full_repr:
                s = str(c)
            else:
                s = c.srepr()
            representation.append("assert " + s)
        if minimize_constraint:
            representation.append("minimize {}".format(self.ch_weight))
        return "\n".join(representation)

    def error(self):
        """Return an upper bound of the weight error of the SMT problem."""
        if self.der_mode == DerMode.ProbabilityOne:
            return 0
        elif self.der_mode == DerMode.Valid:
            return 0
        else:
            e = 0
            for der in self.ch.nonlinear_diffs.values():
                e += der.error()
            return e

    def _new_upper_bound(self, ch_found, previous_bound=math.inf):
        """Computes the new upper bound given the previous solution.

        Let me the maximum error of the characteristic (after applying the ceil function).

        For the previous solution, let:
         - eiw the integer part of the exact weight
         - ew the exact weight

        Assuming there exists a better new solution, let denote by eiw' and ew'
        the integer part and the full exact weight for this new solution.

        The worst case (when the new upper bound reduces the least)
        would be when the better solution has exact weight close
        to the previous solution, that is, eiw - 1 < ew' < eiw.

        The weight of this better solution must belong to the following
        interval with integer endpoints: [eiw - 1, eiw - 1 + me]
        (assuming the exact weight is smaller than the weight).

        Therefore the SMT should find characteristics with integer
        part of the ch_weight up to (but not included) the value eiw + me.
        """
        exact_weight = ch_found.get_exact_weight()
        new_bound = math.floor(exact_weight) + math.ceil(self.error())
        return min(previous_bound, new_bound, self.ch_max_weight + 1)


class SearchSkCh(SearchCh):
    """Represent the problem of finding a `SingleKeyCh` with conditions on the weight.

    Args:
        skch (SingleKeyCh): a symbolic single-key characteristic of a `Cipher`
        der_mode (DerMode): one of the modes available for the derivative weights
        allow_zero_input_diff(bool): if ``True``, allow the input difference to be zero

    See also `SearchCh`.

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import SingleKeyCh
        >>> from arxpy.smt.search_differential import SearchSkCh
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_rounds(1)
        >>> ch = SingleKeyCh(Speck32, XorDiff)
        >>> search_problem = SearchSkCh(ch)
        >>> search_problem.formula_size()
        349
        >>> search_problem.error()
        0
        >>> print(search_problem.hrepr(False))
        assert ~(0x00000000 == (dp0 :: dp1))
        assert 0x0000 == ((~(... << ...) ^ (dx1 << 0x0001)) & (~(... << ...) ^ ((... >>> ...) << 0x0001)) & ((dp1 << 0x0001) ^ dx1 ^ dp1 ^ (... >>> ...)))
        assert w0 == PopCount(~((dx1 ^ ~...) & (~... ^ (... >>> ...)))[14:])
        assert w == w0
        minimize w
        >>> search_problem.solve(0).ch_weight
        0x0

    """

    def __init__(self, skch, der_mode=DerMode.Default, allow_zero_input_diff=False,
                 initial_constraints=None):
        assert isinstance(skch, characteristic.SingleKeyCh)
        super().__init__(skch, der_mode=der_mode, allow_zero_input_diff=allow_zero_input_diff,
                         initial_constraints=initial_constraints)


    def solve(self, initial_weight, solver_name="btor", search_mode=SkChSearchMode.Optimal, check=False,
              return_generator=False, verbose_level=0, filename=None):
        """Solve the SMT problem associated  to the search of a valid `SingleKeyCh`.

        See also `SearchCh.solve`.
        """
        assert initial_weight < sum(self.ch.func.input_widths) + math.floor(self.error())
        assert isinstance(search_mode, SkChSearchMode)

        return self._incremental_solve(initial_weight, solver_name, search_mode, check, return_generator,
                                       verbose_level, filename)


class SearchRkCh(object):
    """Represent the problem of finding an `RelatedKeyCh` with conditions on the weight.

    Args:
        rkch (RelatedKeyCh): a symbolic related-key characteristic of a `Cipher`
        key_der_mode (DerMode): the derivative mode for the key schedule characteristic
        enc_der_mode (DerMode): the derivative mode for the encryption characteristic
        allow_zero_key_input_diff (bool): if ``True``, allow the input difference
            of the key schedule to be zero
        allow_zero_enc_input_diff (bool): if ``True``, allow the input difference
            of the encryption to be zero

    See also `SearchCh`.

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import RelatedKeyCh
        >>> from arxpy.smt.search_differential import SearchRkCh
        >>> from arxpy.primitives.lea import LeaCipher
        >>> LeaCipher.set_rounds(1)
        >>> rkch = RelatedKeyCh(LeaCipher, XorDiff)
        >>> search_problem = SearchRkCh(rkch)
        >>> search_problem.formula_size()
        50495
        >>> search_problem.key_schedule_problem.error(), search_problem.encryption_problem.error()
        (4.199250014423123, 0)
        >>> print(search_problem.hrepr(False))
        assert ~(0x00000000000000000000000000000000 == (dmk0 :: dmk1 :: dmk2 :: dmk3))
        assert 0x00000000 == (((... << ...) & (... << ...) & ~(... ^ ...) & (~(... << ...) | (~... ^ ... ^ ...))) | (~(... << ...) & ~(... << ...) & (dk0 ^ dmk0)))
        assert wk0_tmp_0lz == LeadingZeros(~((~... & ~...) << 0x00000001))
        assert wk0_tmp_1rev == Reverse((~dk0 & ~dmk0) << 0x00000001)
        assert wk0_tmp_2rev == Reverse(~(... >> ...) & ((... & ...) << 0x00000001) & (~(... ^ ...) >> 0x00000001))
        assert wk0_tmp_3rev == Reverse(wk0_tmp_2rev ^ wk0_tmp_1rev ^ (wk0_tmp_1rev + wk0_tmp_2rev))
        assert wk0_tmp_4rev == Reverse((((... & ...) + (... & ...)) & ~(... | ...)) | ((... & ... & (... >> ...)) - ((... + ...) & (... | ...))))
        assert wk0 == (((PopCountDiff((... & ...) | (... << ...), (... & ...) ^ ... ^ ...)) :: 0b000) - (0b00 :: ((0b0 :: PopCount(...)) + (PopCount(...) :: 0b0))))
        assert 0x00000000 == (((... << ...) & (... << ...) & ~(... ^ ...) & (~(... << ...) | (~... ^ ... ^ ...))) | (~(... << ...) & ~(... << ...) & (dk2 ^ dmk1)))
        assert wk1_tmp_0lz == LeadingZeros(~((~... & ~...) << 0x00000001))
        assert wk1_tmp_1rev == Reverse((~dk2 & ~dmk1) << 0x00000001)
        assert wk1_tmp_2rev == Reverse(~(... >> ...) & ((... & ...) << 0x00000001) & (~(... ^ ...) >> 0x00000001))
        assert wk1_tmp_3rev == Reverse(wk1_tmp_2rev ^ wk1_tmp_1rev ^ (wk1_tmp_1rev + wk1_tmp_2rev))
        assert wk1_tmp_4rev == Reverse((((... & ...) + (... & ...)) & ~(... | ...)) | ((... & ... & (... >> ...)) - ((... + ...) & (... | ...))))
        assert wk1 == (((PopCountDiff((... & ...) | (... << ...), (... & ...) ^ ... ^ ...)) :: 0b000) - (0b00 :: ((0b0 :: PopCount(...)) + (PopCount(...) :: 0b0))))
        assert 0x00000000 == (((... << ...) & (... << ...) & ~(... ^ ...) & (~(... << ...) | (~... ^ ... ^ ...))) | (~(... << ...) & ~(... << ...) & (dk4 ^ dmk2)))
        assert wk2_tmp_0lz == LeadingZeros(~((~... & ~...) << 0x00000001))
        assert wk2_tmp_1rev == Reverse((~dk4 & ~dmk2) << 0x00000001)
        assert wk2_tmp_2rev == Reverse(~(... >> ...) & ((... & ...) << 0x00000001) & (~(... ^ ...) >> 0x00000001))
        assert wk2_tmp_3rev == Reverse(wk2_tmp_2rev ^ wk2_tmp_1rev ^ (wk2_tmp_1rev + wk2_tmp_2rev))
        assert wk2_tmp_4rev == Reverse((((... & ...) + (... & ...)) & ~(... | ...)) | ((... & ... & (... >> ...)) - ((... + ...) & (... | ...))))
        assert wk2 == (((PopCountDiff((... & ...) | (... << ...), (... & ...) ^ ... ^ ...)) :: 0b000) - (0b00 :: ((0b0 :: PopCount(...)) + (PopCount(...) :: 0b0))))
        assert (0b0000000000000000000000000000000 == ((... & ... & ~... & (~... | (... ^ ...))) | (~... & ~... & ((dk6[:1]) ^ (dmk3[:1]))))) & ((dk6[0]) == (dmk3[0]))
        assert wk3_tmp_0lz == LeadingZeros(~((~... & ~...) << 0b0000000000000000000000000000001))
        assert wk3_tmp_1rev == Reverse((~(dk6[:1]) & ~(dmk3[:1])) << 0b0000000000000000000000000000001)
        assert wk3_tmp_2rev == Reverse(~(... >> ...) & ((... & ...) << 0b0000000000000000000000000000001) & (~(... ^ ...) >> 0b0000000000000000000000000000001))
        assert wk3_tmp_3rev == Reverse(wk3_tmp_2rev ^ wk3_tmp_1rev ^ (wk3_tmp_1rev + wk3_tmp_2rev))
        assert wk3_tmp_4rev == Reverse((((... & ...) + (... & ...)) & ~(... | ...)) | ((... & ... & (... >> ...)) - ((... + ...) & (... | ...))))
        assert wk3 == (((PopCountDiff((... & ...) | (... << ...), (... & ...) ^ ... ^ ...)) :: 0b000) - (0b00 :: ((0b0 :: PopCount(...)) + (PopCount(...) :: 0b0))))
        assert wk == (((0b0 :: wk0) + (0b0 :: wk1) + (0b0 :: wk2) + (0b00 :: wk3))[:3])
        assert 0x00000000 == ((~(... << ...) ^ (dx2 << 0x00000001)) & (~(... << ...) ^ ((... ^ ...) << 0x00000001)) & (((dp0 ^ (... <<< ...)) << 0x00000001) ^ dx2 ^ dp0 ^ (... <<< ...) ^ dp1 ^ (... <<< ...)))
        assert 0x00000000 == ((~(... << ...) ^ (dx6 << 0x00000001)) & (~(... << ...) ^ ((... ^ ...) << 0x00000001)) & (((dp1 ^ (... <<< ...)) << 0x00000001) ^ dx6 ^ dp1 ^ (... <<< ...) ^ dp2 ^ (... <<< ...)))
        assert 0x00000000 == ((~(... << ...) ^ (dx10 << 0x00000001)) & (~(... << ...) ^ ((... ^ ...) << 0x00000001)) & (((dp2 ^ (... <<< ...)) << 0x00000001) ^ dx10 ^ dp2 ^ (... <<< ...) ^ dp3 ^ (... <<< ...)))
        assert w0w1w2 == (PopCountSum3(~((dx10 ^ ~...) & (~... ^ ... ^ ...))[30:], ~((dx2 ^ ~...) & (~... ^ ... ^ ...))[30:], ~((dx6 ^ ~...) & (~... ^ ... ^ ...))[30:]))
        assert w == w0w1w2
        minimize wk, w
        >>> rkch_found = search_problem.solve(0, 0, solver_name="btor")
        >>> rkch_found.key_ch_found.ch_weight, rkch_found.enc_ch_found.ch_weight
        (0b0000000, 0b0000000)

    """

    def __init__(self, rkch, key_der_mode=DerMode.Default, enc_der_mode=DerMode.Default,
                 allow_zero_key_input_diff=False, allow_zero_enc_input_diff=True,
                 initial_key_constraints=None, initial_enc_constraints=None):
        assert isinstance(rkch, characteristic.RelatedKeyCh)

        env = environment.reset_env()
        self._env = env

        self.key_schedule_problem = SearchCh(
            ch=rkch.key_schedule_ch, allow_zero_input_diff=allow_zero_key_input_diff,
            der_mode=key_der_mode, weight_prefix="wk", env=env,
            initial_constraints=initial_key_constraints)

        self.encryption_problem = SearchCh(
            ch=rkch.encryption_ch, allow_zero_input_diff=allow_zero_enc_input_diff,
            der_mode=enc_der_mode, weight_prefix="w", env=env,
            initial_constraints=initial_enc_constraints)

        self.rkch = rkch

    def formula_size(self, measure=None):
        """Return the size of the underlying SMT problem.

        See  also `SearchCh.formula_size`.
        """
        return self.key_schedule_problem.formula_size(measure) + self.encryption_problem.formula_size(measure)

    def hrepr(self, full_repr=False, minimize_constraint=True):
        """Return a human readable representing of the SMT problem.

        See  also `SearchCh.hrepr`.
        """
        key_hrepr = self.key_schedule_problem.hrepr(full_repr, minimize_constraint=False)
        enc_hrepr = self.encryption_problem.hrepr(full_repr, minimize_constraint=False)
        representation = key_hrepr + "\n" + enc_hrepr
        if minimize_constraint:
            representation += "\nminimize {}, {}".format(
                self.key_schedule_problem.ch_weight, self.encryption_problem.ch_weight)
        return representation

    def solve(self, initial_ew, initial_kw, solver_name="btor", search_mode=RkChSearchMode.OptimalMinSum,
              check=False, verbose_level=0, filename=None, return_generator=False):
        """Solve the SMT problem associated to the search a valid `RelatedKeyCh`.

        Args:
             initial_ew(int): the initial encryption weight for starting the iterative search
             initial_kw(int): the initial key schedule weight for starting the iterative search
             solver_name(str): the name of the solver (according to pySMT) to be used
             search_mode(RkChSearchMode): one of the search modes available
             check(bool): if ``True``, `RkChFound.check_empirical_weight` will be called
                after a characteristic is found. If it is not valid, the search will continue.
             verbose_level(int): an integer between ``0`` (no verbose) and ``3`` (full verbose).
             filename(str): if not ``None``, the output will be  printed to the given file
                rather than the to stdout.

        """
        kp = self.key_schedule_problem
        ep = self.encryption_problem
        assert initial_ew < sum(ep.ch.func.input_widths) + math.floor(ep.error())
        assert initial_kw < sum(kp.ch.func.input_widths) + math.floor(kp.error())
        assert isinstance(search_mode, RkChSearchMode)

        return self._incremental_solve(initial_ew, initial_kw, solver_name, search_mode, check,
                                       verbose_level, filename, return_generator)

    # noinspection PyPep8
    def _incremental_solve(self, initial_ew, initial_kw, solver_name, search_mode, check,
                           verbose_level, filename, return_generator):
        strict_shift = True if solver_name == "btor" else False
        bv2pysmt = functools.partial(types.bv2pysmt, env=self._env, strict_shift=strict_shift)
        smart_print = _get_smart_print(filename)

        if return_generator and search_mode != RkChSearchMode.AllValid:
            raise ValueError("return_generator only supports AllValid search mode")

        kp = self.key_schedule_problem
        ep = self.encryption_problem

        def sum_weights(key_weight, enc_weight):
            ch_max_weight = kp.ch_max_weight + ep.ch_max_weight
            ch_width = max(ch_max_weight.bit_length(), key_weight.width, enc_weight.width)
            return operation.ZeroExtend(key_weight, ch_width - key_weight.width) + \
                   operation.ZeroExtend(enc_weight, ch_width - enc_weight.width)

        differences_in_model = []
        for d in itertools.chain(kp.ch.input_diff, kp.ch.nonlinear_diffs.keys(),
                                 ep.ch.input_diff, ep.ch.nonlinear_diffs.keys()):
            differences_in_model.append(d)

        solver =  self._env.factory.Solver(name=solver_name, logic=logics.QF_BV)

        for c in kp.constraints:
            solver.add_assertion(bv2pysmt(c, boolean=True))

        for c in ep.constraints:
            solver.add_assertion(bv2pysmt(c, boolean=True))

        key_max_error = kp.error()
        key_orig_upper_bound = sum(kp.ch.func.input_widths) + math.ceil(key_max_error)
        key_upper_bound = min(key_orig_upper_bound, kp.ch_max_weight + 1)

        enc_max_error = ep.error()
        enc_orig_upper_bound = sum(ep.ch.func.input_widths) + math.ceil(enc_max_error)
        enc_upper_bound = min(enc_orig_upper_bound, ep.ch_max_weight + 1)

        if verbose_level >= 1:
            smart_print("Key schedule ch. upper bound: {}, max error: {}".format(
                key_upper_bound, key_max_error
            ))
            smart_print("Encryption ch. upper bound: {}, max error: {}\n".format(
                enc_upper_bound, enc_max_error
            ))

        if search_mode in [RkChSearchMode.FirstMinSum,
                           RkChSearchMode.FirstValidKeyMinEnc,
                           RkChSearchMode.FirstFixEncMinKey]:

            if search_mode == RkChSearchMode.FirstMinSum:
                target_weight = int(initial_ew) + int(initial_kw)
                upper_bound = min(
                    enc_upper_bound + key_upper_bound,
                    kp.ch_max_weight + ep.ch_max_weight + 1  # to not count +1 twice
                )
                get_assertion = lambda tw: operation.BvComp(sum_weights(kp.ch_weight, ep.ch_weight),
                                                            core.Constant(tw, sum_weights(kp.ch_weight, ep.ch_weight).width))
            elif search_mode == RkChSearchMode.FirstValidKeyMinEnc:
                target_weight = initial_ew
                upper_bound = enc_upper_bound
                if key_orig_upper_bound <= kp.ch_max_weight:
                    solver.add_assertion(bv2pysmt(kp.ch_weight < key_orig_upper_bound))
                    if verbose_level >= 1:
                        smart_print("Added assertion:", kp.ch_weight < key_orig_upper_bound)
                get_assertion = lambda tw: operation.BvComp(ep.ch_weight,
                                                            core.Constant(tw, ep.ch_weight.width))
            elif search_mode == RkChSearchMode.FirstFixEncMinKey:
                target_weight = initial_kw
                upper_bound = key_upper_bound
                solver.add_assertion(bv2pysmt(
                    operation.BvComp(ep.ch_weight, initial_ew), boolean=True))
                if verbose_level >= 1:
                    smart_print("Added assertion:", operation.BvComp(ep.ch_weight, initial_ew))
                get_assertion = lambda tw: operation.BvComp(kp.ch_weight,
                                                            core.Constant(tw, kp.ch_weight.width))

            while target_weight < upper_bound:
                assertion = get_assertion(target_weight)
                if verbose_level >= 1:
                    smart_print(_get_time(), "| Solving", assertion)
                satisfiable = solver.solve([bv2pysmt(assertion, boolean=True)])

                if satisfiable:
                    model = types.pysmt_model2bv_model(solver.get_model(), differences_in_model)
                    if search_mode == RkChSearchMode.FirstMinSum:
                        assert sum_weights(model[kp.ch_weight], model[ep.ch_weight]) == target_weight
                    elif search_mode == RkChSearchMode.FirstValidKeyMinEnc:
                        assert model[ep.ch_weight] == target_weight
                    elif search_mode == RkChSearchMode.FirstFixEncMinKey:
                        assert model[kp.ch_weight] == target_weight

                    rkch_found = RkChFound(self.rkch, self.key_schedule_problem, self.encryption_problem, model)

                    if check:
                        try:
                            rkch_found.check_empirical_weight(verbose_level, filename)
                        except ExactWeightError:
                            rkch_found.key_ch_found.emp_weight = math.inf
                            rkch_found.enc_ch_found.emp_weight = math.inf

                    solver.exit()
                    return rkch_found
                else:
                    target_weight += 1
            else:
                if verbose_level >= 1:
                    smart_print(_get_time(), "| Unsatisfiable")
                solver.exit()
                return None

        if search_mode == RkChSearchMode.AllValid:
            rkch_sig = self.rkch.signature(characteristic.ChSignatureType.Full)

            if key_orig_upper_bound <= kp.ch_max_weight:
                solver.add_assertion(bv2pysmt(kp.ch_weight < key_orig_upper_bound))
                if verbose_level >= 1:
                    smart_print("Added assertion:", kp.ch_weight < key_orig_upper_bound)
            if enc_orig_upper_bound <= ep.ch_max_weight:
                solver.add_assertion(bv2pysmt(ep.ch_weight < enc_orig_upper_bound))
                if verbose_level >= 1:
                    smart_print("Added assertion:", ep.ch_weight < enc_orig_upper_bound)

            def iterate_results():
                last_rkch_found = None
                for _ in range(MAX_CH_FOUND):
                    if last_rkch_found is not None:
                        candidate_sig = last_rkch_found.signature(characteristic.ChSignatureType.Full)
                        # disable simplification due to recursion error
                        with context.Simplification(False):
                            c = ~operation.BvComp(rkch_sig[0], candidate_sig[0])
                            for i in range(1, len(rkch_sig)):
                                c |= ~operation.BvComp(rkch_sig[i], candidate_sig[i])
                        solver.add_assertion(bv2pysmt(c, boolean=True))
                        if verbose_level >= 3:
                            smart_print("Added assertion:", c)
                    if verbose_level >= 1:
                        smart_print(_get_time(), "| Solving")
                    satisfiable = solver.solve()

                    if satisfiable:
                        model = types.pysmt_model2bv_model(solver.get_model(), differences_in_model)
                        last_rkch_found = RkChFound(self.rkch, self.key_schedule_problem, self.encryption_problem, model)

                        valid_ch = True
                        if check:
                            try:
                                last_rkch_found.check_empirical_weight(verbose_level, filename)
                            except ExactWeightError:
                                valid_ch = False
                                if verbose_level >= 1:
                                    smart_print(_get_time(), "| Found invalid characteristic")
                                    smart_print(last_rkch_found)
                                    if verbose_level >= 2:
                                        smart_print(last_rkch_found.vrepr())

                        if valid_ch:
                            if not return_generator or verbose_level >= 1:
                                smart_print(_get_time(), "| Found characteristic:", last_rkch_found.srepr())
                                if verbose_level >= 2:
                                    smart_print(last_rkch_found.vrepr())
                            if return_generator:
                                yield last_rkch_found
                    else:
                        if verbose_level >= 1:
                            if last_rkch_found is not None:
                                smart_print(_get_time(), "| No more characteristics found")
                            else:
                                smart_print(_get_time(), "| No characteristic found")
                        smart_print()
                        return

            if return_generator:
                return iterate_results()
            else:
                try:
                    next(iterate_results())
                except StopIteration as result:
                    solver.exit()
                    return

        if search_mode in [RkChSearchMode.OptimalMinSum,
                           RkChSearchMode.OptimalMinSumDifferential,
                           RkChSearchMode.OptimalValidKeyMinEnc,
                           RkChSearchMode.OptimalValidKeyMinEncDifferential,
                           RkChSearchMode.OptimalFixEncMinKey,
                           RkChSearchMode.OptimalFixEncMinKeyDifferential,
                           RkChSearchMode.FirstMinSumValid,
                           RkChSearchMode.FirstFixEncMinKeyValid,
                           RkChSearchMode.FirstValidKeyMinEncValid,
                           RkChSearchMode.AllOptimalMinSum]:
            min_exact_weight = math.inf
            best_rkch_found = None
            last_rkch_found = None
            signature_type = None

            if search_mode in [RkChSearchMode.OptimalMinSum,
                               RkChSearchMode.OptimalMinSumDifferential,
                               RkChSearchMode.FirstMinSumValid,
                               RkChSearchMode.AllOptimalMinSum]:
                target_weight = int(initial_ew) + int(initial_kw)
                upper_bound = min(
                    enc_upper_bound + key_upper_bound,
                    kp.ch_max_weight + ep.ch_max_weight + 1  # + 1, not + 2
                )
                if search_mode in [RkChSearchMode.OptimalMinSum, RkChSearchMode.AllOptimalMinSum]:
                    signature_type = characteristic.ChSignatureType.Full
                elif search_mode in [RkChSearchMode.OptimalMinSumDifferential, RkChSearchMode.FirstMinSumValid]:
                    signature_type = characteristic.ChSignatureType.InputOutput
                else:
                    raise ValueError("invalid search_mode")
                rkch_sig = self.rkch.signature(signature_type)

                get_assertion = lambda tw: operation.BvComp(sum_weights(kp.ch_weight, ep.ch_weight), tw)
                get_sig = lambda rkf: rkf.signature(signature_type)
                get_sig_str = lambda rkf: rkf.signature(signature_type, return_str=True)
                get_exact_weight = lambda rkf: rkf.key_ch_found.get_exact_weight() + rkf.enc_ch_found.get_exact_weight()
                if search_mode == RkChSearchMode.AllOptimalMinSum:
                    get_upper_bound = lambda rkf, pub: pub
                else:
                    get_upper_bound = lambda rkf, pub: min(
                        kp._new_upper_bound(rkf.key_ch_found, key_upper_bound) +
                        ep._new_upper_bound(rkf.enc_ch_found, enc_upper_bound),
                        pub
                    )

            elif search_mode in [RkChSearchMode.OptimalValidKeyMinEnc,
                                 RkChSearchMode.OptimalValidKeyMinEncDifferential,
                                 RkChSearchMode.FirstValidKeyMinEncValid]:
                target_weight = initial_ew
                upper_bound = enc_upper_bound
                if key_orig_upper_bound <= kp.ch_max_weight:
                    solver.add_assertion(bv2pysmt(kp.ch_weight < key_orig_upper_bound))
                    if verbose_level >= 1:
                        smart_print("Added assertion:", kp.ch_weight < key_orig_upper_bound)
                if search_mode == RkChSearchMode.OptimalValidKeyMinEnc:
                    signature_type = characteristic.ChSignatureType.Full
                else:
                    signature_type = characteristic.ChSignatureType.InputOutput
                rkch_sig = self.rkch.encryption_ch.signature(signature_type)

                get_assertion = lambda tw: operation.BvComp(ep.ch_weight, tw)
                get_sig = lambda rkf: rkf.enc_ch_found.signature(signature_type)
                get_sig_str = lambda rkf: rkf.enc_ch_found.signature(signature_type, return_str=True)
                get_exact_weight = lambda rkf: rkf.enc_ch_found.get_exact_weight()
                get_upper_bound = lambda rkf, pub: ep._new_upper_bound(rkf.enc_ch_found, pub)

            elif search_mode in [RkChSearchMode.OptimalFixEncMinKey,
                                 RkChSearchMode.OptimalFixEncMinKeyDifferential,
                                 RkChSearchMode.FirstFixEncMinKeyValid]:
                target_weight = initial_kw
                upper_bound = key_upper_bound
                solver.add_assertion(bv2pysmt(
                    operation.BvComp(ep.ch_weight, initial_ew), boolean=True))
                if verbose_level >= 1:
                    smart_print("Added assertion:", operation.BvComp(ep.ch_weight, initial_ew))

                if search_mode == RkChSearchMode.OptimalFixEncMinKey:
                    signature_type = characteristic.ChSignatureType.Full
                else:
                    signature_type = characteristic.ChSignatureType.InputOutput
                rkch_sig = self.rkch.key_schedule_ch.signature(signature_type)

                get_assertion = lambda tw: operation.BvComp(kp.ch_weight, tw)
                get_sig = lambda rkf: rkf.key_ch_found.signature(signature_type)
                get_sig_str = lambda rkf: rkf.key_ch_found.signature(signature_type, return_str=True)
                get_exact_weight = lambda rkf: rkf.key_ch_found.get_exact_weight()
                get_upper_bound = lambda rkf, pub: kp._new_upper_bound(rkf.key_ch_found, pub)
            else:
                raise ValueError("invalid search mode: {}".format(search_mode))

            solver.push()
            solver.add_assertion(bv2pysmt(get_assertion(target_weight), boolean=True))

            while target_weight < upper_bound:
                if last_rkch_found is not None:
                    candidate_sig = get_sig(last_rkch_found)
                    assert len(candidate_sig) > 0
                    c = ~operation.BvComp(rkch_sig[0], candidate_sig[0])
                    for i in range(1, len(rkch_sig)):
                        c |= ~operation.BvComp(rkch_sig[i], candidate_sig[i])
                    solver.add_assertion(bv2pysmt(c, boolean=True))
                    if verbose_level >= 3:
                        smart_print("Added assertion:", c)
                if verbose_level >= 1:
                    smart_print(_get_time(), "| Solving", get_assertion(target_weight))
                satisfiable = solver.solve()

                if satisfiable:
                    model = types.pysmt_model2bv_model(solver.get_model(), differences_in_model)

                    last_rkch_found = RkChFound(self.rkch, self.key_schedule_problem, self.encryption_problem, model)

                    exact_weight = get_exact_weight(last_rkch_found)

                    valid_ch = True

                    if exact_weight < min_exact_weight:
                        if check:
                            try:
                                last_rkch_found.check_empirical_weight(verbose_level, filename)
                            except ExactWeightError:
                                valid_ch = False
                                if verbose_level >= 1:
                                    smart_print(_get_time(), "| Found invalid characteristic")
                                    if verbose_level >= 2:
                                        smart_print(last_rkch_found.vrepr())

                        if valid_ch:
                            best_rkch_found = last_rkch_found
                            min_exact_weight = exact_weight
                            if verbose_level >= 1:
                                smart_print(_get_time(), "| Found better characteristic")
                                smart_print(last_rkch_found)
                                if verbose_level >= 2:
                                    smart_print(last_rkch_found.vrepr())
                    else:
                        if verbose_level >= 1:
                            smart_print(_get_time(), "| Found worse characteristic (not checked)")
                            if verbose_level >= 2:
                                smart_print(last_rkch_found.vrepr())

                    if valid_ch:
                        if search_mode in [RkChSearchMode.FirstMinSumValid,
                                           RkChSearchMode.FirstValidKeyMinEncValid,
                                           RkChSearchMode.FirstFixEncMinKeyValid]:
                            solver.exit()
                            return last_rkch_found
                        else:
                            new_upper_bound = get_upper_bound(last_rkch_found, upper_bound)
                            if int(exact_weight) == 0 or target_weight >= upper_bound:
                                break
                            if new_upper_bound != upper_bound and verbose_level >= 1:
                                smart_print("New upper bound:", new_upper_bound)
                            upper_bound = new_upper_bound
                else:
                    if verbose_level >= 1:
                        if last_rkch_found is not None:
                            smart_print(_get_time(), "| No more characteristics found")
                        else:
                            smart_print(_get_time(), "| No characteristic found")
                        if verbose_level >= 3 and last_rkch_found is not None:
                            smart_print("Removed assertions")
                        smart_print()

                    solver.pop()

                    target_weight += 1
                    last_rkch_found = None
                    solver.push()
                    solver.add_assertion(bv2pysmt(get_assertion(target_weight), boolean=True))

            solver.exit()
            if best_rkch_found is not None:
                return best_rkch_found
            else:
                if verbose_level >= 1:
                    smart_print(_get_time(), "| No valid characteristic found")
                return None


def round_based_search_SkCh(cipher, diff_type, initial_weight, solver_name, start_round, end_round,
                            der_mode, search_mode, check, verbose_level, filename,
                            fix_input_diff=None, fix_output_diff=None, fix_round_inputs=None, return_best=None):
    """Find valid single-key characteristics over consecutive rounds.

    Args:
        cipher(Cipher): an (iterated) cipher
        diff_type(Difference): a type of difference
        initial_weight(int): the initial weight for starting the iterative search
        solver_name(str): the name of the solver (according to pySMT) to be used
        start_round(int): the minimum number of rounds to consider
        end_round(int): the maximum number of rounds to consider
        der_mode(DerMode): one of the modes available for the derivative weights
        search_mode(SkChSearchMode): one of the search modes available
        check(bool): if ``True``, `SkChFound.check_empirical_weight` will be called
            after a characteristic is found. If it is not valid, the search will continue.
        verbose_level(int): an integer between ``0`` (no verbose) and ``3`` (full verbose).
        filename(str): if not ``None``, the output will be  printed to the given file
            rather than the to stdout.

    See also `SearchSkCh.solve`.

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.smt.search_differential import SkChSearchMode, DerMode, round_based_search_SkCh
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> round_based_search_SkCh(Speck32, XorDiff, 0, "btor", 1, 2,
        ...                         DerMode.Default, SkChSearchMode.Optimal, True, 0, None)  # doctest:+ELLIPSIS
        Num rounds: 1
        Best characteristic found:
        (weight 0) ... -> ...
        <BLANKLINE>
        Num rounds: 2
        Best characteristic found:
        (weight 1) ... -> ...

    """
    assert start_round <= end_round
    assert search_mode not in [SkChSearchMode.AllValid, SkChSearchMode.AllOptimal]
    assert verbose_level >= 0

    smart_print = _get_smart_print(filename)

    if verbose_level >= 1:
        smart_print(cipher.__name__, "Single-Key Search\n")
        smart_print("Parameters:")
        smart_print("\tcipher:", cipher.__name__)
        smart_print("\tdiff_type:", diff_type.__name__)
        smart_print("\tinitial_weight:", initial_weight)
        smart_print("\tsolver_name:", solver_name)
        smart_print("\tstart:", start_round)
        smart_print("\tend:", end_round)
        smart_print("\tder_mode:", der_mode)
        smart_print("\tsearch_mode:", search_mode)
        smart_print("\tcheck:", check)
        smart_print("\tverbose_level:", verbose_level)
        smart_print("\tfilename:", filename)
        if fix_input_diff is not None:
            smart_print("\tfix_input_diff:", fix_input_diff)
        if fix_output_diff is not None:
            smart_print("\tfix_enc_output_diff:", fix_output_diff)
        if fix_round_inputs is not None:
            smart_print("\tfix_round_inputs:", fix_round_inputs)
        if return_best is not None:
            smart_print("\treturn best:", return_best)
        if hasattr(cipher.encryption, "skip_rounds"):
            smart_print("\tencryption skip_rounds ({}): {}".format(
                len(cipher.encryption.skip_rounds), cipher.encryption.skip_rounds))
        if hasattr(cipher.key_schedule, "skip_rounds"):
            smart_print("\tkey_schedule skip_rounds ({}): {}".format(
                len(cipher.key_schedule.skip_rounds), cipher.key_schedule.skip_rounds))
        smart_print()

    best = None

    for num_rounds in range(start_round, end_round + 1):
        cipher.set_rounds(num_rounds)

        if verbose_level >= 0:
            if num_rounds != start_round:
                smart_print()
            smart_print("Num rounds:", num_rounds)

        ch = characteristic.SingleKeyCh(cipher, diff_type)

        if verbose_level >= 2:
            smart_print("Characteristic:")
            smart_print(ch)

        allow_zero_input_diff = initial_weight > 0

        initial_constraints = []
        if fix_input_diff is not None:
            allow_zero_input_diff = True
            assert len(fix_input_diff) == len(ch.input_diff)
            for i in range(len(ch.input_diff)):
                val = ch.input_diff[i].val
                initial_constraints.append(operation.BvComp(val, fix_input_diff[i]))
        if fix_output_diff is not None:
            assert len(fix_output_diff) == len(ch.output_diff)
            for i in range(len(ch.output_diff)):
                val = ch.output_diff[i][1].val
                initial_constraints.append(operation.BvComp(val, fix_output_diff[i]))
        if fix_round_inputs:
            for fix_list_diff, index_round_state in fix_round_inputs:
                if index_round_state == 0:
                    allow_zero_input_diff = True
                if verbose_level >= 1:
                    smart_print("fixing round_inputs[{}] = {}".format(index_round_state, ch.func.round_inputs[index_round_state]))
                for i in range(len(fix_list_diff)):
                    val = ch.func.round_inputs[index_round_state][i]
                    initial_constraints.append(operation.BvComp(val, fix_list_diff[i]))

        problem = SearchSkCh(skch=ch, der_mode=der_mode, allow_zero_input_diff=allow_zero_input_diff, initial_constraints=initial_constraints)

        if verbose_level >= 2:
            smart_print("SMT problem (size {}):".format(problem.formula_size()))
            smart_print(problem.hrepr(full_repr=verbose_level >= 3))

        ch_found = problem.solve(initial_weight, solver_name=solver_name, search_mode=search_mode, check=check,
                                 verbose_level=verbose_level, filename=filename)

        if verbose_level >= 1:
            prefix = str(_get_time()) + " | "
            smart_print()
        else:
            prefix = ""

        if ch_found is None:
            if verbose_level >= 0:
                smart_print(prefix + "No characteristic found")
            break
        else:
            best = num_rounds, ch_found
            if search_mode == SkChSearchMode.TopDifferentials:
                ch_found, top_differentials = ch_found
            initial_weight = int(ch_found.ch_weight)
            if verbose_level >= 0:
                smart_print(prefix + "Best characteristic found:")
                if verbose_level == 0:
                    smart_print(ch_found.srepr())
                else:
                    smart_print(ch_found)
                if verbose_level >= 3:
                    smart_print(ch_found.vrepr())

                if search_mode == SkChSearchMode.TopDifferentials:
                    smart_print("Best differentials found:")
                    smart_print(top_differentials)

    if return_best:
        return best


def round_based_search_RkCh(cipher, diff_type, initial_ew, initial_kw, solver_name,
                            start_round, end_round, key_der_mode, enc_der_mode, allow_zero_enc_input_diff,
                            search_mode, check, verbose_level, filename,
                            allow_zero_key_input_diff=False,  # for RXDiff
                            fix_key_input_diff=None, fix_enc_input_diff=None, fix_enc_output_diff=None, fix_enc_round_inputs=None, return_best=None):  # for RXDiff
    """Find valid related-key characteristics over consecutive rounds.

    Args:
        cipher(Cipher): an (iterated) cipher
        diff_type(Difference): a type of difference
        initial_ew(int): the initial encryption weight for starting the iterative search
        initial_kw(int): the initial key schedule weight for starting the iterative search
        solver_name(str): the name of the solver (according to pySMT) to be used
        start_round(int): the minimum number of rounds to consider
        end_round(int): the maximum number of rounds to consider
        key_der_mode(DerMode):  the derivative mode for the key schedule characteristic
        enc_der_mode(DerMode):  the derivative mode for the encryption characteristic
        allow_zero_enc_input_diff (bool): if ``True``, allow the input difference
            of the encryption to be zero
        search_mode(RkChSearchMode): one of the search modes available
        check(bool): if ``True``, `RkChFound.check_empirical_weight` will be called
            after a characteristic is found. If it is not valid, the search will continue.
        verbose_level(int): an integer between ``0`` (no verbose) and ``3`` (full verbose).
        filename(str): if not ``None``, the output will be  printed to the given file
            rather than the to stdout.
        allow_zero_key_input_diff (bool): if ``True``, allow the input difference
            of the key schedule to be zero.

    See also `SearchRkCh.solve`.

        >>> from arxpy.differential.difference import XorDiff, RXDiff
        >>> from arxpy.smt.search_differential import RkChSearchMode, DerMode, round_based_search_RkCh
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> round_based_search_RkCh(Speck32, XorDiff, 0, 0, "btor",
        ...                         4, 5, DerMode.Default, DerMode.Default, True,
        ...                         RkChSearchMode.OptimalMinSum, True, 0, None)  # doctest:+ELLIPSIS
        Num rounds: 4
        Best related-key characteristic found:
        K: (weight 0) ... -> ... | E: (weight 0) ... -> ...
        <BLANKLINE>
        Num rounds: 5
        Best related-key characteristic found:
        K: (weight 0) ... -> ... | E: (weight 1) ... -> ...
        >>> round_based_search_RkCh(Speck32, RXDiff, 0, 0, "btor",
        ...                         2, 3, DerMode.Default, DerMode.Default, True,
        ...                         RkChSearchMode.FirstMinSumValid, True, 0, None, True)  # doctest:+ELLIPSIS
        Num rounds: 2
        Best related-key characteristic found:
        K: (weight 1) ... -> ... | E: (weight 2) ... -> ...
        <BLANKLINE>
        Num rounds: 3
        Best related-key characteristic found:
        K: (weight 2) ... -> ... | E: (weight 4) ... -> ...

    """
    assert start_round <= end_round
    assert search_mode != RkChSearchMode.AllValid
    assert verbose_level >= 0

    smart_print = _get_smart_print(filename)

    if verbose_level >= 1:
        smart_print(cipher.__name__, "Related-Key Search\n")
        smart_print("Parameters:")
        smart_print("\tcipher:", cipher.__name__)
        smart_print("\tdiff_type:", diff_type.__name__)
        smart_print("\tinitial_ew:", initial_ew)
        smart_print("\tinitial_kw:", initial_kw)
        smart_print("\tsolver_name:", solver_name)
        smart_print("\tstart:", start_round)
        smart_print("\tend:", end_round)
        smart_print("\tkey_der_mode:", key_der_mode)
        smart_print("\tenc_der_mode:", enc_der_mode)
        smart_print("\tallow_zero_enc_input_diff:", allow_zero_enc_input_diff)
        smart_print("\tsearch_mode:", search_mode)
        smart_print("\tcheck:", check)
        smart_print("\tverbose_level:", verbose_level)
        smart_print("\tfilename:", filename)
        if allow_zero_key_input_diff:
            smart_print("\tallow_zero_key_input_diff:", allow_zero_key_input_diff)
        if fix_key_input_diff is not None:
            smart_print("\tfix_key_input_diff:", fix_key_input_diff)
        if fix_enc_input_diff is not None:
            smart_print("\tfix_enc_input_diff:", fix_enc_input_diff)
        if fix_enc_output_diff is not None:
            smart_print("\tfix_enc_output_diff:", fix_enc_output_diff)
        if fix_enc_round_inputs is not None:
            smart_print("\tfix_enc_round_inputs:", fix_enc_round_inputs)
        if return_best is not None:
            smart_print("\treturn_best:", return_best)
        if hasattr(cipher.encryption, "skip_rounds"):
            smart_print("\tencryption skip_rounds ({}): {}".format(
                len(cipher.encryption.skip_rounds), cipher.encryption.skip_rounds))
        if hasattr(cipher.key_schedule, "skip_rounds"):
            smart_print("\tkey_schedule skip_rounds ({}): {}".format(
                len(cipher.key_schedule.skip_rounds), cipher.key_schedule.skip_rounds))
        smart_print()

    best = None

    for num_rounds in range(start_round, end_round + 1):
        cipher.set_rounds(num_rounds)

        if verbose_level >= 0:
            if num_rounds != start_round:
                smart_print()
            smart_print("Num rounds:", num_rounds)

        ch = characteristic.RelatedKeyCh(cipher, diff_type)

        if verbose_level >= 2:
            smart_print("Characteristic:")
            smart_print(ch)

        initial_key_constraints = []
        if fix_key_input_diff is not None:
            assert len(fix_key_input_diff) == len(ch.key_schedule_ch.input_diff)
            allow_zero_key_input_diff = True
            for i in range(len(ch.key_schedule_ch.input_diff)):
                val = ch.key_schedule_ch.input_diff[i].val
                initial_key_constraints.append(operation.BvComp(val, fix_key_input_diff[i]))

        initial_enc_constraints = []
        if fix_enc_input_diff is not None:
            assert len(fix_enc_input_diff) == len(ch.encryption_ch.input_diff)
            allow_zero_enc_input_diff = True
            for i in range(len(ch.encryption_ch.input_diff)):
                val = ch.encryption_ch.input_diff[i].val
                initial_enc_constraints.append(operation.BvComp(val, fix_enc_input_diff[i]))
        if fix_enc_output_diff is not None:
            assert len(fix_enc_output_diff) == len(ch.encryption_ch.output_diff)
            for i in range(len(fix_enc_output_diff)):
                val = ch.encryption_ch.output_diff[i][1].val
                initial_enc_constraints.append(operation.BvComp(val, fix_enc_output_diff[i]))
        if fix_enc_round_inputs is not None:
            for fix_list_diff, index_round_state in fix_enc_round_inputs:
                if index_round_state == 0:
                    allow_zero_enc_input_diff = True
                if verbose_level >= 1:
                    smart_print("fixing round_inputs[{}] = {}".format(index_round_state, ch.encryption_ch.func.round_inputs[index_round_state]))
                for i in range(len(fix_list_diff)):
                    val = ch.encryption_ch.func.round_inputs[index_round_state][i]
                    initial_enc_constraints.append(operation.BvComp(val, fix_list_diff[i]))

        problem = SearchRkCh(rkch=ch, key_der_mode=key_der_mode, enc_der_mode=enc_der_mode,
                             allow_zero_enc_input_diff=allow_zero_enc_input_diff,
                             allow_zero_key_input_diff=allow_zero_key_input_diff,
                             initial_enc_constraints=initial_enc_constraints,
                             initial_key_constraints=initial_key_constraints,)

        if verbose_level >= 2:
            smart_print("SMT problem (size {}):".format(problem.formula_size()))
            smart_print(problem.hrepr(full_repr=verbose_level >= 3))

        rkch_found = problem.solve(initial_ew, initial_kw, solver_name=solver_name, search_mode=search_mode, check=check,
                                   verbose_level=verbose_level, filename=filename)

        if verbose_level >= 1:
            prefix = str(_get_time()) + " | "
            smart_print()
        else:
            prefix = ""

        if rkch_found is None:
            if verbose_level >= 0:
                smart_print(prefix + "No related-key characteristic found")
            break
        else:
            best = num_rounds, rkch_found
            initial_ew = int(rkch_found.enc_ch_found.ch_weight)
            initial_kw = int(rkch_found.key_ch_found.ch_weight)
            if verbose_level >= 0:
                smart_print(prefix + "Best related-key characteristic found:")
                if verbose_level == 0:
                    smart_print(rkch_found.srepr())
                else:
                    smart_print(rkch_found)
                if verbose_level >= 3:
                    smart_print(rkch_found.vrepr())

    if return_best:
        return best