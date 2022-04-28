"""Search for impossible differentials (ID) by modeling the search as a SMT problem."""
import collections
import enum
import functools
import itertools
import math
import pprint

from pysmt import logics

from arxpy.bitvector import core
from arxpy.bitvector import operation
from arxpy.differential import characteristic
from arxpy.differential import difference
from arxpy.primitives import primitives
from arxpy.smt import search_differential
from arxpy.smt import types
from arxpy.smt import verification_impossible

from arxpy.smt.search_differential import _get_time, _get_smart_print


MAX_TOTAL_ACTIVE_BITS = 4
MAX_ASSIGNMENTS = 8


class KeySetting(enum.Enum):
    """Represent the different types of ID of block ciphers."""
    SingleKey = enum.auto()
    RelatedKey = enum.auto()


class SearchMode(enum.Enum):
    """Represent the different options for searching ID

    Attributes:
        FirstID: returns the first ID found by iteratively searching 
            with increasing number of active bits. The ID found is returned 
            as a `IDFound` object. If none is found, ``None`` is returned.
        AllID: similar to `FirstID`, but the search is not stopped
            after the first ID, and all found ID are printed.

    """
    FirstID = enum.auto()
    AllID = enum.auto()


class ActivationMode(enum.Enum):
    """Represent the different options to consider when activating.

    Attributes:
        Default: no restrictions
        SingleBit: activates at most one bit in each word
        MSBbit: activates at most the MSB in each word
        Zero: all words are set to zero

    """
    Default = enum.auto()
    SingleBit = enum.auto()
    MSBBit = enum.auto()
    Zero = enum.auto()


def generate_active_bvlists(list_width, num_active_bits, active_bits_mode):
    """Generate all bitvector list with given width, number of active bits and activation mode.

        >>> list(generate_active_bvlists([2, 2], 1, active_bits_mode=ActivationMode.Default))
        [[0b01, 0b00], [0b10, 0b00], [0b00, 0b01], [0b00, 0b10]]
        >>> list(generate_active_bvlists([2, 2], 1, active_bits_mode=ActivationMode.SingleBit))
        [[0b01, 0b00], [0b00, 0b01]]

    """
    if active_bits_mode == ActivationMode.Zero or num_active_bits == 0:
        if num_active_bits != 0:
            raise ValueError("num_active_bits != 0 but active_bits_mode=Zero")
        yield [core.Constant(0, w) for w in list_width]

    elif active_bits_mode in [ActivationMode.SingleBit, ActivationMode.MSBBit]:
        for combination in itertools.combinations(range(len(list_width)), num_active_bits):
            if active_bits_mode == ActivationMode.MSBBit:
                iterables = [[w_i - 1] for i, w_i in enumerate(list_width) if i in combination]
            else:
                # active_bits_mode == SingleBit
                iterables = [range(w_i - 1) for i, w_i in enumerate(list_width) if i in combination]
            for w_combination in itertools.product(*iterables):
                bv_list = []
                counter_w_c = 0
                for index_w, w in enumerate(list_width):
                    if index_w in combination:
                        bv_list.append(core.Constant(1 << w_combination[counter_w_c], w))
                        counter_w_c += 1
                    else:
                        bv_list.append(core.Constant(0, w))
                yield bv_list

    elif active_bits_mode == ActivationMode.Default:
        # Source: https://stackoverflow.com/a/10838990 and
        #   https://en.wikipedia.org/wiki/Combinatorial_number_system#Applications.
        assert num_active_bits > 0
        total_width = sum(list_width)

        n = total_width
        k = num_active_bits

        def next_combination(x):
            u = (x & -x)
            v = u + x
            return v + (((v ^ x) // u) >> 2)

        x = (1 << k) - 1  # smallest number with k active bits
        while (x >> n) == 0:
            bv = core.Constant(x, n)
            bv_list = []
            sum_w = 0
            for w in list_width:
                bv_list.append(bv[sum_w + w - 1:sum_w])
                sum_w += w
            yield bv_list
            x = next_combination(x)

    else:
        raise ValueError("invalid active_bits_mode")


class IDFound(object):
    """Represent (non-symbolic) ID found over a `BvCharacteristic`.

    Attributes:
        ch: the associated symbolic differential characteristic
        input_diff: a list, where the i-th element is a pair containing
            the i-th input symbolic difference and its value
        output_diff: a list, where the i-th element is a pair containing
            the i-th output symbolic difference and its value

    ::

        >>> from arxpy.bitvector.operation import BvComp
        >>> from arxpy.differential.difference import XorDiff, RXDiff
        >>> from arxpy.differential.characteristic import BvCharacteristic
        >>> from arxpy.primitives.chaskey import ChaskeyPi
        >>> from arxpy.smt.search_impossible import SearchID
        >>> ChaskeyPi.set_rounds(1)
        >>> ch = BvCharacteristic(ChaskeyPi, XorDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> search_problem = SearchID(ch)
        >>> id_found = search_problem.solve(2)
        >>> id_found.input_diff  # doctest: +NORMALIZE_WHITESPACE
        [[XorDiff(dv0), XorDiff(0x00000001)], [XorDiff(dv1), XorDiff(0x00000000)],
        [XorDiff(dv2), XorDiff(0x00000000)], [XorDiff(dv3), XorDiff(0x00000000)]]
        >>> id_found.output_diff  # doctest: +NORMALIZE_WHITESPACE
        [[XorDiff(d7), XorDiff(0x00000001)], [XorDiff(d12), XorDiff(0x00000000)],
        [XorDiff(d13), XorDiff(0x00000000)], [XorDiff(d9), XorDiff(0x00000000)]]
        >>> print(id_found)  # doctest: +NORMALIZE_WHITESPACE
        {'input_diff': [[dv0, 0x00000001], [dv1, 0x00000000], [dv2, 0x00000000], [dv3, 0x00000000]],
        'output_diff': [[d7, 0x00000001], [d12, 0x00000000], [d13, 0x00000000], [d9, 0x00000000]]}
        >>> ch = BvCharacteristic(ChaskeyPi, RXDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> search_problem = SearchID(ch)
        >>> id_found = search_problem.solve(2)
        >>> id_found.input_diff  # doctest: +NORMALIZE_WHITESPACE
        [[RXDiff(dv0), RXDiff(0x00000001)], [RXDiff(dv1), RXDiff(0x00000000)],
        [RXDiff(dv2), RXDiff(0x00000000)], [RXDiff(dv3), RXDiff(0x00000000)]]
        >>> id_found.output_diff  # doctest: +NORMALIZE_WHITESPACE
        [[RXDiff(d7), RXDiff(0x00000001)], [RXDiff(d12), RXDiff(0x00000000)],
        [RXDiff(d13), RXDiff(0x00000000)], [RXDiff(d9), RXDiff(0x00000000)]]
        >>> print(id_found)  # doctest: +NORMALIZE_WHITESPACE
        {'input_diff': [[dv0, 0x00000001], [dv1, 0x00000000], [dv2, 0x00000000], [dv3, 0x00000000]],
        'output_diff': [[d7, 0x00000001], [d12, 0x00000000], [d13, 0x00000000], [d9, 0x00000000]]}

    """
    def __init__(self, ch, input_diff_found, output_diff_found):
        self.ch = ch

        assert all(isinstance(d, difference.Difference) for d in input_diff_found)
        assert all(isinstance(d, difference.Difference) for d in output_diff_found)
        assert len(input_diff_found) == len(self.ch.input_diff)
        assert len(output_diff_found) in [0, len(self.ch.output_diff)]

        # diff_model results in a logical error
        # when the input and output diff shared a variable with a != value
        #
        # diff_model = {}
        # for i in range(len(input_diff_found)):
        #     diff_model[self.ch.input_diff[i]] = input_diff_found[i]
        # for i in range(len(output_diff_found)):
        #     diff_model[self.ch.output_diff[i][0]] = output_diff_found[i]
        #
        # self._diff_model = diff_model
        #
        # self.input_diff = [[d, diff_model[d]] for d in self.ch.input_diff]
        # self.output_diff = [[d, diff_model[d]] for d, _ in self.ch.output_diff]

        self.input_diff = [[var, value] for var, value in zip(self.ch.input_diff, input_diff_found)]
        self.output_diff = [[var, value] for (var, _), value in zip(self.ch.output_diff, output_diff_found)]

        self.emp_weight = None

    def signature(self, return_str=False):
        sig = []
        for value, value in itertools.chain(self.input_diff, self.output_diff):
            sig.append(value.val)
        if return_str:
            return ''.join([str(s) for s in sig])
        else:
            return sig

    def check_empirical_weight(self, verbose_lvl=0, filename=None):
        """Check the empirical weight is `math.inf`.

        If the empirical weight is not `math.inf`, an `ExactWeightError`
        exception is raised.

        If ``filename`` is not ``None``, the output will be printed
        to the given file rather than the to stdout.

        The argument ``verbose_lvl`` can also take the values ``1`` and ``2`` for a
        more detailed output.
        """
        smart_print = _get_smart_print(filename)

        input_diff = [value for var, value in self.input_diff]
        output_diff = [value for var, value in self.output_diff]

        if hasattr(self.ch, "_cipher"):
            weak_check = False if not hasattr(self.ch._cipher, "weak_check") else self.ch._cipher.weak_check
        else:
            weak_check = False if not hasattr(self.ch.func, "weak_check") else self.ch.func.weak_check

        if len(self.ch.ssa["assignments"]) > MAX_ASSIGNMENTS and not weak_check:
            assert len(self.ch.nonlinear_diffs) > 0
            emp_weight = verification_impossible.fast_empirical_weight(self, verbose_lvl=verbose_lvl, filename=filename)
        else:
            pair_samples = min(2 ** 12 // (len(self.ch.ssa["assignments"]) + 1), 100)
            if verbose_lvl >= 2:
                smart_print("- checking {} -> {} with {} pair samples".format(
                    '|'.join([str(d.val) for d in input_diff]),
                    '|'.join([str(d.val) for d in output_diff]),
                    pair_samples))
            emp_weight = self.ch.empirical_weight(input_diff, output_diff, pair_samples)
            if verbose_lvl >= 2:
                smart_print("- empirical weight: {}".format(emp_weight))

        if emp_weight is not math.inf:
            msg = "The empirical weight do not match\n"
            msg += " - theoretical weight: {}\n".format(math.inf)
            msg += " - empirical weight:   {}\n".format(emp_weight)
            msg += str(self._to_dict(vrepr=verbose_lvl >= 3))
            raise search_differential.ExactWeightError(msg)
        else:
            self.emp_weight = emp_weight

    def _check_empirical_distribution_weight(self, cipher, verbose_lvl=0, filename=None, rk_dict_diffs=None):
        # similar to _empirical_distribution_weight of characteristic module
        smart_print = _get_smart_print(filename)

        input_diff = [value for var, value in self.input_diff]
        output_diff = [value for var, value in self.output_diff]

        weak_check = False if not hasattr(cipher, "weak_check") else cipher.weak_check

        if len(self.ch.ssa["assignments"]) > MAX_ASSIGNMENTS and not weak_check:
            assert len(self.ch.nonlinear_diffs) != 0
            emp_weight_dist = verification_impossible._fast_empirical_weight_distribution(
                self, cipher, rk_dict_diffs, verbose_lvl=verbose_lvl, filename=filename)
        else:
            key_samples = 2**8
            pair_samples = min(2**12 // (len(self.ch.ssa["assignments"]) + 1), 100)

            if verbose_lvl >= 2:
                smart_print("- checking {} -> {} with {} pair samples and {} key samples".format(
                    '|'.join([str(d.val) for d in input_diff]),
                    '|'.join([str(d.val) for d in output_diff]),
                    pair_samples, key_samples))
            if rk_dict_diffs is None:
                rk_output_diff = None
            else:
                assert len(rk_dict_diffs) > 0
                rk_output_diff = [value for var, value in rk_dict_diffs]
            emp_weight_dist = self.ch._empirical_weight_distribution(cipher, input_diff, output_diff, pair_samples,
                                                                     key_samples, rk_output_diff=rk_output_diff)
            if verbose_lvl >= 2:
                smart_print("- empirical weight: {}".format(emp_weight_dist))

        if any(v != math.inf for v in emp_weight_dist.keys()):
            msg = "The empirical weight do not match\n"
            msg += " - theoretical weight: {}\n".format(math.inf)
            msg += " - empirical weights: {}\n".format(emp_weight_dist)
            msg += str(self._to_dict(vrepr=verbose_lvl >= 3))
            raise search_differential.ExactWeightError(msg)
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

        input_diff = DictItem(self.input_diff)
        output_diff = DictItem(self.output_diff)

        dict_ch = {
            'input_diff': input_diff,
            'output_diff': output_diff,
        }
        if self.emp_weight is not None:
            dict_ch['emp_weight'] = DictItem(self.emp_weight)

        return dict_ch

    def __str__(self):
        return pprint.pformat(self._to_dict(), width=100, compact=True)

    def vrepr(self):
        """Return a verbose dictionary-like representation of the ID.

            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.characteristic import BvCharacteristic
            >>> from arxpy.primitives.chaskey import ChaskeyPi
            >>> from arxpy.smt.search_impossible import SearchID
            >>> ChaskeyPi.set_rounds(1)
            >>> ch = BvCharacteristic(ChaskeyPi, XorDiff, ["dv0", "dv1", "dv2", "dv3"])
            >>> search_problem = SearchID(ch)
            >>> id_found = search_problem.solve(2)
            >>> print(id_found.vrepr())  # doctest:+NORMALIZE_WHITESPACE
            {'input_diff': [[XorDiff(Variable('dv0', width=32)), XorDiff(Constant(0b00000000000000000000000000000001, width=32))],
            [XorDiff(Variable('dv1', width=32)), XorDiff(Constant(0b00000000000000000000000000000000, width=32))],
            [XorDiff(Variable('dv2', width=32)), XorDiff(Constant(0b00000000000000000000000000000000, width=32))],
            [XorDiff(Variable('dv3', width=32)), XorDiff(Constant(0b00000000000000000000000000000000, width=32))]],
            'output_diff': [[XorDiff(Variable('d7', width=32)), XorDiff(Constant(0b00000000000000000000000000000001, width=32))],
            [XorDiff(Variable('d12', width=32)), XorDiff(Constant(0b00000000000000000000000000000000, width=32))],
            [XorDiff(Variable('d13', width=32)), XorDiff(Constant(0b00000000000000000000000000000000, width=32))],
            [XorDiff(Variable('d9', width=32)), XorDiff(Constant(0b00000000000000000000000000000000, width=32))]]}

        """
        return str(self._to_dict(vrepr=True))

    def srepr(self):
        """Return a short representation of the ID."""
        input_diff = ' '.join([x.val.hex()[2:] if x.val.width >= 8 else x.val.bin()[2:] for _, x in self.input_diff])
        output_diff = ' '.join([x.val.hex()[2:] if x.val.width >= 8 else x.val.bin()[2:] for _, x in self.output_diff])
        return "{} -> {}".format(input_diff, output_diff)


class SkIDFound(IDFound):
    """Represent (non-symbolic) single-key ID found over a `SingleKeyCh`.

    See also `IDFound`.

    ::

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import SingleKeyCh
        >>> from arxpy.smt.search_impossible import SearchSkID
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_rounds(1)
        >>> ch = SingleKeyCh(Speck32, XorDiff)
        >>> search_problem = SearchSkID(ch)
        >>> id_found = search_problem.solve(2)
        >>> id_found.input_diff
        [[XorDiff(dp0), XorDiff(0x0001)], [XorDiff(dp1), XorDiff(0x0000)]]
        >>> id_found.output_diff
        [[XorDiff(dx2), XorDiff(0x0001)], [XorDiff(dx4), XorDiff(0x0000)]]
        >>> print(id_found)
        {'input_diff': [[dp0, 0x0001], [dp1, 0x0000]], 'output_diff': [[dx2, 0x0001], [dx4, 0x0000]]}

    """

    def __init__(self, ch, input_diff_found, output_diff_found):
        super().__init__(ch, input_diff_found, output_diff_found)

        self._cipher = ch._cipher

    def check_empirical_weight(self, verbose_lvl=0, filename=None):
        """Check the empirical weight is `math.inf` for many keys.

        If the empirical weight is not `math.inf` for some key,
        an `ExactWeightError` exception is raised.

        If ``filename`` is not ``None``, the output will be printed
        to the given file rather than the to stdout.

        The argument ``verbose_lvl`` can also take the values ``1`` and ``2`` for a
        more detailed output.
        """
        return self._check_empirical_distribution_weight(self._cipher, verbose_lvl, filename, rk_dict_diffs=None)


class RkIDFound(object):
    """Represent (non-symbolic) related-key ID found over a `RelatedKeyCh`.

    Attributes:
        rkch: the associated symbolic `RelatedKeyCh`
        key_id_found: a `IDFound` denoting the ID over the key schedule
        enc_id_found: a `IDFound` denoting the ID over the encryption

    ::

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import RelatedKeyCh
        >>> from arxpy.smt.search_impossible import SearchRkID
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_rounds(1)
        >>> rkch = RelatedKeyCh(Speck32, XorDiff)
        >>> search_problem = SearchRkID(rkch)
        >>> rkid_found = search_problem.solve(2)
        >>> rkid_found.key_id_found.input_diff
        [[XorDiff(dmk0), XorDiff(0x0001)]]
        >>> rkid_found.enc_id_found.input_diff
        [[XorDiff(dp0), XorDiff(0x0000)], [XorDiff(dp1), XorDiff(0x0000)]]
        >>> rkid_found.enc_id_found.output_diff
        [[XorDiff(dx2), XorDiff(0x0001)], [XorDiff(dx4), XorDiff(0x0000)]]
        >>> print(rkid_found)
        {'enc_id_found': {'input_diff': [[dp0, 0x0000], [dp1, 0x0000]],
                          'output_diff': [[dx2, 0x0001], [dx4, 0x0000]]},
         'key_id_found': {'input_diff': [[dmk0, 0x0001]], 'output_diff': []}}

    """
    def __init__(self, rkch, key_input_diff_found, enc_input_diff_found, enc_output_diff_found):
        self.rkch = rkch
        self.key_id_found = IDFound(rkch.key_schedule_ch, key_input_diff_found, [])
        self.enc_id_found = IDFound(rkch.encryption_ch, enc_input_diff_found, enc_output_diff_found)
        self._cipher = rkch._cipher

    def signature(self, ch_signature_type, return_str=False):
        return self.key_id_found.signature(ch_signature_type, return_str=return_str) + \
               self.enc_id_found.signature(ch_signature_type, return_str=return_str)

    def check_empirical_weight(self, verbose_lvl=0, filename=None):
        """Check the empirical weight is `math.inf` for many keys.

        If the empirical weight is not `math.inf` for some key,
        an `ExactWeightError` exception is raised.

        If ``filename`` is not ``None``, the output will be printed
        to the given file rather than the to stdout.

        The argument ``verbose_lvl`` can also take the values ``1`` and ``2`` for a
        more detailed output.
        """
        raise NotImplementedError("")

    def _to_dict(self, vrepr=False):
        dict_ch = {
            "key_id_found": self.key_id_found._to_dict(vrepr=vrepr),
            "enc_id_found": self.enc_id_found._to_dict(vrepr=vrepr),
        }
        return dict_ch

    def __str__(self):
        return pprint.pformat(self._to_dict(), width=100, compact=True)

    def vrepr(self):
        """Return a verbose dictionary-like representation of the ID.

        See also `IDFound.vrepr`.
        """
        return str(self._to_dict(vrepr=True))

    def srepr(self):
        """Return a short representation of the ID."""
        return "K: {} | E: {}".format(self.key_id_found.srepr(), self.enc_id_found.srepr())


class SearchID(search_differential.SearchCh):
    """Represent the problem of finding an ID over a `BvCharacteristic`.

    Args:
        ch (BvCharacteristic): a symbolic characteristic of a `BvFunction`

    See also `SearchCh`.

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import BvCharacteristic
        >>> from arxpy.smt.search_impossible import SearchID
        >>> from arxpy.primitives.chaskey import ChaskeyPi
        >>> ChaskeyPi.set_rounds(1)
        >>> ch = BvCharacteristic(ChaskeyPi, XorDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> search_problem = SearchID(ch)
        >>> search_problem.formula_size()
        142
        >>> search_problem.error()
        0
        >>> print(search_problem.hrepr(False))
        assert 0x00000000 == ((~(... << ...) ^ (d0 << 0x00000001)) & (~(... << ...) ^ (dv1 << 0x00000001)) & ((dv0 << 0x00000001) ^ d0 ^ dv0 ^ dv1))
        assert 0x00000000 == ((~(... << ...) ^ (d4 << 0x00000001)) & (~(... << ...) ^ (dv3 << 0x00000001)) & ((dv2 << 0x00000001) ^ d4 ^ dv2 ^ dv3))
        assert 0x00000000 == ((~(... << ...) ^ (d7 << 0x00000001)) & (~(... << ...) ^ ((... ^ ...) << 0x00000001)) & (((d0 <<< 16) << 0x00000001) ^ d7 ^ ... ^ ... ^ (... <<< ...)))
        assert 0x00000000 == ((~(... << ...) ^ (d10 << 0x00000001)) & (~(... << ...) ^ (d4 << 0x00000001)) & (((d0 ^ (... <<< ...)) << 0x00000001) ^ d10 ^ d4 ^ ... ^ ...))
        assert 0b0 == w
        check-unsat

    """

    def __init__(self, ch, initial_constraints=None):
        assert isinstance(ch, characteristic.BvCharacteristic)
        super().__init__(ch, der_mode=search_differential.DerMode.Valid,
                         allow_zero_input_diff=True,  # non-zero constraints not needed
                         initial_constraints=initial_constraints)

    def hrepr(self, full_repr=False):
        """Return a human readable representing of the SMT problem.

        If ``full_repr`` is False, the short string representation `srepr` is used.
        """
        representation = super().hrepr(full_repr, minimize_constraint=False)
        return representation + "\ncheck-unsat"

    def extend(self, id_found, solver_name, check, verbose_level, filename):
        """Extend the ID by appending a characteristic with probability one."""
        raise NotImplementedError("subclasses need to override this method")

    def solve(self, initial_active_bits, solver_name="btor", search_mode=SearchMode.FirstID,
              input_diff_mode=ActivationMode.Default, output_diff_mode=ActivationMode.Default,
              to_extend=False, check=False, verbose_level=0, filename=None):
        """Solve the SMT problem associated  to the search of an ID over a `BvCharacteristic`.

        Args:
             initial_active_bits(int): the initial number of active bits in the input and output
             solver_name(str): the name of the solver (according to pySMT) to be used
             search_mode(SearchMode): one of the search modes available
             input_diff_mode(ActivationMode): the diff-mode for the input difference
             output_diff_mode(ActivationMode): the diff-mode for the output difference
             check(bool): if ``True``, `IDFound.check_empirical_weight` will be called
                after an ID is found. If it is not valid, the search will continue.
             verbose_level(int): an integer between ``0`` (no verbose) and ``3`` (full verbose).
             filename(str): if not ``None``, the output will be  printed to the given file
                rather than the to stdout.

        """
        if ActivationMode.Zero not in [input_diff_mode, output_diff_mode] and initial_active_bits == 0:
            initial_active_bits = 1

        strict_shift = True if solver_name == "btor" else False  # e.g., btor_rol: width must be a power of 2
        bv2pysmt = functools.partial(types.bv2pysmt, env=self._env, strict_shift=strict_shift)

        smart_print = _get_smart_print(filename)

        if type(self) == SearchID:
            id_found_class = IDFound
            if to_extend:
                raise NotImplementedError("to_extend not supported for SearchID")
        elif type(self) == SearchSkID:
            id_found_class = SkIDFound
        else:
            raise ValueError("invalid subclass of SearchID")

        input_widths = [d.val.width for d in self.ch.input_diff]
        output_widths = [d.val.width for d, _ in self.ch.output_diff]

        def mode2max_max_active_bits(mode, list_widths):
            min_b = 1
            if mode == ActivationMode.Default:
                max_b = sum(list_widths)
            elif mode in [ActivationMode.SingleBit, ActivationMode.MSBBit]:
                max_b = len(list_widths)
            elif mode == ActivationMode.Zero:
                max_b = min_b = 0
            else:
                raise ValueError("invalid mode")
            return max_b, min_b

        max_input_bits, min_input_bits = mode2max_max_active_bits(input_diff_mode, input_widths)
        max_output_bits, min_output_bits = mode2max_max_active_bits(output_diff_mode, output_widths)
        max_active_bits = min(MAX_TOTAL_ACTIVE_BITS, max_input_bits + max_output_bits)

        with self._env.factory.Solver(name=solver_name, logic=logics.QF_BV) as solver:
            for c in self.constraints:
                solver.add_assertion(bv2pysmt(c, boolean=True))

            for active_bits in range(initial_active_bits, max_active_bits + 1):
                for input_active_bits in range(min_input_bits, max_input_bits + 1):
                    output_active_bits = active_bits - input_active_bits
                    if output_active_bits < min_output_bits or output_active_bits > max_output_bits:
                        continue

                    if verbose_level >= 1:
                        smart_print(_get_time(), "| Finding input/output diff with {}/{} active bits".format(
                            input_active_bits, output_active_bits
                        ))

                    for input_diff in generate_active_bvlists(input_widths, input_active_bits, input_diff_mode):
                        solver.push()

                        # with context.Simplification(False):
                        c = True
                        for diff, val in zip(self.ch.input_diff, input_diff):
                            c &= operation.BvComp(diff.val, val)
                        solver.add_assertion(bv2pysmt(c, boolean=True))

                        if verbose_level >= 2:
                            smart_print(_get_time(), "| Fixed input diff:  ", input_diff)

                        for output_diff in generate_active_bvlists(output_widths, output_active_bits, output_diff_mode):
                            solver.push()

                            # with context.Simplification(False):
                            c = True
                            for (_, diff), val in zip(self.ch.output_diff, output_diff):
                                c &= operation.BvComp(diff.val, val)
                            solver.add_assertion(bv2pysmt(c, boolean=True))
                            if verbose_level >= 3:
                                smart_print(_get_time(), "| Fixed output diff: ", output_diff)

                            satisfiable = solver.solve()

                            if not satisfiable:
                                id_found = id_found_class(self.ch,  [self.ch.diff_type(d) for d in input_diff],
                                                          [self.ch.diff_type(d) for d in output_diff])

                                if check:
                                    id_found.check_empirical_weight(verbose_level, filename)

                                if to_extend:
                                    self.extend(id_found, solver_name, check, verbose_level, filename)
                                else:
                                    if search_mode == SearchMode.FirstID:
                                        return id_found
                                    elif search_mode == SearchMode.AllID:
                                        if verbose_level >= 1:
                                            prefix = str(_get_time()) + " | "
                                        else:
                                            prefix = ""
                                        smart_print(prefix + "ID found:", end=" ")
                                        if verbose_level >= 2:
                                            smart_print("\n", id_found)
                                        else:
                                            smart_print(id_found.srepr())
                                        if verbose_level >= 3:
                                            smart_print(id_found.vrepr())

                            solver.pop()

                        solver.pop()
            else:
                if verbose_level >= 1:
                    smart_print(_get_time(), "| No ID found")
                return None


class SearchSkID(SearchID):
    """Represent the problem of finding an single-key ID over a `SingleKeyCh`.

    Args:
        skch (SingleKeyCh): a symbolic single-key characteristic of a `Cipher`

    See also `SearchID`.

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import SingleKeyCh
        >>> from arxpy.smt.search_impossible import SearchSkID
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_rounds(1)
        >>> ch = SingleKeyCh(Speck32, XorDiff)
        >>> search_problem = SearchSkID(ch)
        >>> search_problem.formula_size()
        35
        >>> search_problem.error()
        0
        >>> print(search_problem.hrepr(False))
        assert 0x0000 == ((~(... << ...) ^ (dx1 << 0x0001)) & (~(... << ...) ^ ((... >>> ...) << 0x0001)) & ((dp1 << 0x0001) ^ dx1 ^ dp1 ^ (... >>> ...)))
        assert 0b0 == w
        check-unsat
        >>> print(search_problem.solve(2))
        {'input_diff': [[dp0, 0x0001], [dp1, 0x0000]], 'output_diff': [[dx2, 0x0001], [dx4, 0x0000]]}

    """

    def __init__(self, skch, initial_constraints=None):
        assert isinstance(skch, characteristic.SingleKeyCh)
        super().__init__(skch, initial_constraints=initial_constraints)

    def extend(self, id_found, solver_name, check, verbose_level, filename):
        import tempfile
        from arxpy.smt import search_differential

        smart_print = _get_smart_print(filename)

        if verbose_level >= 1:
            smart_print(_get_time(), "| non-extended ID found:", id_found.srepr())

        # assert end_round is not None

        cipher = self.ch._cipher
        diff_type = self.ch.diff_type
        num_rounds_skip_and_id = cipher.rounds  # skip_round + round_id
        if hasattr(cipher.encryption, "skip_rounds"):
            list_rounds_skip = cipher.encryption.skip_rounds
            assert list_rounds_skip == list(range(len(list_rounds_skip)))
        else:
            list_rounds_skip = []
        list_rounds_id = [i for i in range(num_rounds_skip_and_id) if i not in list_rounds_skip]
        num_rounds_id = len(list_rounds_id)
        if verbose_level >= 2:
            smart_print("- skip-rounds / rounds indices of ID ({} / {}): {} / {}".format(
                len(list_rounds_skip), len(list_rounds_id), list_rounds_skip, list_rounds_id))

        max_round_found = 0

        for num_rounds_backwards in range(0, len(list_rounds_skip) + 1):
            num_rounds_before_backwards = len(list_rounds_skip) - num_rounds_backwards
            if verbose_level >= 2:
                smart_print("- extending {} round backwards:".format(num_rounds_backwards))
            cipher.set_skip_rounds([i for i in range(num_rounds_before_backwards)] + list_rounds_id)
            if verbose_level >= 2:
                smart_print(" - current skip rounds indices of SkCh ({}): {}".format(
                    len(cipher.encryption.skip_rounds), cipher.encryption.skip_rounds))

            # # (end_round - num_rounds_skip_and_id) is the maximum number of num_round_forwards
            # if num_rounds_backwards + num_rounds_id + (end_round - num_rounds_skip_and_id) < max_round_found:
            #     if verbose_level >= 2:
            #         smart_print(
            #             " - not enough rounds {}b,{}ID,{}f to find an extended id with at least {} rounds".format(
            #                 num_rounds_backwards, num_rounds_id, (end_round - num_rounds_skip_and_id), max_round_found
            #             ))
            #     continue

            if verbose_level >= 2:
                smart_print(" - setting ID-input/output-diff to SkCh round input index: {} / {}".format(
                    list_rounds_id[0], list_rounds_id[-1] + 1))
            skch_file = tempfile.NamedTemporaryFile()
            start_round = max(num_rounds_skip_and_id + 1, num_rounds_before_backwards + max_round_found)
            if verbose_level >= 2:
                smart_print(" - searching skch with start_round {} >= (nr_before {} + {} max_r_found)".format(
                    start_round, num_rounds_before_backwards, max_round_found))

            skch_found = search_differential.round_based_search_SkCh(
                cipher=cipher, diff_type=diff_type, initial_weight=0, solver_name=solver_name,
                start_round=start_round, end_round=2**32,
                der_mode=search_differential.DerMode.ProbabilityOne,
                search_mode=search_differential.SkChSearchMode.FirstCh,
                check=check, verbose_level=max(verbose_level - 1, 0), filename=skch_file.name,
                fix_round_inputs=[[[value.val for var, value in id_found.input_diff], list_rounds_id[0]],
                                  [[value.val for var, value in id_found.output_diff], list_rounds_id[-1] + 1]],
                return_best=True
            )

            if skch_found is not None:
                num_rounds_beforeback_and_extendedid, skch_found = skch_found
                num_rounds_extended_id = num_rounds_beforeback_and_extendedid - num_rounds_before_backwards
                num_rounds_forwards = num_rounds_extended_id - num_rounds_backwards - num_rounds_id
                assert num_rounds_extended_id >= max_round_found
                max_round_found = max(max_round_found, num_rounds_extended_id)
                # if verbose_level >= 2 or num_rounds_extended_id >= max_round_found:
                if verbose_level >= 0:
                    smart_print("\tFound {}-r ({}-backward/{}-forward) | ID: {} | SkCh: {}".format(
                        num_rounds_extended_id, num_rounds_backwards, num_rounds_forwards, id_found.srepr(), skch_found.srepr()
                    ))
                if verbose_level >= 3:
                    for line in skch_file:
                        if line == b'\n':
                            continue
                        smart_print("\t\t", line.decode(), end="")
            elif num_rounds_id >= max_round_found:
                num_rounds_extended_id = num_rounds_id
                max_round_found = max(max_round_found, num_rounds_extended_id)
                if verbose_level >= 0:
                    smart_print(
                        "\tFound {}-r  (0-backward/0-forward) ID: {}".format(num_rounds_extended_id, id_found.srepr()))
            elif verbose_level >= 3:
                smart_print(" - not possible to extend {} round backwards to find an ID with at least {} rounds".format(
                    num_rounds_backwards, max_round_found))

        cipher.rounds = num_rounds_skip_and_id
        cipher.set_skip_rounds(list_rounds_skip)


class SearchRkID(search_differential.SearchRkCh):
    """Represent the problem of finding a related-key ID over a `RelatedKeyCh`.

    Args:
        rkch (RelatedKeyCh): a symbolic related-key characteristic of a `Cipher`

    See also `SearchID`.

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import RelatedKeyCh
        >>> from arxpy.smt.search_impossible import SearchRkID
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_rounds(1)
        >>> ch = RelatedKeyCh(Speck32, XorDiff)
        >>> search_problem = SearchRkID(ch)
        >>> search_problem.formula_size()
        38
        >>> print(search_problem.hrepr(False))
        assert 0b0 == wk
        assert 0x0000 == ((~(... << ...) ^ (dx1 << 0x0001)) & (~(... << ...) ^ ((... >>> ...) << 0x0001)) & ((dp1 << 0x0001) ^ dx1 ^ dp1 ^ (... >>> ...)))
        assert 0b0 == w
        check-unsat
        >>> print(search_problem.solve(2))
        {'enc_id_found': {'input_diff': [[dp0, 0x0000], [dp1, 0x0000]],
                          'output_diff': [[dx2, 0x0001], [dx4, 0x0000]]},
         'key_id_found': {'input_diff': [[dmk0, 0x0001]], 'output_diff': []}}

    """

    def __init__(self, rkch, initial_key_constraints=None, initial_enc_constraints=None):
        assert isinstance(rkch, characteristic.RelatedKeyCh)
        super().__init__(rkch, key_der_mode=search_differential.DerMode.Valid,
                         enc_der_mode=search_differential.DerMode.Valid,
                         allow_zero_key_input_diff=True, allow_zero_enc_input_diff=True,
                         initial_key_constraints=initial_key_constraints,
                         initial_enc_constraints=initial_enc_constraints)

    def hrepr(self, full_repr=False):
        """Return a human readable representing of the SMT problem.

        See  also `SearchID.hrepr`.
        """
        representation = super().hrepr(full_repr, minimize_constraint=False)
        return representation + "\ncheck-unsat"

    def extend(self, rkid_found, solver_name, check, verbose_level, filename):
        import tempfile
        from arxpy.smt import search_differential

        smart_print = _get_smart_print(filename)

        if verbose_level >= 2:
            smart_print(_get_time(), "| non-extended RkID found:", rkid_found.srepr())

        # assert end_round is not None

        cipher = self.rkch._cipher
        diff_type = self.rkch.diff_type
        num_rounds_skip_and_id = cipher.rounds  # skip_round + round_id
        if hasattr(cipher.encryption, "skip_rounds"):
            list_rounds_skip = cipher.encryption.skip_rounds
            assert list_rounds_skip == list(range(len(list_rounds_skip)))
        else:
            list_rounds_skip = []
        list_rounds_id = [i for i in range(num_rounds_skip_and_id) if i not in list_rounds_skip]
        num_rounds_id = len(list_rounds_id)
        if verbose_level >= 2:
            smart_print("- skip-rounds / rounds indices of ID ({} / {}): {} / {}".format(
                len(list_rounds_skip), len(list_rounds_id), list_rounds_skip, list_rounds_id))

        max_round_found = 0

        for num_rounds_backwards in range(0, len(list_rounds_skip) + 1):
            num_rounds_before_backwards = len(list_rounds_skip) - num_rounds_backwards
            if verbose_level >= 2:
                smart_print("- extending {} round backwards:".format(num_rounds_backwards))
            cipher.set_skip_rounds([i for i in range(num_rounds_before_backwards)] + list_rounds_id)
            if verbose_level >= 2:
                smart_print(" - current skip rounds indices of RkCh ({}): {}".format(
                    len(cipher.encryption.skip_rounds), cipher.encryption.skip_rounds))

            # # (end_round - num_rounds_skip_and_id) is the maximum number of num_round_forwards
            # if num_rounds_backwards + num_rounds_id + (end_round - num_rounds_skip_and_id) < max_round_found:
            #     if verbose_level >= 2:
            #         smart_print(
            #             " - not enough rounds {}b,{}ID,{}f to find an extended id with at least {} rounds".format(
            #                 num_rounds_backwards, num_rounds_id, (end_round - num_rounds_skip_and_id), max_round_found
            #             ))
            #     continue

            if verbose_level >= 2:
                smart_print(" - setting ID-enc input/output diff to RkCh round input index: {} / {}".format(
                    list_rounds_id[0], list_rounds_id[-1] + 1))
            rkch_file = tempfile.NamedTemporaryFile()
            start_round = max(num_rounds_skip_and_id + 1, num_rounds_before_backwards + max_round_found)
            if verbose_level >= 2:
                smart_print(" - searching rkch with start_round {} >= (nr_before {} + {} max_r_found)".format(
                    start_round, num_rounds_before_backwards, max_round_found))

            rkch_best = search_differential.round_based_search_RkCh(
                cipher=cipher, diff_type=diff_type, initial_ew=0, initial_kw=0, solver_name=solver_name,
                start_round=start_round, end_round=2**32,
                key_der_mode=search_differential.DerMode.ProbabilityOne,
                enc_der_mode=search_differential.DerMode.ProbabilityOne,
                search_mode=search_differential.RkChSearchMode.FirstMinSum,
                allow_zero_enc_input_diff=True, allow_zero_key_input_diff=True,
                check=check, verbose_level=max(verbose_level - 1, 0), filename=rkch_file.name,
                fix_key_input_diff=[value.val for var, value in rkid_found.key_id_found.input_diff],
                fix_enc_round_inputs=[
                    [[value.val for var, value in rkid_found.enc_id_found.input_diff], list_rounds_id[0]],
                    [[value.val for var, value in rkid_found.enc_id_found.output_diff], list_rounds_id[-1] + 1],
                ],
                return_best=True
            )

            if rkch_best is not None:
                num_rounds_beforeback_and_extendedid, rkch_found = rkch_best
                num_rounds_extended_id = num_rounds_beforeback_and_extendedid - num_rounds_before_backwards
                num_rounds_forwards = num_rounds_extended_id - num_rounds_backwards - num_rounds_id
                assert num_rounds_extended_id >= max_round_found
                max_round_found = max(max_round_found, num_rounds_extended_id)
                # if verbose_level >= 2 or num_rounds_extended_id >= max_round_found:
                if verbose_level >= 0:
                    smart_print("\tFound {}-r ({}-backward/{}-forward) | ID: {} | RKCH: {}".format(
                        num_rounds_extended_id, num_rounds_backwards, num_rounds_forwards, rkid_found.srepr(),
                        rkch_found.srepr()
                    ))
                if verbose_level >= 3:
                    for line in rkch_file:
                        if line == b'\n':
                            continue
                        smart_print("\t\t", line.decode(), end="")
            elif num_rounds_id >= max_round_found:
                num_rounds_extended_id = num_rounds_id
                max_round_found = max(max_round_found, num_rounds_extended_id)
                if verbose_level >= 0:
                    smart_print(
                        "\tFound {}-r  (0-backward/0-forward) ID: {}".format(num_rounds_extended_id, rkid_found.srepr()))
            elif verbose_level >= 3:
                smart_print(" - not possible to extend {} round backwards to find an ID with at least {} rounds".format(
                    num_rounds_backwards, max_round_found))

        cipher.rounds = num_rounds_skip_and_id
        cipher.set_skip_rounds(list_rounds_skip)

    def solve(self, initial_active_bits, solver_name="btor", search_mode=SearchMode.FirstID,
              key_input_diff_mode = ActivationMode.Default,
              enc_input_diff_mode=ActivationMode.Default,  enc_output_diff_mode=ActivationMode.Default,
              to_extend=False, check=False, verbose_level=0, filename=None):
        """Solve the SMT problem associated to the search of an ID over a `RelatedKeyCh`.

        Args:
             initial_active_bits(int): the initial number of active bits in the input and output
             solver_name(str): the name of the solver (according to pySMT) to be used
             search_mode(SearchMode): one of the search modes available
             key_input_diff_mode(ActivationMode): the diff-mode for the master key difference
             enc_input_diff_mode(ActivationMode): the diff-mode for the plaintext difference
             enc_output_diff_mode(ActivationMode): the diff-mode for the ciphertext difference
             check(bool): if ``True``, `IDFound.check_empirical_weight` will be called
                after an ID is found. If it is not valid, the search will continue.
             verbose_level(int): an integer between ``0`` (no verbose) and ``3`` (full verbose).
             filename(str): if not ``None``, the output will be  printed to the given file
                rather than the to stdout.

        """
        if ActivationMode.Zero not in [key_input_diff_mode, enc_input_diff_mode, enc_output_diff_mode] and \
                initial_active_bits == 0:
            initial_active_bits = 1

        strict_shift = True if solver_name == "btor" else False  # e.g., btor_rol: width must be a power of 2
        bv2pysmt = functools.partial(types.bv2pysmt, env=self._env, strict_shift=strict_shift)

        smart_print = _get_smart_print(filename)

        kp = self.key_schedule_problem
        ep = self.encryption_problem

        key_input_widths = [d.val.width for d in kp.ch.input_diff]
        enc_input_widths = [d.val.width for d in ep.ch.input_diff]
        enc_output_widths = [d.val.width for d, _ in ep.ch.output_diff]

        def mode2max_min_active_bits(mode, list_widths):
            min_b = 1
            if mode == ActivationMode.Default:
                max_b = sum(list_widths)
            elif mode in [ActivationMode.SingleBit, ActivationMode.MSBBit]:
                max_b = len(list_widths)
            elif mode == ActivationMode.Zero:
                max_b = min_b = 0
            else:
                raise ValueError("invalid mode")
            return min(max_b, MAX_TOTAL_ACTIVE_BITS), min_b

        max_key_input_bits, min_key_input_bits = mode2max_min_active_bits(key_input_diff_mode, key_input_widths)
        max_enc_input_bits, _ = mode2max_min_active_bits(enc_input_diff_mode, enc_input_widths)
        max_enc_output_bits, _ = mode2max_min_active_bits(enc_output_diff_mode, enc_output_widths)
        min_enc_input_bits = min_enc_output_bits = 0
        max_active_bits = min(MAX_TOTAL_ACTIVE_BITS, max_key_input_bits + max_enc_input_bits + max_enc_output_bits)

        with self._env.factory.Solver(name=solver_name, logic=logics.QF_BV) as solver:
            for c in kp.constraints:
                solver.add_assertion(bv2pysmt(c, boolean=True))

            for c in ep.constraints:
                solver.add_assertion(bv2pysmt(c, boolean=True))

            for active_bits in range(initial_active_bits, max_active_bits + 1):
                for key_input_active_bits in range(min_key_input_bits, max_key_input_bits + 1):
                    for enc_input_active_bits in range(min_enc_input_bits, max_enc_input_bits + 1):
                        enc_output_active_bits = active_bits - (key_input_active_bits + enc_input_active_bits)
                        if enc_output_active_bits < min_enc_output_bits or enc_output_active_bits > max_enc_output_bits:
                            continue

                        if verbose_level >= 1:
                            smart_print(_get_time(), "| Finding key-input/enc-input/enc-output diff with {}/{}/{} active bits".format(
                                key_input_active_bits, enc_input_active_bits, enc_output_active_bits
                            ))

                        for key_input_diff in generate_active_bvlists(key_input_widths, key_input_active_bits, key_input_diff_mode):
                            solver.push()
                            c = True
                            for diff, val in zip(kp.ch.input_diff, key_input_diff):
                                c &= operation.BvComp(diff.val, val)
                            solver.add_assertion(bv2pysmt(c, boolean=True))
                            if verbose_level >= 2:
                                smart_print(_get_time(), "| Fixed key input diff:  ", key_input_diff)

                            for enc_input_diff in generate_active_bvlists(enc_input_widths, enc_input_active_bits, enc_input_diff_mode):
                                solver.push()
                                c = True
                                for diff, val in zip(ep.ch.input_diff, enc_input_diff):
                                    c &= operation.BvComp(diff.val, val)
                                solver.add_assertion(bv2pysmt(c, boolean=True))
                                if verbose_level >= 3:
                                    smart_print(_get_time(), "| Fixed enc input diff:  ", enc_input_diff)

                                for enc_output_diff in generate_active_bvlists(enc_output_widths, enc_output_active_bits, enc_output_diff_mode):
                                    solver.push()
                                    c = True
                                    for (_, diff), val in zip(ep.ch.output_diff, enc_output_diff):
                                        c &= operation.BvComp(diff.val, val)
                                    solver.add_assertion(bv2pysmt(c, boolean=True))
                                    if verbose_level >= 3:
                                        smart_print(_get_time(), "| Fixed enc output diff: ", enc_output_diff)

                                    satisfiable = solver.solve()

                                    if not satisfiable:
                                        id_found = RkIDFound(self.rkch, [kp.ch.diff_type(d) for d in key_input_diff],
                                                             [ep.ch.diff_type(d) for d in enc_input_diff],
                                                             [ep.ch.diff_type(d) for d in enc_output_diff])

                                        if check:
                                            id_found.check_empirical_weight(verbose_level, filename)

                                        if to_extend:
                                            id_found = self.extend(id_found, solver_name, check, verbose_level, filename)
                                            if check:
                                                id_found.check_empirical_weight(verbose_level, filename)

                                        if search_mode == SearchMode.FirstID:
                                            return id_found
                                        elif search_mode == SearchMode.AllID:
                                            if verbose_level >= 1:
                                                prefix = str(_get_time()) + " | "
                                            else:
                                                prefix = ""
                                            smart_print(prefix + "ID found:", end=" ")
                                            if verbose_level >= 2:
                                                smart_print("\n", id_found)
                                            else:
                                                smart_print(id_found.srepr())
                                            if verbose_level >= 3:
                                                smart_print(id_found.vrepr())

                                    solver.pop()

                                solver.pop()

                            solver.pop()

            if verbose_level >= 1:
                smart_print(_get_time(), "| No ID found")
            return None


class SearchLinearRkID(object):
    def __init__(self, cipher, diff_type):
        self.cipher = cipher
        self.diff_type = diff_type

    @classmethod
    def get_solver_id(cls, problem, solver_name):
        strict_shift = True if solver_name == "btor" else False  # e.g., btor_rol: width must be a power of 2
        bv2pysmt = functools.partial(types.bv2pysmt, env=problem._env, strict_shift=strict_shift)
        solver = problem._env.factory.Solver(name=solver_name, logic=logics.QF_BV)
        for c in problem.key_schedule_problem.constraints:
            solver.add_assertion(bv2pysmt(c, boolean=True))
        for c in problem.encryption_problem.constraints:
            solver.add_assertion(bv2pysmt(c, boolean=True))
        return solver

    @classmethod
    def is_unsat_problem_id(cls, key_input_diff, enc_input_diff, enc_output_diff, problem, solver, solver_name):
        strict_shift = True if solver_name == "btor" else False  # e.g., btor_rol: width must be a power of 2
        bv2pysmt = functools.partial(types.bv2pysmt, env=problem._env, strict_shift=strict_shift)
        kp = problem.key_schedule_problem
        ep = problem.encryption_problem

        solver.push()

        c = True
        for diff, val in zip(kp.ch.input_diff, key_input_diff):
            c &= operation.BvComp(diff.val, val)
        solver.add_assertion(bv2pysmt(c, boolean=True))
        c = True
        for diff, val in zip(ep.ch.input_diff, enc_input_diff):
            c &= operation.BvComp(diff.val, val)
        solver.add_assertion(bv2pysmt(c, boolean=True))
        c = True
        for (_, diff), val in zip(ep.ch.output_diff, enc_output_diff):
            c &= operation.BvComp(diff.val, val)
        solver.add_assertion(bv2pysmt(c, boolean=True))

        satisfiable = solver.solve()

        solver.pop()

        return not satisfiable

    def solve(self, num_skip_first_r, num_back_r, num_id_r, num_forw_r, search_mode=SearchMode.FirstID,
              solver_name="btor", check=False, verbose_level=0, filename=None):
        import tempfile
        from arxpy.smt import search_differential

        total_r = num_skip_first_r + num_back_r + num_id_r + num_forw_r

        cipher = self.cipher
        smart_print = _get_smart_print(filename)

        class KSLinear(cipher.key_schedule):
            rounds = None

        class ELinear(cipher.encryption):
            rounds, skip_rounds = None, None

        class CipherLinear(cipher):
            key_schedule, encryption, rounds = KSLinear, ELinear, None

        class KSID(cipher.key_schedule):
            rounds = None

        class EID(cipher.encryption):
            rounds, skip_rounds = None, None

        class CipherID(cipher):
            key_schedule, encryption, rounds = KSID, EID, None

        class KSCheckFullID(cipher.key_schedule):
            rounds = None

        class ECheckFullID(cipher.encryption):
            rounds, skip_rounds = None, None

        class CipherCheckFullID(cipher):
            key_schedule, encryption, rounds = KSCheckFullID, ECheckFullID, None

        list_rounds_skip_first = [i for i in range(num_skip_first_r)]

        # settings RkCh
        list_rounds_id = [i for i in range(num_skip_first_r + num_back_r, num_skip_first_r + num_back_r + num_id_r)]  # non-full ID
        CipherLinear.set_rounds(total_r)
        CipherLinear.set_skip_rounds(list_rounds_skip_first + list_rounds_id)  # only backwards and forwards rounds
        rkch_linear = characteristic.RelatedKeyCh(CipherLinear, self.diff_type)
        problem_linear = search_differential.SearchRkCh(
            rkch_linear, key_der_mode=search_differential.DerMode.ProbabilityOne,
            enc_der_mode=search_differential.DerMode.ProbabilityOne,
            allow_zero_key_input_diff=False, allow_zero_enc_input_diff=True,
        )

        # settings RkID
        list_rounds_back = [i for i in range(num_skip_first_r, num_skip_first_r + num_back_r)]
        CipherID.set_rounds(num_skip_first_r + num_back_r + num_id_r)
        CipherID.set_skip_rounds(list_rounds_skip_first + list_rounds_back)  # only non-full ID rounds
        rkch_id = characteristic.RelatedKeyCh(CipherID, self.diff_type)
        problem_id = search_differential.SearchRkCh(
            rkch_id, key_der_mode=search_differential.DerMode.Valid,
            enc_der_mode=search_differential.DerMode.Valid,
            allow_zero_key_input_diff=True, allow_zero_enc_input_diff=True,
        )
        solver_id = self.get_solver_id(problem_id, solver_name)

        # settings CheckFullID
        CipherCheckFullID.set_skip_rounds(list_rounds_skip_first)

        if verbose_level >= 3:
            smart_print("rkch_linear key_schedule/encryption ssa:")
            smart_print("\t", rkch_linear.key_schedule_ch.ssa)
            smart_print("\t", rkch_linear.encryption_ch.ssa)
            smart_print("rkch_id key_schedule/encryption ssa:")
            smart_print("\t", rkch_id.key_schedule_ch.ssa)
            smart_print("\t", rkch_id.encryption_ch.ssa)
            smart_print()

        # ensure that CipherLinear and CipherID do not share class attributes
        if CipherLinear.rounds == CipherID.rounds or CipherLinear.encryption.skip_rounds == CipherID.encryption.skip_rounds:
            raise ValueError("rounds {} == {} or skip_rounds {} == {}".format(
                CipherLinear.rounds, CipherID.rounds, CipherLinear.encryption.skip_rounds, CipherID.encryption.skip_rounds
            ))
        if rkch_linear.encryption_ch.func.rounds == rkch_id.encryption_ch.func.rounds or \
                rkch_linear.encryption_ch.func.skip_rounds == rkch_id.encryption_ch.func.skip_rounds:
            raise ValueError("rounds {} == {} or skip_rounds {} == {}".format(
                CipherLinear.rounds, CipherID.rounds, CipherLinear.encryption.skip_rounds, CipherID.encryption.skip_rounds
            ))

        # Step 1 - Find a probability 1 RkCh over the cipher

        if verbose_level >= 1:
            smart_print("Finding RkCH over {}-r with {}s-{}b-{}id-{}f".format(
                total_r - num_skip_first_r, num_skip_first_r, num_back_r, num_id_r, num_forw_r))

        rkch_file = tempfile.NamedTemporaryFile()
        for rkch_found in problem_linear.solve(0, 0, solver_name=solver_name,
                                               search_mode=search_differential.RkChSearchMode.AllValid, return_generator=True,
                                               check=False, verbose_level=max(verbose_level - 2, 0), filename=rkch_file.name):
            if verbose_level >= 2:
                for line in rkch_file:
                    if line == b'\n': continue
                    smart_print("\t", line.decode(), end="")

            key_id_input_diff = [diff.val for var, diff in rkch_found.key_ch_found.input_diff]
            enc_id_input_diff = [diff.val for diff in rkch_found.enc_ch_found.round_inputs[num_skip_first_r + num_back_r]]
            enc_id_output_diff = [diff.val for diff in rkch_found.enc_ch_found.round_inputs[num_skip_first_r + num_back_r + num_id_r]]

            if verbose_level >= 1:
                smart_print(" - {} | Found RkCh: {}".format(_get_time(), rkch_found.srepr()))
                if verbose_level >= 2:
                    smart_print("\t", "key_input_diff:", key_id_input_diff)
                    smart_print("\t", "enc_id_input_diff (input round {}): {}".format(num_skip_first_r + num_back_r, enc_id_input_diff))
                    smart_print("\t", "enc_id_output_diff (input round {}): {}".format(num_skip_first_r + num_back_r + num_id_r, enc_id_output_diff))

            # Step 2 - Find an ID over the inner part of the cipher using the differences found

            if verbose_level >= 2:
                smart_print("  - Finding RkID over {}-r ({} to {}) with skipping {}".format(
                    num_id_r, num_skip_first_r + num_back_r, num_skip_first_r + num_back_r + num_id_r,
                    list_rounds_skip_first + list_rounds_back
                ))

            unsat = self.is_unsat_problem_id(key_id_input_diff, enc_id_input_diff, enc_id_output_diff,
                                             problem_id, solver_id, solver_name)

            if unsat:
                id_found = RkIDFound(
                    rkch_id, [problem_id.key_schedule_problem.ch.diff_type(d) for d in key_id_input_diff],
                    [problem_id.encryption_problem.ch.diff_type(d) for d in enc_id_input_diff],
                    [problem_id.encryption_problem.ch.diff_type(d) for d in enc_id_output_diff])

                if verbose_level >= 1:
                    smart_print("  - {} | Found ID: {}".format(_get_time(), id_found.srepr()))
                if verbose_level >= 2:
                    smart_print(id_found)

                if check:
                    fix_enc_input_diff = [diff.val for var, diff in rkch_found.enc_ch_found.input_diff]
                    fix_enc_output_diff = [diff.val for var, diff in rkch_found.enc_ch_found.output_diff]
                    rkch_file_check = tempfile.NamedTemporaryFile()
                    rkch_found_check = search_differential.round_based_search_RkCh(
                        cipher=CipherCheckFullID, diff_type=self.diff_type,
                        initial_kw=0, initial_ew=0, solver_name=solver_name,
                        start_round=total_r, end_round=total_r,
                        key_der_mode=search_differential.DerMode.Valid,
                        enc_der_mode=search_differential.DerMode.Valid,
                        search_mode=search_differential.RkChSearchMode.FirstMinSum,
                        allow_zero_enc_input_diff=True,
                        check=False, verbose_level=max(verbose_level - 2, 0), filename=rkch_file_check.name,
                        fix_key_input_diff=key_id_input_diff,
                        fix_enc_input_diff=fix_enc_input_diff,
                        fix_enc_output_diff=fix_enc_output_diff,
                        return_best=True
                    )
                    if rkch_found_check is not None or verbose_level >= 2:
                        for line in rkch_file_check:
                            if line == b'\n': continue
                            smart_print("\t", line.decode(), end="")
                    if rkch_found_check is not None or verbose_level >= 1:
                        smart_print("\t", "RkCh found with same input/output diff as full ID found:", rkch_found_check)
                    if rkch_found_check is not None:
                        raise ValueError("Error: found RkCh with same input/output diff as full ID found")

                if search_mode == SearchMode.FirstID:
                    return id_found, rkch_found
            else:
                if verbose_level >= 2:
                    smart_print("\t", "No ID found")

        solver_id.exit()


def round_based_search_ID(cipher, diff_type, initial_active_bits, solver_name,
                          start_round, end_round, search_mode, input_diff_mode, output_diff_mode,
                          check, verbose_level, filename, to_extend=False):
    """Find single-key impossible differentials over multiple rounds.

    Args:
        cipher(Cipher): an (iterated) cipher
        diff_type(Difference): a type of difference
        initial_active_bits(int): the initial number of active bits for starting the iterative search
        solver_name(str): the name of the solver (according to pySMT) to be used
        start_round(int): the minimum number of rounds to consider
        end_round(int): the maximum number of rounds to consider
        search_mode(SearchMode): one of the search modes available
        input_diff_mode:(ActivationMode): the diff-mode for the input difference
        output_diff_mode:(ActivationMode): the diff-mode for the output difference
        check(bool): if ``True``, `check_empirical_weight` will be called after an ID is found.
        verbose_level(int): an integer between ``0`` (no verbose) and ``3`` (full verbose).
        filename(str): if not ``None``, the output will be  printed to the given file
            rather than the to stdout.

        >>> from arxpy.differential.difference import XorDiff, RXDiff
        >>> from arxpy.smt.search_impossible import round_based_search_ID, KeySetting, SearchMode, ActivationMode
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> round_based_search_ID(Speck32, XorDiff, 2, "btor", 1, 2, SearchMode.FirstID,
        ...                       ActivationMode.Default, ActivationMode.Default, False, 0, None)  # doctest:+ELLIPSIS
        Num rounds: 1
        ID found:
        0001 0000 -> 0001 0000
        <BLANKLINE>
        Num rounds: 2
        ID found:
        0001 0000 -> 0001 0000

    """
    assert start_round <= end_round
    assert search_mode in [SearchMode.FirstID, SearchMode.AllID]
    assert verbose_level >= 0

    smart_print = _get_smart_print(filename)

    if verbose_level >= 1:
        smart_print(cipher.__name__, "Impossible Differential Single-Key Search\n")
        smart_print("Parameters:")
        smart_print("\tcipher:", cipher.__name__)
        smart_print("\tdiff_type:", diff_type.__name__)
        smart_print("\tinitial_active_bits:", initial_active_bits)
        smart_print("\tsolver_name:", solver_name)
        smart_print("\tstart:", start_round)
        smart_print("\tend:", end_round)
        smart_print("\tsearch_mode:", search_mode)
        smart_print("\tinput_diff_mode:", input_diff_mode)
        smart_print("\toutput_diff_mode:", output_diff_mode)
        smart_print("\tcheck:", check)
        smart_print("\tverbose_level:", verbose_level)
        smart_print("\tfilename:", filename)
        if to_extend:
            smart_print("\tto_extend:", to_extend)
        if hasattr(cipher.encryption, "skip_rounds"):
            smart_print("\tskip_rounds ({}): {}".format(len(cipher.encryption.skip_rounds), cipher.encryption.skip_rounds))
        smart_print()

    if to_extend and not hasattr(cipher, "set_skip_rounds"):
        raise ValueError("cipher does not support to_extend")

    for num_rounds in range(start_round, end_round + 1):
        cipher.set_rounds(num_rounds)

        if verbose_level >= 0:
            if num_rounds != start_round:
                smart_print()
            if hasattr(cipher.encryption, "skip_rounds"):
                num_rounds_id = num_rounds - len(cipher.encryption.skip_rounds)
                smart_print("Num rounds: {} ({} skipped)".format(num_rounds_id, len(cipher.encryption.skip_rounds)))
            else:
                smart_print("Num rounds:", num_rounds)

        ch = characteristic.SingleKeyCh(cipher, diff_type)

        if verbose_level >= 2:
            smart_print("Characteristic:")
            smart_print(ch)

        problem = SearchSkID(skch=ch)

        if verbose_level >= 2:
            smart_print("SMT problem (size {}):".format(problem.formula_size()))
            smart_print(problem.hrepr(full_repr=verbose_level >= 3))

        if verbose_level >= 1:
            prefix = str(_get_time()) + " | "
            smart_print()
        else:
            prefix = ""

        id_found = problem.solve(initial_active_bits, solver_name=solver_name, search_mode=search_mode,
                                 input_diff_mode=input_diff_mode, output_diff_mode=output_diff_mode,
                                 to_extend=to_extend, check=check, verbose_level=verbose_level, filename=filename)

        # if no r-round impossible ch exists with A active bits
        # that does NOT imply that no (r+1)-round imp ch exists with A (or less) active bits
        if search_mode == SearchMode.FirstID:
            if id_found is None:
                if verbose_level >= 0:
                    smart_print(prefix + "No ID found")
            else:
                if verbose_level >= 0:
                    smart_print(prefix + "ID found:")
                    if verbose_level == 0:
                        smart_print(id_found.srepr())
                    else:
                        smart_print(id_found)
                    if verbose_level >= 3:
                        smart_print(id_found.vrepr())


def round_based_search_RkID(cipher, diff_type, initial_active_bits, solver_name,
                            start_round, end_round, search_mode,
                            key_input_diff_mode, enc_input_diff_mode, enc_output_diff_mode,
                            check, verbose_level, filename, to_extend=False):
    """Find related-key impossible differentials over multiple rounds.

    Args:
        cipher(Cipher): an (iterated) cipher
        diff_type(Difference): a type of difference
        initial_active_bits(int): the initial number of active bits for starting the iterative search
        solver_name(str): the name of the solver (according to pySMT) to be used
        start_round(int): the minimum number of rounds to consider
        end_round(int): the maximum number of rounds to consider
        search_mode(SearchMode): one of the search modes available
        key_input_diff_mode:(ActivationMode): the diff-mode for the master key difference
        enc_input_diff_mode:(ActivationMode): the diff-mode for the plaintext difference
        enc_output_diff_mode:(ActivationMode): the diff-mode for the ciphertext difference
        check(bool): if ``True``, `check_empirical_weight` will be called after an ID is found.
        verbose_level(int): an integer between ``0`` (no verbose) and ``3`` (full verbose).
        filename(str): if not ``None``, the output will be  printed to the given file
            rather than the to stdout.

        >>> from arxpy.differential.difference import XorDiff, RXDiff
        >>> from arxpy.smt.search_impossible import round_based_search_RkID, KeySetting, SearchMode, ActivationMode
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> round_based_search_RkID(Speck32, XorDiff, 2, "btor", 1, 2, SearchMode.FirstID, ActivationMode.Default,
        ...                         ActivationMode.Default, ActivationMode.Default, False, 0, None)  # doctest:+ELLIPSIS
        Num rounds: 1
        ID found:
        K: 0001 ->  | E: 0000 0000 -> 0001 0000
        <BLANKLINE>
        Num rounds: 2
        ID found:
        K: 0001 0000 ->  | E: 0000 0000 -> 0001 0000

    """
    assert start_round <= end_round
    assert search_mode in [SearchMode.FirstID, SearchMode.AllID]
    assert verbose_level >= 0

    smart_print = _get_smart_print(filename)

    if verbose_level >= 1:
        smart_print(cipher.__name__, "Impossible Differential Related-Key Search\n")
        smart_print("Parameters:")
        smart_print("\tcipher:", cipher.__name__)
        smart_print("\tdiff_type:", diff_type.__name__)
        smart_print("\tinitial_active_bits:", initial_active_bits)
        smart_print("\tsolver_name:", solver_name)
        smart_print("\tstart:", start_round)
        smart_print("\tend:", end_round)
        smart_print("\tsearch_mode:", search_mode)
        smart_print("\tkey_input_diff_mode:", key_input_diff_mode)
        smart_print("\tenc_input_diff_mode:", enc_input_diff_mode)
        smart_print("\tenc_output_diff_mode:", enc_output_diff_mode)
        smart_print("\tcheck:", check)
        smart_print("\tverbose_level:", verbose_level)
        smart_print("\tfilename:", filename)
        if to_extend:
            smart_print("\tto_extend:", to_extend)
        if hasattr(cipher.encryption, "skip_rounds"):
            smart_print("\tencryption skip_rounds ({}): {}".format(
                len(cipher.encryption.skip_rounds), cipher.encryption.skip_rounds))
        if hasattr(cipher.key_schedule, "skip_rounds"):
            smart_print("\tkey_schedule skip_rounds ({}): {}".format(
                len(cipher.key_schedule.skip_rounds), cipher.key_schedule.skip_rounds))
        smart_print()

    if to_extend and not hasattr(cipher, "set_skip_rounds"):
        raise ValueError("cipher does not support to_extend")

    for num_rounds in range(start_round, end_round + 1):
        cipher.set_rounds(num_rounds)

        if verbose_level >= 0:
            if num_rounds != start_round:
                smart_print()
            if hasattr(cipher.encryption, "skip_rounds"):
                num_rounds_id = num_rounds - len(cipher.encryption.skip_rounds)
                smart_print("Num rounds: {} ({} skipped)".format(num_rounds_id, len(cipher.encryption.skip_rounds)))
            else:
                smart_print("Num rounds:", num_rounds)

        ch = characteristic.RelatedKeyCh(cipher, diff_type)

        if verbose_level >= 2:
            smart_print("Characteristic:")
            smart_print(ch)

        problem = SearchRkID(rkch=ch)

        if verbose_level >= 2:
            smart_print("SMT problem (size {}):".format(problem.formula_size()))
            smart_print(problem.hrepr(full_repr=verbose_level >= 3))

        if verbose_level >= 1:
            prefix = str(_get_time()) + " | "
            smart_print()
        else:
            prefix = ""

        id_found = problem.solve(initial_active_bits, solver_name=solver_name, search_mode=search_mode,
                                 key_input_diff_mode=key_input_diff_mode,
                                 enc_input_diff_mode=enc_input_diff_mode, enc_output_diff_mode=enc_output_diff_mode,
                                 to_extend=to_extend, check=check, verbose_level=verbose_level, filename=filename)

        # if no r-round impossible ch exists with A active bits
        # that does NOT imply that no (r+1)-round imp ch exists with A (or less) active bits
        if search_mode == SearchMode.FirstID:
            if id_found is None:
                if verbose_level == 0:
                    smart_print(prefix + "No ID found")
            else:
                if verbose_level >= 0:
                    smart_print(prefix + "ID found:")
                    if verbose_level == 0:
                        smart_print(id_found.srepr())
                    else:
                        smart_print(id_found)
                    if verbose_level >= 3:
                        smart_print(id_found.vrepr())