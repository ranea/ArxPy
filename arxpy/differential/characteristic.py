"""Manage characteristics."""
import enum
import collections
import itertools
import math
import random
import pprint

from arxpy import primitives
from arxpy.bitvector import core
from arxpy.bitvector import context
from arxpy.bitvector import operation
from arxpy.bitvector import extraop
from arxpy.differential import difference
from arxpy.differential import derivative
from arxpy.primitives import primitives


class ChSignatureType(enum.Enum):
    """Represent the different types of signatures available for a characteristic.

    Attributes:
        Full: the signature includes the input and all output differences of each
            non-linear operation.
        InputOutput: the signature only includes the input and output differences

    """
    Full = enum.auto()
    InputOutput = enum.auto()


class BvCharacteristic(object):
    """Represent characteristics of bit-vector functions.

    Given a bit-vector function `BvFunction` :math:`f`,
    a characteristic is a trail of differences obtained by
    propagating an input difference over :math:`f`.

    In particular, a characteristic is composed of the
    input difference and the output difference of each
    non-linear operation.

    This class manages symbolic characteristics,
    where the input difference is given symbolically
    and the intermediate differences are `Term`
    that depend on the input difference.

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.differential.difference import XorDiff, RXDiff
        >>> from arxpy.differential.characteristic import BvCharacteristic
        >>> from arxpy.primitives.primitives import BvFunction
        >>> from arxpy.primitives.chaskey import ChaskeyPi
        >>> issubclass(ChaskeyPi, BvFunction)
        True
        >>> ChaskeyPi.set_rounds(1)
        >>> ch = BvCharacteristic(ChaskeyPi, XorDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> ch.ssa  # doctest: +NORMALIZE_WHITESPACE
        {'input_vars': (dv0, dv1, dv2, dv3),
        'output_vars': (d7, d12, d13, d9),
        'assignments': ((d0, dv0 + dv1), (d1, dv1 <<< 5), (d2, d0 ^ d1), (d3, d0 <<< 16), (d4, dv2 + dv3),
        (d5, dv3 <<< 8), (d6, d4 ^ d5), (d7, d3 + d6), (d8, d6 <<< 13), (d9, d7 ^ d8), (d10, d2 + d4),
        (d11, d2 <<< 7), (d12, d10 ^ d11), (d13, d10 <<< 16))}
        >>> ch.input_diff
        (XorDiff(dv0), XorDiff(dv1), XorDiff(dv2), XorDiff(dv3))
        >>> ch.nonlinear_diffs # doctest: +NORMALIZE_WHITESPACE
        OrderedDict([(XorDiff(d0), XDA(XorDiff(dv0), XorDiff(dv1))),
        (XorDiff(d4), XDA(XorDiff(dv2), XorDiff(dv3))),
        (XorDiff(d7), XDA(XorDiff(d0 <<< 16), XorDiff(d4 ^ (dv3 <<< 8)))),
        (XorDiff(d10), XDA(XorDiff(d0 ^ (dv1 <<< 5)), XorDiff(d4)))])
        >>> ch.output_diff # doctest: +NORMALIZE_WHITESPACE
        [[XorDiff(d7), XorDiff(d7)],
        [XorDiff(d12), XorDiff(d10 ^ ((d0 ^ (dv1 <<< 5)) <<< 7))],
        [XorDiff(d13), XorDiff(d10 <<< 16)],
        [XorDiff(d9), XorDiff(d7 ^ ((d4 ^ (dv3 <<< 8)) <<< 13))]]
        >>> ch = BvCharacteristic(ChaskeyPi, RXDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> ch.input_diff
        (RXDiff(dv0), RXDiff(dv1), RXDiff(dv2), RXDiff(dv3))
        >>> ch.nonlinear_diffs # doctest: +NORMALIZE_WHITESPACE
        OrderedDict([(RXDiff(d0), RXDA(RXDiff(dv0), RXDiff(dv1))),
        (RXDiff(d4), RXDA(RXDiff(dv2), RXDiff(dv3))),
        (RXDiff(d7), RXDA(RXDiff(d0 <<< 16), RXDiff(d4 ^ (dv3 <<< 8)))),
        (RXDiff(d10), RXDA(RXDiff(d0 ^ (dv1 <<< 5)), RXDiff(d4)))])
        >>> ch.output_diff # doctest: +NORMALIZE_WHITESPACE
        [[RXDiff(d7), RXDiff(d7)],
        [RXDiff(d12), RXDiff(d10 ^ ((d0 ^ (dv1 <<< 5)) <<< 7))],
        [RXDiff(d13), RXDiff(d10 <<< 16)],
        [RXDiff(d9), RXDiff(d7 ^ ((d4 ^ (dv3 <<< 8)) <<< 13))]]

    Attributes:
        func: the `BvFunction`
        diff_type: the `Difference` of the characteristic
        input_diff: a list containing the input symbolic differences
        nonlinear_diffs: an `collections.OrderedDict` mapping non-linear symbolic differences
            to their corresponding `Derivative`
        output_diff: a list, where the i-th element is a pair containing
            the i-th output symbolic difference and its value
    """

    def __init__(self, func, diff_type, input_diff_names, prefix="d", initial_var2diff=None):
        assert issubclass(func, primitives.BvFunction)
        assert issubclass(diff_type, difference.Difference)

        assert len(input_diff_names) == len(func.input_widths)
        input_diff = []
        for name, width in zip(input_diff_names, func.input_widths):
            input_diff.append(diff_type(core.Variable(name, width)))
        input_diff = tuple(input_diff)

        self.func = func
        self.diff_type = diff_type
        self.input_diff = input_diff

        # Propagate the input difference through the function

        names = [d.val.name for d in self.input_diff]
        ssa = self.func.ssa(names, id_prefix=prefix)
        self.ssa = ssa
        self._prefix = prefix
        self._input_diff_names = input_diff_names

        for var in ssa["output_vars"]:
            if isinstance(var, core.Constant):
                raise ValueError("constant outputs (independent of the inputs) are not supported")

        var2diff = {}  # Variable to Difference
        for var, diff in zip(ssa["input_vars"], self.input_diff):
            var2diff[var] = diff

        if initial_var2diff is not None:
            for var in initial_var2diff:
                if str(var) in names:
                    raise ValueError("the input differences cannot be replaced by initial_var2diff")
            var2diff.update(initial_var2diff)

        self.nonlinear_diffs = collections.OrderedDict()
        for var, expr in ssa["assignments"]:
            expr_args = []
            for arg in expr.args:
                if isinstance(arg, int):
                    expr_args.append(arg)  # 'int' object has no attribute 'xreplace'
                else:
                    expr_args.append(arg.xreplace(var2diff))

            if all(not isinstance(arg, diff_type) for arg in expr_args):
                # symbolic computations with the key
                var2diff[var] = expr
                continue

            if all(isinstance(arg, diff_type) for arg in expr_args):
                der = self.diff_type.derivative(type(expr), expr_args)
            else:
                def contains_key_var(term):
                    from sympy import basic
                    for sub in basic.preorder_traversal(term):
                        if sub in func.round_keys:
                            return True
                    else:
                        return False

                if type(expr) == operation.BvAdd and hasattr(func, 'round_keys') and \
                        all(isinstance(r, core.Variable) for r in func.round_keys) and \
                        any(contains_key_var(a) for a in expr_args):
                    # temporary solution to Derivative(BvAddCte_k(x)) != Derivative(x + k)
                    # with x a Diff and k a key variable
                    keyed_indices = []
                    for i, a in enumerate(expr_args):
                        if contains_key_var(a):
                            keyed_indices.append(i)
                    if len(keyed_indices) != 1 or expr_args[keyed_indices[0]] not in func.round_keys:
                        raise NotImplementedError("invalid expression: op={}, args={}".format(
                            type(expr).__name__, expr_args))
                    # expr_args[keyed_indices[0]] replaced to the zero diff
                    zero_diff = diff_type(core.Constant(0, expr_args[keyed_indices[0]].width))
                    der = self.diff_type.derivative(type(expr), [expr_args[(keyed_indices[0] + 1) % 2], zero_diff])
                elif hasattr(expr, "xor_derivative"):
                    # temporary solution to operations containing a custom derivative
                    input_diff_expr = []
                    for i, arg in enumerate(expr_args):
                        if isinstance(arg, diff_type):
                            input_diff_expr.append(arg)
                        else:
                            assert isinstance(arg, core.Term)  # int arguments currently not supported
                            input_diff_expr.append(diff_type.from_pair(arg, arg))
                    der = self.diff_type.derivative(type(expr), input_diff_expr)
                else:
                    fixed_args = []
                    for i, arg in enumerate(expr_args):
                        if not isinstance(arg, diff_type):
                            fixed_args.append(arg)
                        else:
                            fixed_args.append(None)
                    new_op = extraop.make_partial_operation(type(expr), tuple(fixed_args))
                    der = self.diff_type.derivative(new_op, [arg for arg in expr_args if isinstance(arg, diff_type)])

            if isinstance(der, derivative.Derivative):
                diff = self.diff_type(var)
                var2diff[var] = diff
                self.nonlinear_diffs[diff] = der
            else:
                var2diff[var] = der

        self._var2diff = var2diff

        self.output_diff = []
        for var in ssa["output_vars"]:
            self.output_diff.append([self.diff_type(var), var2diff[var]])

    def empirical_weight(self, input_diff, output_diff, pair_samples):
        """Return the empirical weight of a given differential.

        Given a differential (a pair of input and output differences),
        the differential probability is the fraction of input pairs
        with the given input difference leading to output pairs
        with the given output difference.

        This method returns an approximation of the weight of the
        differential probability by sampling a given number
        of input pairs.

        If no correct output pairs are found, `math.inf` is returned.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.differential.difference import XorDiff, RXDiff
            >>> from arxpy.differential.characteristic import BvCharacteristic
            >>> from arxpy.primitives.chaskey import ChaskeyPi
            >>> ChaskeyPi.set_rounds(1)
            >>> ch = BvCharacteristic(ChaskeyPi, XorDiff, ["dv" + str(i) for i in range(4)])
            >>> zero, one = XorDiff(Constant(0, 32)), XorDiff(Constant(1, 32))
            >>> ch.empirical_weight([zero, zero, zero, zero], [zero, zero, zero, zero], 100)
            0.0
            >>> ch.empirical_weight([zero, zero, zero, zero], [one, one, one, one], 100)
            inf
            >>> ch = BvCharacteristic(ChaskeyPi, RXDiff, ["dv" + str(i) for i in range(4)])
            >>> zero, one = RXDiff(Constant(0, 32)), RXDiff(Constant(1, 32))
            >>> 4 - 1 <= ch.empirical_weight([zero]*4, [zero]*4, 3 * 2**6) <= 8
            True
            >>> ch.empirical_weight([zero]*4, [one]*4, 3 * 2**6)
            inf

        """
        assert isinstance(input_diff, collections.abc.Sequence)
        assert isinstance(output_diff, collections.abc.Sequence)
        assert all(isinstance(d, difference.Difference) for d in input_diff)
        assert all(isinstance(d, difference.Difference) for d in output_diff)
        assert all(isinstance(d.val, core.Constant) for d in input_diff)
        assert all(isinstance(d.val, core.Constant) for d in output_diff)

        assert len(input_diff) == len(self.input_diff)
        assert len(output_diff) == len(self.output_diff)
        assert len(self.ssa["input_vars"]) == len(input_diff)
        assert len(self.ssa["output_vars"]) == len(output_diff)

        with context.Simplification(False):
            input_widths = [d.val.width for d in self.input_diff]
            if pair_samples >= 2**sum(input_widths):
                iterators = [range(2 ** w) for w in input_widths]
                list_pairs = []
                for x in itertools.product(*iterators):
                    pt = [core.Constant(x_i, w) for x_i, w in zip(x, input_widths)]
                    other_pt = [diff.get_pair_element(pt[i]) for i, diff in enumerate(input_diff)]
                    list_pairs.append([pt, other_pt])
                pair_samples = len(list_pairs)
                assert pair_samples == 2**sum(input_widths)
            else:
                list_pairs = []
                for _ in range(pair_samples):
                    pt = []
                    other_pt = []
                    for diff in input_diff:
                        random_int = random.randrange(2 ** diff.val.width)
                        random_bv = core.Constant(random_int, diff.val.width)
                        pt.append(random_bv)
                        other_pt.append(diff.get_pair_element(random_bv))
                    list_pairs.append([pt, other_pt])

            correct_pairs = 0

            for index_input in range(pair_samples):
                pt, other_pt = list_pairs[index_input]
                ct = self.func(*pt)
                other_ct = self.func(*other_pt)

                assert all(isinstance(x, core.Constant) for x in ct), str(ct)
                assert all(isinstance(x, core.Constant) for x in other_ct), str(other_ct)

                for i, diff in enumerate(output_diff):
                    # noinspection PyUnresolvedReferences
                    if self.diff_type.from_pair(ct[i], other_ct[i]) != diff:
                        break
                else:
                    correct_pairs += 1

            if correct_pairs == 0:
                weight = math.inf
            else:
                weight = abs(-math.log(correct_pairs * 1.0 / pair_samples, 2))

        return weight

    def _empirical_weight_distribution(self, cipher, input_diff, output_diff, pair_samples, key_samples,
                                       precision=1, rk_output_diff=None):
        # this function is not part of SingleKeyCh since it must be accessible
        # for the encryption characteristic of RelatedKeyCh (which is a
        # plain BvCharacteristic)
        assert isinstance(input_diff, collections.abc.Sequence)
        assert isinstance(output_diff, collections.abc.Sequence)
        assert all(isinstance(d, difference.Difference) for d in input_diff)
        assert all(isinstance(d, difference.Difference) for d in output_diff)
        assert all(isinstance(d.val, core.Constant) for d in input_diff)
        assert all(isinstance(d.val, core.Constant) for d in output_diff)

        assert len(input_diff) == len(self.input_diff)
        assert len(output_diff) == len(self.output_diff)
        assert len(self.ssa["input_vars"]) == len(input_diff)
        assert len(self.ssa["output_vars"]) == len(output_diff)

        old_round_keys = self.func.round_keys

        empirical_weights = collections.Counter()

        if rk_output_diff is not None:
            class RelatedFunc(self.func):
                pass
        else:
            RelatedFunc = self.func

        with context.Simplification(False):
            input_widths = [d.val.width for d in self.input_diff]
            if pair_samples >= 2**sum(input_widths):
                iterators = [range(2 ** w) for w in input_widths]
                list_pairs = []
                for x in itertools.product(*iterators):
                    pt = [core.Constant(x_i, w) for x_i, w in zip(x, input_widths)]
                    other_pt = [diff.get_pair_element(pt[i]) for i, diff in enumerate(input_diff)]
                    list_pairs.append([pt, other_pt])
                pair_samples = len(list_pairs)
                assert pair_samples == 2**sum(input_widths)
            else:
                list_pairs = []
                for _ in range(pair_samples):
                    pt = []
                    other_pt = []
                    for diff in input_diff:
                        random_int = random.randrange(2 ** diff.val.width)
                        random_bv = core.Constant(random_int, diff.val.width)
                        pt.append(random_bv)
                        other_pt.append(diff.get_pair_element(random_bv))
                    list_pairs.append([pt, other_pt])

            for _ in range(key_samples):
                master_key = []
                for width in cipher.key_schedule.input_widths:
                    master_key.append(core.Constant(random.randrange(2 ** width), width))
                self.func.round_keys = cipher.key_schedule(*master_key)
                assert all(isinstance(rk, core.Constant) for rk in self.func.round_keys), str(self.func.round_keys)

                if rk_output_diff is not None:
                    RelatedFunc.round_keys = [d.get_pair_element(r) for r, d in zip(self.func.round_keys, rk_output_diff)]
                    assert all(isinstance(rk, core.Constant) for rk in RelatedFunc.round_keys), str(RelatedFunc.round_keys)

                correct_pairs = 0

                for index_input in range(pair_samples):
                    pt, other_pt = list_pairs[index_input]
                    ct = self.func(*pt)
                    other_ct = RelatedFunc(*other_pt)

                    assert all(isinstance(x, core.Constant) for x in ct), str(ct)
                    assert all(isinstance(x, core.Constant) for x in other_ct), str(other_ct)

                    for i, diff in enumerate(output_diff):
                        # noinspection PyUnresolvedReferences
                        if self.diff_type.from_pair(ct[i], other_ct[i]) != diff:
                            break
                    else:
                        correct_pairs += 1

                if correct_pairs == 0:
                    weight = math.inf
                else:
                    weight = abs(-math.log(correct_pairs * 1.0 / pair_samples, 2))
                # weight = float(("{0:."+str(precision)+"f}").format(weight))
                weight = round(weight, precision)
                empirical_weights[weight] += 1

        self.func.round_keys = old_round_keys

        return empirical_weights

    def signature(self, ch_signature_type):
        """Return the signature of the characteristic.

        The signature is a "hash" of the characteristic used for comparing.

        For the type of the signature, see `ChSignatureType`.

            >>> from arxpy.bitvector.core import Variable
            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.characteristic import BvCharacteristic, ChSignatureType
            >>> from arxpy.primitives.primitives import BvFunction
            >>> from arxpy.primitives.chaskey import ChaskeyPi
            >>> issubclass(ChaskeyPi, BvFunction)
            True
            >>> ChaskeyPi.set_rounds(1)
            >>> ch = BvCharacteristic(ChaskeyPi, XorDiff, ["dv0", "dv1", "dv2", "dv3"])
            >>> ch.signature(ChSignatureType.Full)
            [dv0, dv1, dv2, dv3, d0, d4, d7, d10]
            >>> ch.signature(ChSignatureType.InputOutput)
            [dv0, dv1, dv2, dv3, d7, d12, d13, d9]

        """
        if ch_signature_type == ChSignatureType.Full:
            return [d.val for d in self.input_diff] + [d.val for d in self.nonlinear_diffs]
        elif ch_signature_type == ChSignatureType.InputOutput:
            # sig = [d for d in self.input_diff]
            sig_var = [d.val for d in self.input_diff]

            for out_diff, _ in self.output_diff:
                for aux_var in self._var2diff[out_diff.val].val.atoms(core.Variable):
                    if aux_var not in sig_var:
                        # sig.append(out_diff)
                        sig_var.append(out_diff.val)
                        break

            return sig_var
        else:
            raise ValueError("invalid ch_signature_type: {}".format(ch_signature_type))

    def _to_dict(self):
        dict_ch = {
            "ssa": self.ssa,
            "input_diff": self.input_diff,
            "output_diff": self.output_diff,
            "nonlinear_diffs": self.nonlinear_diffs,
        }
        return dict_ch

    def __str__(self):
        return pprint.pformat(self._to_dict(), width=100, compact=True)


class SingleKeyCh(BvCharacteristic):
    """Represent single-key characteristics of block ciphers.

    A single-key characteristic of a `Cipher` is a `BvCharacteristic`
    over the `Encryption` function of the cipher.

    The plaintext differences start with the prefix ``"dp"``
    and the non-linear differences start with the prefix ``"dx"``.

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import SingleKeyCh
        >>> from arxpy.primitives.primitives import Cipher
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> issubclass(Speck32, Cipher)
        True
        >>> Speck32.set_rounds(1)
        >>> ch = SingleKeyCh(Speck32, XorDiff)
        >>> ch .ssa  # doctest: +NORMALIZE_WHITESPACE
        {'input_vars': (dp0, dp1), 'output_vars': (dx2, dx4),
        'assignments': ((dx0, dp0 >>> 7), (dx1, dp1 + dx0), (dx2, dx1 ^ k0), (dx3, dp1 <<< 2), (dx4, dx2 ^ dx3))}
        >>> ch.input_diff
        (XorDiff(dp0), XorDiff(dp1))
        >>> ch.nonlinear_diffs
        OrderedDict([(XorDiff(dx1), XDA(XorDiff(dp1), XorDiff(dp0 >>> 7)))])
        >>> ch.output_diff
        [[XorDiff(dx2), XorDiff(dx1)], [XorDiff(dx4), XorDiff(dx1 ^ (dp1 <<< 2))]]

    """

    def __init__(self, bv_cipher, diff_type):
        assert issubclass(bv_cipher, primitives.Cipher)
        assert issubclass(diff_type, difference.Difference)

        rk = []
        for i, width in enumerate(bv_cipher.key_schedule.output_widths):
            rk.append(core.Variable("k" + str(i), width))

        class Encryption(bv_cipher.encryption):
            round_keys = tuple(rk)

        func = Encryption
        num_inputs = len(func.input_widths)
        input_diff_names = ["dp" + str(i) for i in range(num_inputs)]
        prefix = "dx"
        super().__init__(func, diff_type, input_diff_names, prefix)
        self._cipher = bv_cipher

    def empirical_weight(self, input_diff, output_diff, pair_samples, key_samples,
                         precision=1, rk_diffs=None):
        """Return the empirical weight distribution of a given differential.

        This method returns a `collections.Counter` storing the distribution of
        differential probability weights over the given number of keys.

        The weights are rounded to the given number of precision
        digits after the decimal point.

        See also `BvCharacteristic.empirical_weight`.

            >>> from arxpy.bitvector.core import Constant
            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.characteristic import SingleKeyCh
            >>> from arxpy.primitives import speck
            >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
            >>> Speck32.set_rounds(1)
            >>> ch = SingleKeyCh(Speck32, XorDiff)
            >>> zero, one = XorDiff(Constant(0, 16)), XorDiff(Constant(1, 16))
            >>> ch.empirical_weight([zero, zero], [zero, zero], 100, 10)
            Counter({0.0: 10})
            >>> ch.empirical_weight([zero, zero], [one, one], 100, 10)
            Counter({inf: 10})

        """
        return self._empirical_weight_distribution(self._cipher, input_diff, output_diff, pair_samples, key_samples,
                                                   precision, rk_diffs)


class RelatedKeyCh(object):
    """Represent related-key characteristics of block ciphers.

    A related-key characteristic of a `Cipher` is a pair `BvCharacteristic`,
    one over the `KeySchedule` of the cipher, and another one over the
    the `Encryption` function of the cipher, where the output differences
    of the key schedule characteristic are used as round key differences
    in the encryption characteristic.

    The master key differences start with the prefix ``"dmk"``,
    the round key differences start with the prefix ``"dk"``,
    the plaintext differences start with the prefix ``"dp"``
    and the non-linear differences start with the prefix ``"dx"``.

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import RelatedKeyCh
        >>> from arxpy.primitives.primitives import Cipher
        >>> from arxpy.primitives.lea import LeaCipher
        >>> issubclass(LeaCipher, Cipher)
        True
        >>> LeaCipher.set_rounds(1)
        >>> rkch = RelatedKeyCh(LeaCipher, XorDiff)
        >>> rkch .key_schedule_ch.ssa  # doctest: +NORMALIZE_WHITESPACE
        {'input_vars': (dmk0, dmk1, dmk2, dmk3),
        'output_vars': (dk1, dk3, dk5, dk3, dk7, dk3),
        'assignments': ((dk0, 0xc3efe9db + dmk0), (dk1, dk0 <<< 1), (dk2, 0x87dfd3b7 + dmk1), (dk3, dk2 <<< 3),
        (dk4, 0x0fbfa76f + dmk2), (dk5, dk4 <<< 6), (dk6, 0x1f7f4ede + dmk3), (dk7, dk6 <<< 11))}
        >>> rkch.key_schedule_ch.input_diff
        (XorDiff(dmk0), XorDiff(dmk1), XorDiff(dmk2), XorDiff(dmk3))
        >>> rkch.key_schedule_ch.output_diff  # doctest: +NORMALIZE_WHITESPACE
        [[XorDiff(dk1), XorDiff(dk0 <<< 1)], [XorDiff(dk3), XorDiff(dk2 <<< 3)],
        [XorDiff(dk5), XorDiff(dk4 <<< 6)], [XorDiff(dk3), XorDiff(dk2 <<< 3)],
        [XorDiff(dk7), XorDiff(dk6 <<< 11)], [XorDiff(dk3), XorDiff(dk2 <<< 3)]]
        >>> rkch.key_schedule_ch.nonlinear_diffs  # doctest: +NORMALIZE_WHITESPACE
        OrderedDict([(XorDiff(dk0), XDCA_0xc3efe9db(XorDiff(dmk0))),
        (XorDiff(dk2), XDCA_0x87dfd3b7(XorDiff(dmk1))),
        (XorDiff(dk4), XDCA_0x0fbfa76f(XorDiff(dmk2))),
        (XorDiff(dk6), XDCA_0x1f7f4ede(XorDiff(dmk3)))])
        >>> rkch.encryption_ch.ssa  # doctest: +NORMALIZE_WHITESPACE
        {'input_vars': (dp0, dp1, dp2, dp3),
        'output_vars': (dx3, dx7, dx11, dp0),
        'assignments': ((dx0, dk1 ^ dp0), (dx1, dk3 ^ dp1), (dx2, dx0 + dx1), (dx3, dx2 <<< 9),
        (dx4, dk5 ^ dp1), (dx5, dk3 ^ dp2), (dx6, dx4 + dx5), (dx7, dx6 >>> 5), (dx8, dk7 ^ dp2),
        (dx9, dk3 ^ dp3), (dx10, dx8 + dx9), (dx11, dx10 >>> 3))}
        >>> rkch.encryption_ch.input_diff
        (XorDiff(dp0), XorDiff(dp1), XorDiff(dp2), XorDiff(dp3))
        >>> rkch.encryption_ch.output_diff # doctest: +NORMALIZE_WHITESPACE
        [[XorDiff(dx3), XorDiff(dx2 <<< 9)], [XorDiff(dx7), XorDiff(dx6 >>> 5)],
        [XorDiff(dx11), XorDiff(dx10 >>> 3)], [XorDiff(dp0), XorDiff(dp0)]]
        >>> rkch.encryption_ch.nonlinear_diffs  # doctest: +NORMALIZE_WHITESPACE
        OrderedDict([(XorDiff(dx2), XDA(XorDiff(dp0 ^ (dk0 <<< 1)), XorDiff(dp1 ^ (dk2 <<< 3)))),
        (XorDiff(dx6), XDA(XorDiff(dp1 ^ (dk4 <<< 6)), XorDiff(dp2 ^ (dk2 <<< 3)))),
        (XorDiff(dx10), XDA(XorDiff(dp2 ^ (dk6 <<< 11)), XorDiff(dp3 ^ (dk2 <<< 3))))])

    Attributes:
        key_schedule_ch: the `BvCharacteristic` over the key schedule
        encryption_ch: the `BvCharacteristic` over the encryption function
    """

    def __init__(self, bv_cipher, diff_type):
        assert issubclass(bv_cipher, primitives.Cipher)
        assert issubclass(diff_type, difference.Difference)

        func = bv_cipher.key_schedule
        prefix = "dk"
        input_diff_names = tuple(["dmk" + str(i) for i in range(len(func.input_widths))])
        ks_ch = BvCharacteristic(func, diff_type, input_diff_names, prefix)

        class Encryption(bv_cipher.encryption):
            round_keys = ks_ch.ssa["output_vars"]

        func = Encryption
        prefix = "dx"
        input_diff_names = ["dp" + str(i) for i in range(len(func.input_widths))]
        round_key_diff = {}
        for var, diff in ks_ch.output_diff:
            round_key_diff[var.val] = diff
        encryption_ch = BvCharacteristic(func, diff_type, input_diff_names,
                                         prefix, round_key_diff)

        self.diff_type = diff_type
        self.key_schedule_ch = ks_ch
        self.encryption_ch = encryption_ch
        self._cipher = bv_cipher

    def empirical_weight(self, key_input_diff, key_output_diff, key_samples,
                         enc_input_diff, enc_output_diff, enc_samples, precision=1):
        """Return the empirical weight of a given differential for multiple keys.

        This method returns the differential probability weight for the
        key schedule characteristic (see `BvCharacteristic.empirical_weight`)
        and the `collections.Counter` storing the distribution of weights for the encryption
        characteristic (see `SingleKeyCh.empirical_weight`).

            >>> from arxpy.bitvector.core import Variable, Constant
            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.characteristic import RelatedKeyCh
            >>> from arxpy.primitives.lea import LeaCipher
            >>> LeaCipher.set_rounds(1)
            >>> rkch = RelatedKeyCh(LeaCipher, XorDiff)
            >>> zero, one = XorDiff(Constant(0, 32)), XorDiff(Constant(1, 32))
            >>> kid, kod = [zero]*4, [zero]*6
            >>> eid, eod = [zero]*4, [zero]*4
            >>> rkch.empirical_weight(kid, kod, 10, eid, eod, 100)
            (0.0, Counter({0.0: 10}))
            >>> kid, kod = [zero]*4, [one]*6
            >>> eid, eod = [zero]*4, [one]*4
            >>> rkch.empirical_weight(kid, kod, 10, eid, eod, 100)
            (inf, Counter({inf: 10}))

        """
        key_weight = self.key_schedule_ch.empirical_weight(key_input_diff, key_output_diff, key_samples)
        # noinspection PyProtectedMember
        enc_counter = self.encryption_ch._empirical_weight_distribution(self._cipher,
                                                                        enc_input_diff, enc_output_diff, enc_samples,
                                                                        key_samples, precision, key_output_diff)
        return key_weight, enc_counter

    def signature(self, ch_signature_type):
        """Return the signature of the related-key characteristic.

        The signature of a related-key characteristic is the
        concatenation of the key schedule and encryption signatures.

        See also `BvCharacteristic.signature`.

            >>> from arxpy.bitvector.core import Variable
            >>> from arxpy.differential.difference import XorDiff
            >>> from arxpy.differential.characteristic import RelatedKeyCh, ChSignatureType
            >>> from arxpy.primitives.primitives import Cipher
            >>> from arxpy.primitives.lea import LeaCipher
            >>> LeaCipher.set_rounds(1)
            >>> rkch = RelatedKeyCh(LeaCipher, XorDiff)
            >>> rkch.signature(ChSignatureType.Full)  # doctest:+NORMALIZE_WHITESPACE
            [dmk0, dmk1, dmk2, dmk3, dk0, dk2, dk4, dk6, dp0, dp1, dp2, dp3, dx2, dx6, dx10]
            >>> rkch.signature(ChSignatureType.InputOutput)  # doctest:+NORMALIZE_WHITESPACE
            [dmk0, dmk1, dmk2, dmk3, dk1, dk3, dk5, dk3, dk7, dk3, dp0, dp1, dp2, dp3, dx3, dx7, dx11]

        """
        return self.key_schedule_ch.signature(ch_signature_type) + self.encryption_ch.signature(ch_signature_type)

    def _to_dict(self):
        dict_ch = {
            "key_schedule_ch": self.key_schedule_ch._to_dict(),
            "encryption_ch": self.encryption_ch._to_dict(),
        }
        return dict_ch

    def __str__(self):
        return pprint.pformat(self._to_dict(), width=100, compact=True)
