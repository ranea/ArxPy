"""Verify impossible differentials found by the SMT solver."""
import collections
import itertools
import math
import multiprocessing
import random

from arxpy.bitvector import core
from arxpy.smt.verification_differential import (
    ssa2ccode, relatedssa2ccode, compile_run_empirical_weight
)


MAX_WEIGHT = 20  # pairs = 10 * (2**(MAX_WEIGHT), MW = 20, pairs = 10**7
KEY_SAMPLES = 256  # total complexity 2^{30}


def fast_empirical_weight(id_found, verbose_lvl=0, debug=False, filename=None):
    """Computes the empirical weight of the model using C code.

    If ``filename`` is not ``None``, the output will be printed
    to the given file rather than the to stdout.

    The argument ``verbose_lvl`` can take an integer between
    ``0`` (no verbose) and ``3`` (full verbose).

        >>> from arxpy.differential.difference import XorDiff, RXDiff
        >>> from arxpy.differential.characteristic import BvCharacteristic
        >>> from arxpy.primitives.chaskey import ChaskeyPi
        >>> from arxpy.smt.search_impossible import SearchID
        >>> from arxpy.smt.verification_impossible import fast_empirical_weight
        >>> ChaskeyPi.set_rounds(2)
        >>> ch = BvCharacteristic(ChaskeyPi, XorDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> search_problem = SearchID(ch)
        >>> id_found = search_problem.solve(2)
        >>> fast_empirical_weight(id_found)
        inf
        >>> ch = BvCharacteristic(ChaskeyPi, RXDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> search_problem = SearchID(ch)
        >>> id_found = search_problem.solve(2)
        >>> fast_empirical_weight(id_found)
        inf

    """
    from arxpy.smt.search_differential import _get_smart_print  # avoid cyclic imports

    smart_print = _get_smart_print(filename)

    if debug:
        smart_print("Symbolic characteristic:")
        smart_print(id_found.ch)
        smart_print("ID found:")
        smart_print(id_found)
        smart_print()

    assert len(id_found.ch.nonlinear_diffs.items()) > 0

    ssa = id_found.ch.ssa.copy()
    ssa["assignments"] = list(ssa["assignments"])
    ssa["output_vars"] = list(ssa["output_vars"])

    # fixing duplicate var problem
    var2diffval = {}
    for diff_var, diff_value in itertools.chain(id_found.input_diff, id_found.output_diff):
        var2diffval[diff_var.val] = diff_value.val

    for j in range(len(ssa["output_vars"])):
        var_j = ssa["output_vars"][j]
        index_out = 0
        if var_j in ssa["input_vars"]:
            new_var = type(var_j)(var_j.name + "_o" + str(index_out), var_j.width)
            index_out += 1
            ssa["assignments"].append([new_var, var_j])
            ssa["output_vars"][j] = new_var
            var2diffval[new_var] = var2diffval[var_j]

        for k in range(j + 1, len(ssa["output_vars"])):
            if var_j == ssa["output_vars"][k]:
                new_var = type(var_j)(var_j.name + "_o" + str(index_out), var_j.width)
                index_out += 1
                ssa["assignments"].append([new_var, var_j])
                ssa["output_vars"][k] = new_var
                var2diffval[new_var] = var2diffval[var_j]

    ccode = ssa2ccode(ssa, id_found.ch.diff_type)

    if verbose_lvl >= 3:
        smart_print("  - ssa:", ssa)  # pprint.pformat(ssa, width=100))
    if debug:
        smart_print(ccode[0])
        smart_print(ccode[1])
        smart_print()

    input_diff_c = [v.xreplace(var2diffval) for v in ssa["input_vars"]]
    output_diff_c = [v.xreplace(var2diffval) for v in ssa["output_vars"]]

    if verbose_lvl >= 2:
        smart_print("  - checking {} -> {} pairs 2**{}".format(
            '|'.join([str(d) for d in input_diff_c]), '|'.join([str(d) for d in output_diff_c]),
            MAX_WEIGHT))

    input_diff_c = [int(d.val) for d in input_diff_c]
    output_diff_c = [int(d.val) for d in output_diff_c]

    assert all(isinstance(d, (int, core.Constant)) for d in input_diff_c), "{}".format(input_diff_c)
    assert all(isinstance(d, (int, core.Constant)) for d in output_diff_c), "{}".format(output_diff_c)

    current_empirical_weight = compile_run_empirical_weight(
        ccode,
        "_libver" + id_found.ch.func.__name__,
        input_diff_c,
        output_diff_c,
        MAX_WEIGHT,
        verbose=verbose_lvl >= 4)

    if verbose_lvl >= 2:
        smart_print("  - empirical weight: {}".format(current_empirical_weight))

    if current_empirical_weight == math.inf:
        return math.inf
    else:
        return current_empirical_weight


def _fast_empirical_weight_distribution(ch_found, cipher, rk_dict_diffs=None,
                                        verbose_lvl=0, debug=False, filename=None, precision=0):
    """

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import SingleKeyCh
        >>> from arxpy.smt.search_impossible import SearchSkID
        >>> from arxpy.primitives import speck
        >>> from arxpy.smt.verification_impossible import _fast_empirical_weight_distribution
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_rounds(1)
        >>> ch = SingleKeyCh(Speck32, XorDiff)
        >>> search_problem = SearchSkID(ch)
        >>> id_found = search_problem.solve(2)
        >>> _fast_empirical_weight_distribution(id_found, Speck32)
        Counter({inf: 256})

    """
    if rk_dict_diffs is not None:
        raise ValueError("rk_dict_diffs must be None")

    from arxpy.smt.search_differential import _get_smart_print  # avoid cyclic imports

    smart_print = _get_smart_print(filename)

    # if rk_dict_diffs is not None:
    #     assert "nonlinear_diffs" in rk_dict_diffs and "output_diff" in rk_dict_diffs

    if debug:
        smart_print("Symbolic characteristic:")
        smart_print(ch_found.ch)
        smart_print("ID found:")
        smart_print(ch_found)
        # if rk_dict_diffs is not None:
        #     smart_print("rk_dict_diffs:", rk_dict_diffs)
        smart_print()

    # if rk_dict_diffs is not None:
    #     rk_var = [var.val for var, _ in rk_dict_diffs["output_diff"]]
    # else:
    rk_var = []
    for i, width in enumerate(cipher.key_schedule.output_widths):
        rk_var.append(core.Variable("k" + str(i), width))

    var2diffval = {}
    for diff_var, diff_value in itertools.chain(ch_found.input_diff, ch_found.output_diff):
        var2diffval[diff_var.val] = diff_value.val
    # if rk_dict_diffs is not None:
    #     for var, diff in rk_dict_diffs["output_diff"]:
    #         var2diffval[var.val] = diff.val

    # for each related-key pair, we associated a pair of ssa
    rkey2pair_ssa = [None for _ in range(KEY_SAMPLES)]
    for key_index in range(KEY_SAMPLES):
        master_key = []
        for width in cipher.key_schedule.input_widths:
            master_key.append(core.Constant(random.randrange(2 ** width), width))
        rk_val = cipher.key_schedule(*master_key)
        # if rk_dict_diffs is not None:
        #     rk_other_val = tuple([d.get_pair_element(r) for r, (_, d) in zip(rk_val, rk_dict_diffs["output_diff"])])
        # else:
        rk_other_val = rk_val
        assert len(rk_var) == len(rk_other_val)
        assert all(isinstance(rk, core.Constant) for rk in rk_val)
        assert all(isinstance(rk, core.Constant) for rk in rk_other_val)

        def replace_roundkeys(var2val):
            new_ssa = ch_found.ch.ssa.copy()
            new_ssa["assignments"] = list(new_ssa["assignments"])
            new_ssa["output_vars"] = list(new_ssa["output_vars"])

            for i, (var, expr) in enumerate(ch_found.ch.ssa["assignments"]):
                new_ssa["assignments"][i] = (var, expr.xreplace(var2val))

            return new_ssa

        pair_ssa = []
        for index_pair in range(2):
            current_rk_val = rk_val if index_pair == 0 else rk_other_val
            rkvar2rkval = {var: val for var, val in zip(rk_var, current_rk_val)}
            ssa = replace_roundkeys(rkvar2rkval)

            for j in range(len(ssa["output_vars"])):
                var_j = ssa["output_vars"][j]
                index_out = 0
                if var_j in ssa["input_vars"]:
                    new_var = type(var_j)(var_j.name + "_o" + str(index_out), var_j.width)
                    index_out += 1
                    ssa["assignments"].append([new_var, var_j])
                    ssa["output_vars"][j] = new_var
                    var2diffval[new_var] = var2diffval[var_j]

                for k in range(j + 1, len(ssa["output_vars"])):
                    if var_j == ssa["output_vars"][k]:
                        new_var = type(var_j)(var_j.name + "_o" + str(index_out), var_j.width)
                        index_out += 1
                        ssa["assignments"].append([new_var, var_j])
                        ssa["output_vars"][k] = new_var
                        var2diffval[new_var] = var2diffval[var_j]

            pair_ssa.append(ssa)

        rkey2pair_ssa[key_index] = pair_ssa

    # for each related-key pair, we associated their weight
    rkey2subch_ew = [0 for _ in range(KEY_SAMPLES)]

    # start multiprocessing
    with multiprocessing.Pool() as pool:
        for key_index in range(KEY_SAMPLES):
            ssa1 = rkey2pair_ssa[key_index][0]
            ssa2 = rkey2pair_ssa[key_index][1]

            if key_index <= 1:
                if verbose_lvl >= 3:
                    smart_print("  - related-key pair index", key_index)
                    smart_print("  - ssa1:", ssa1)
                    if ssa1 == ssa2:
                        smart_print("  - ssa2: (same as ssa1)")
                    else:
                        smart_print("  - ssa2:", ssa2)

            if ssa1 == ssa2:
                ccode = ssa2ccode(ssa1, ch_found.ch.diff_type)
            else:
                ccode = relatedssa2ccode(ssa1, ssa2, ch_found.ch.diff_type)

            if key_index <= 1 and debug:
                smart_print(ccode[0])
                smart_print(ccode[1])
                smart_print()

            input_diff_c = [v.xreplace(var2diffval) for v in ssa1["input_vars"]]
            output_diff_c = [v.xreplace(var2diffval) for v in ssa1["output_vars"]]

            if key_index <= 1 and verbose_lvl >= 2:
                smart_print("  - rk{} | checking {} -> {} with pairs 2**{}".format(
                    key_index,
                    '|'.join([str(d) for d in input_diff_c]), '|'.join([str(d) for d in output_diff_c]),
                    MAX_WEIGHT))

            assert all(isinstance(d, (int, core.Constant)) for d in input_diff_c), "{}".format(input_diff_c)
            assert all(isinstance(d, (int, core.Constant)) for d in output_diff_c), "{}".format(output_diff_c)

            input_diff_c = [int(d) for d in input_diff_c]
            output_diff_c = [int(d) for d in output_diff_c]

            rkey2subch_ew[key_index] = pool.apply_async(
                compile_run_empirical_weight,
                (
                    ccode,
                    "_libver" + ch_found.ch.func.__name__,
                    input_diff_c,
                    output_diff_c,
                    MAX_WEIGHT,
                    False
                )
            )

        # wait until all have been compiled and run
        # and replace the Async object by the result
        for key_index in range(KEY_SAMPLES):
            if isinstance(rkey2subch_ew[key_index], multiprocessing.pool.AsyncResult):
                rkey2subch_ew[key_index] = rkey2subch_ew[key_index].get()

            if key_index <= 1 and verbose_lvl >= 2:
                smart_print("  - rk{} | empirical weight: {}".format(
                    key_index, rkey2subch_ew[key_index]))

    # end multiprocessing

    empirical_weight_distribution = collections.Counter()
    all_rkey_weights = []
    for key_index in range(KEY_SAMPLES):
        rkey_weight = rkey2subch_ew[key_index]

        if precision == 0:
            weight = int(rkey_weight) if rkey_weight != math.inf else math.inf
        else:
            weight = round(rkey_weight, precision)

        all_rkey_weights.append(rkey_weight)
        empirical_weight_distribution[weight] += 1

    if verbose_lvl >= 2:
        smart_print("- distribution empirical weights: {}".format(empirical_weight_distribution))

    if verbose_lvl >= 3:
        smart_print("- list empirical weights:", [round(x, 8) for x in all_rkey_weights if x != math.inf])

    return empirical_weight_distribution

