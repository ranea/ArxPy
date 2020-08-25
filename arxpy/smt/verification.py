"""Verify characteristics found by the SMT solver.

To this end, a characteristic is split into sub-characteristics
where each sub-ch has up to `MAX_WEIGHT` weight.
"""
import collections
import importlib.util
import itertools
import math
import multiprocessing
import random
import tempfile

from arxpy.bitvector import operation, extraop, core
from arxpy.differential import difference

import cffi

MAX_WEIGHT = 20  # pairs = 10 * (2**(MAX_WEIGHT), MW = 20, pairs = 10**7
KEY_SAMPLES = 256  # total complexity 2^{30}


def bv2ccode(bv):
    """Convert a bit-vector type to C code.

    Args:
        bv: the bit-vector `Term` to convert
    ::

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import RotateLeft
        >>> from arxpy.smt.verification import bv2ccode
        >>> x, y = Variable("x", 8), Variable("y", 8)
        >>> bv2ccode(x | y)
        'x | y'
        >>> bv2ccode(x + y)
        '(x + y) & 255'
        >>> bv2ccode(RotateLeft(x, 1))
        '((x << 1) | (x >> 7)) & 255'
        >>> bv2ccode((~x) + y)
        Traceback (most recent call last):
        ...
        ValueError: nested bit-vector operations are not supported

    """
    # Note that every operation involving a shift or
    # an arithmetic operation (add) requires the result to be masked.

    if issubclass(type(bv), extraop.PartialOperation):
        raise NotImplementedError("PartialOperation is not supported")

    # only variables or constants
    if not all(isinstance(arg, (int, core.Constant, core.Variable)) for arg in bv.args):
        raise ValueError("nested bit-vector operations are not supported")

    if isinstance(bv, (core.Constant, core.Variable)):
        return str(bv)

    if type(bv) == operation.BvNot:
        return str(bv)
    elif type(bv) == operation.BvAnd:
        return str(bv)
    elif type(bv) == operation.BvOr:
        return str(bv)
    elif type(bv) == operation.BvXor:
        return str(bv)
    elif type(bv) in [operation.BvShl, operation.BvLshr]:
        return "({}) & {}".format(bv, 2**bv.width - 1)

    elif type(bv) == operation.RotateLeft:
        # #define ROTL1(a) ((a << 1) | (a >> (width - 1))) & (2**width - 1)
        x, r = bv.args
        return "(({0} << {1}) | ({0} >> {2})) & {3}".format(x, r, x.width - r, 2**x.width - 1)
    elif type(bv) == operation.RotateRight:
        # #define ROTL1(a) ((a >> 1) | (a << width - 1)) & (2**width - 1)
        x, r = bv.args
        return "(({0} >> {1}) | ({0} << {2})) & {3}".format(x, r, x.width - r, 2**x.width - 1)
    elif type(bv) == operation.BvAdd:
        x, y = bv.args
        return "({} + {}) & {}".format(x, y, 2**x.width - 1)
    elif type(bv) == operation.BvSub:
        x, y = bv.args
        m = 2**x.width - 1
        # no need to mask (result is <= 2**n)
        return f"({x} >= {y}) ? {x} - {y} : {m} - {y} + {x}"

    from arxpy.primitives.simon import SimonRF
    from arxpy.primitives.shacal1 import BvIf, BvMaj
    from arxpy.primitives.multi2 import BvOr as Multi2BvOr

    if isinstance(bv, SimonRF):
        #  ((x <<< a) & (x <<< b)) ^ (x <<< c)
        return "( (({}) & ({})) ^ ({}) ) & {}".format(
            bv2ccode(operation.RotateLeft(bv.args[0], SimonRF.a)),
            bv2ccode(operation.RotateLeft(bv.args[0], SimonRF.b)),
            bv2ccode(operation.RotateLeft(bv.args[0], SimonRF.c)),
            2 ** bv.args[0].width -1
        )
    elif isinstance(bv, BvIf):
        # (x & y) | ((~x) & z)
        x, y, z = bv.args
        return "({0} & {1}) | ((~{0}) & {2})".format(x, y, z)
    elif isinstance(bv, BvMaj):
        # (x & y) | (x & z) | (y & z)
        x, y, z = bv.args
        return "({0} & {1}) | ({0} & {2}) | ({1} & {2})".format(x, y, z)
    elif isinstance(bv, Multi2BvOr):
        x, y = bv.args
        return "{} | {}".format(x, y)

    else:
        raise ValueError("invalid operation: {}".format(type(bv)))

    # elif type(bv) == operation.Ite:
    # elif type(bv) == operation.BvComp:
    # elif type(bv) == operation.BvUlt:
    # elif type(bv) == operation.BvUle:
    # elif type(bv) == operation.BvUgt:
    # elif type(bv) == operation.BvUge:
    # elif type(bv) == operation.Extract:
    # elif type(bv) == operation.Concat:
    # elif type(bv) == operation.ZeroExtend:
    # elif type(bv) == operation.Repeat:
    # elif type(bv) == operation.BvNeg:
    # elif type(bv) == operation.BvSub:
    # elif type(bv) == operation.BvMul:
    # elif type(bv) == operation.BvUdiv:
    # elif type(bv) == operation.BvUrem:


# no need to include <stdint.h>
# input testing:
# printf("input %d | %u %u\\n", j, input1[j], input2[j]);
# output testing (XorDiff):
# printf("%d | out1=%u out2=%u diff_out=%u th_diff_out=%u\\n", j, output1[j], output2[j], output1[j] ^ output2[j], output_diff[j]);

get_num_valid_pairs_header = "static unsigned long get_num_valid_pairs({ctype_input} input_diff[], {ctype_output} output_diff[], unsigned long pair_samples, unsigned int seed);"

get_num_valid_pairs_source = """
static unsigned long get_num_valid_pairs({ctype_input} input_diff[], {ctype_output} output_diff[], unsigned long pair_samples, unsigned int seed){{
    if (seed == 0) srand((unsigned int) time (NULL));
    else srand(seed);
    unsigned long num_valid_pairs = 0;
    unsigned long i = 0;
    unsigned int j = 0;
    {ctype_input} input1[{len_input}];
    {ctype_input} input2[{len_input}];
    {ctype_output} output1[{len_output}];
    {ctype_output} output2[{len_output}];
    for (; i < pair_samples; ++i) {{
        for (j = 0; j < {len_input}; ++j) {{
            input1[j] = rand();
            input2[j] = {input_diff_ccode};
        }}
        {eval1}({input1}, {output1});
        {eval2}({input2}, {output2});
        for (j = 0; j < {len_output}; ++j) {{
            if ( ({output_diff_ccode}) != output_diff[j] ) break;
            if (j == {len_output} - 1) num_valid_pairs += 1;
        }}
    }}
    return num_valid_pairs;
}}"""


def ssa2ccode(ssa, diff_type):
    """Return the C code to compute a differential probability over a function in ssa form.

        >>> from arxpy.differential.difference import XorDiff, RXDiff
        >>> from arxpy.primitives.chaskey import ChaskeyPi
        >>> from arxpy.smt.verification import ssa2ccode
        >>> ChaskeyPi.set_rounds(1)
        >>> ssa = ChaskeyPi.ssa(["v0", "v1", "v2", "v3"], "x")  # doctest: +NORMALIZE_WHITESPACE
        >>> header, source = ssa2ccode(ssa, XorDiff)
        >>> print(header)
        static unsigned long get_num_valid_pairs(uint32_t input_diff[], uint32_t output_diff[], unsigned long pair_samples, unsigned int seed);
        >>> print(source)  # doctest:+NORMALIZE_WHITESPACE
        void eval(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, uint32_t* x7, uint32_t* x12, uint32_t* x13, uint32_t* x9){
            uint32_t x0 = (v0 + v1) & 4294967295;
            uint32_t x1 = ((v1 << 5) | (v1 >> 27)) & 4294967295;
            uint32_t x2 = x0 ^ x1;
            uint32_t x3 = ((x0 << 16) | (x0 >> 16)) & 4294967295;
            uint32_t x4 = (v2 + v3) & 4294967295;
            uint32_t x5 = ((v3 << 8) | (v3 >> 24)) & 4294967295;
            uint32_t x6 = x4 ^ x5;
            *x7 = (x3 + x6) & 4294967295;
            uint32_t x8 = ((x6 << 13) | (x6 >> 19)) & 4294967295;
            *x9 = *x7 ^ x8;
            uint32_t x10 = (x2 + x4) & 4294967295;
            uint32_t x11 = ((x2 << 7) | (x2 >> 25)) & 4294967295;
            *x12 = x10 ^ x11;
            *x13 = ((x10 << 16) | (x10 >> 16)) & 4294967295;
        };
        static unsigned long get_num_valid_pairs(uint32_t input_diff[], uint32_t output_diff[], unsigned long pair_samples, unsigned int seed){
            if (seed == 0) srand((unsigned int) time (NULL));
            else srand(seed);
            unsigned long num_valid_pairs = 0;
            unsigned long i = 0;
            unsigned int j = 0;
            uint32_t input1[4];
            uint32_t input2[4];
            uint32_t output1[4];
            uint32_t output2[4];
            for (; i < pair_samples; ++i) {
                for (j = 0; j < 4; ++j) {
                    input1[j] = rand();
                    input2[j] = input1[j] ^ input_diff[j];
                }
                eval(input1[0], input1[1], input1[2], input1[3], &output1[0], &output1[1], &output1[2], &output1[3]);
                eval(input2[0], input2[1], input2[2], input2[3], &output2[0], &output2[1], &output2[2], &output2[3]);
                for (j = 0; j < 4; ++j) {
                    if ( (output1[j] ^ output2[j]) != output_diff[j] ) break;
                    if (j == 4 - 1) num_valid_pairs += 1;
                }
            }
            return num_valid_pairs;
        }
        >>> header, source = ssa2ccode(ssa, RXDiff)
        >>> print(header)
        static unsigned long get_num_valid_pairs(uint32_t input_diff[], uint32_t output_diff[], unsigned long pair_samples, unsigned int seed);
        >>> print(source)  # doctest:+NORMALIZE_WHITESPACE
        void eval(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, uint32_t* x7, uint32_t* x12, uint32_t* x13, uint32_t* x9){
            uint32_t x0 = (v0 + v1) & 4294967295;
            uint32_t x1 = ((v1 << 5) | (v1 >> 27)) & 4294967295;
            uint32_t x2 = x0 ^ x1;
            uint32_t x3 = ((x0 << 16) | (x0 >> 16)) & 4294967295;
            uint32_t x4 = (v2 + v3) & 4294967295;
            uint32_t x5 = ((v3 << 8) | (v3 >> 24)) & 4294967295;
            uint32_t x6 = x4 ^ x5;
            *x7 = (x3 + x6) & 4294967295;
            uint32_t x8 = ((x6 << 13) | (x6 >> 19)) & 4294967295;
            *x9 = *x7 ^ x8;
            uint32_t x10 = (x2 + x4) & 4294967295;
            uint32_t x11 = ((x2 << 7) | (x2 >> 25)) & 4294967295;
            *x12 = x10 ^ x11;
            *x13 = ((x10 << 16) | (x10 >> 16)) & 4294967295;
        };
        static unsigned long get_num_valid_pairs(uint32_t input_diff[], uint32_t output_diff[], unsigned long pair_samples, unsigned int seed){
            if (seed == 0) srand((unsigned int) time (NULL));
            else srand(seed);
            unsigned long num_valid_pairs = 0;
            unsigned long i = 0;
            unsigned int j = 0;
            uint32_t input1[4];
            uint32_t input2[4];
            uint32_t output1[4];
            uint32_t output2[4];
            for (; i < pair_samples; ++i) {
                for (j = 0; j < 4; ++j) {
                    input1[j] = rand();
                    input2[j] = ( ((input1[j] << 1) | (input1[j] >> 31)) ^ input_diff[j] ) & 4294967295;
                }
                eval(input1[0], input1[1], input1[2], input1[3], &output1[0], &output1[1], &output1[2], &output1[3]);
                eval(input2[0], input2[1], input2[2], input2[3], &output2[0], &output2[1], &output2[2], &output2[3]);
                for (j = 0; j < 4; ++j) {
                    if ( (( ((output1[j] << 1) | (output1[j] >> 31)) ^ output2[j] ) & 4294967295) != output_diff[j] ) break;
                    if (j == 4 - 1) num_valid_pairs += 1;
                }
            }
            return num_valid_pairs;
        }

    """
    name_foo = "eval"

    width2type = {
        8: "uint8_t",
        16: "uint16_t",
        32: "uint32_t",
        64: "uint64_t"
    }

    input_vars = ssa["input_vars"]
    output_vars = ssa["output_vars"]

    input_vars_c = ["{} {}".format(width2type[v.width], v.name)  for v in input_vars]
    output_vars_c = ["{}* {}".format(width2type[v.width], v.name)  for v in output_vars]

    outvar2outvar_c = {v: type(v)("*" + v.name, v.width) for v in output_vars}

    eval_ccode = "void {}({}, {}){{\n".format(name_foo, ', '.join(input_vars_c), ', '.join(output_vars_c))
    for var, expr in ssa["assignments"]:
        expr = expr.xreplace(outvar2outvar_c)
        if var in output_vars:
            eval_ccode += "\t*{} = {};\n".format(var, bv2ccode(expr))
        else:
            eval_ccode += "\t{} {} = {};\n".format(width2type[var.width], var, bv2ccode(expr))
    eval_ccode += "};"

    assert all(input_vars[0].width == v.width for v in input_vars)
    assert all(output_vars[0].width == v.width for v in output_vars)

    if diff_type == difference.XorDiff:
        input_diff_ccode = "input1[j] ^ input_diff[j]"
        output_diff_ccode = "output1[j] ^ output2[j]"
    elif diff_type == difference.RXDiff:
        # operation.RotateLeft(a, 1) ^ b
        # # ROTL1(a) ((a << 1) | (a >> (width - 1))) & (2**width - 1)
        input_diff_ccode = "( ((input1[j] << 1) | (input1[j] >> {})) ^ input_diff[j] ) & {}".format(
            input_vars[0].width - 1, 2**input_vars[0].width - 1)
        output_diff_ccode = "( ((output1[j] << 1) | (output1[j] >> {})) ^ output2[j] ) & {}".format(
            output_vars[0].width - 1, 2**output_vars[0].width - 1)
    else:
        raise ValueError("invalid diff_type")

    get_num_valid_pairs_source_formatted = get_num_valid_pairs_source.format(
        ctype_input=width2type[input_vars[0].width],
        ctype_output=width2type[output_vars[0].width],
        len_input=len(input_vars),
        len_output=len(output_vars),
        eval1=name_foo,
        eval2=name_foo,
        input1=', '.join(["input1[{}]".format(i) for i in range(len(input_vars))]),
        output1=', '.join(["&output1[{}]".format(i) for i in range(len(output_vars))]),
        input2=', '.join(["input2[{}]".format(i) for i in range(len(input_vars))]),
        output2=', '.join(["&output2[{}]".format(i) for i in range(len(output_vars))]),
        input_diff_ccode=input_diff_ccode,
        output_diff_ccode=output_diff_ccode,
    )

    header_ccode = get_num_valid_pairs_header.format(
        ctype_input=width2type[input_vars[0].width],
        ctype_output=width2type[output_vars[0].width],
    )
    source_ccode = eval_ccode + get_num_valid_pairs_source_formatted

    return header_ccode, source_ccode


def relatedssa2ccode(ssa1, ssa2, diff_type):
    """Return the C code to compute a differential probability over a (related) function in ssa form.

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.primitives.chaskey import ChaskeyPi
        >>> from arxpy.smt.verification import relatedssa2ccode
        >>> ChaskeyPi.set_rounds(1)
        >>> ssa1 = ChaskeyPi.ssa(["v0", "v1", "v2", "v3"], "x")  # doctest: +NORMALIZE_WHITESPACE
        >>> ssa2 = dict(ssa1)
        >>> ssa2["assignments"] = list(ssa2["assignments"])
        >>> ssa2["assignments"].append([ssa2["output_vars"][0], ssa2["output_vars"][0]])
        >>> header, source = relatedssa2ccode(ssa1, ssa2, XorDiff)
        >>> print(header)
        static unsigned long get_num_valid_pairs(uint32_t input_diff[], uint32_t output_diff[], unsigned long pair_samples, unsigned int seed);
        >>> print(source)  # doctest:+NORMALIZE_WHITESPACE
        void eval1(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, uint32_t* x7, uint32_t* x12, uint32_t* x13, uint32_t* x9){
            uint32_t x0 = (v0 + v1) & 4294967295;
            uint32_t x1 = ((v1 << 5) | (v1 >> 27)) & 4294967295;
            uint32_t x2 = x0 ^ x1;
            uint32_t x3 = ((x0 << 16) | (x0 >> 16)) & 4294967295;
            uint32_t x4 = (v2 + v3) & 4294967295;
            uint32_t x5 = ((v3 << 8) | (v3 >> 24)) & 4294967295;
            uint32_t x6 = x4 ^ x5;
            *x7 = (x3 + x6) & 4294967295;
            uint32_t x8 = ((x6 << 13) | (x6 >> 19)) & 4294967295;
            *x9 = *x7 ^ x8;
            uint32_t x10 = (x2 + x4) & 4294967295;
            uint32_t x11 = ((x2 << 7) | (x2 >> 25)) & 4294967295;
            *x12 = x10 ^ x11;
            *x13 = ((x10 << 16) | (x10 >> 16)) & 4294967295;
        };
        void eval2(uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3, uint32_t* x7, uint32_t* x12, uint32_t* x13, uint32_t* x9){
            uint32_t x0 = (v0 + v1) & 4294967295;
            uint32_t x1 = ((v1 << 5) | (v1 >> 27)) & 4294967295;
            uint32_t x2 = x0 ^ x1;
            uint32_t x3 = ((x0 << 16) | (x0 >> 16)) & 4294967295;
            uint32_t x4 = (v2 + v3) & 4294967295;
            uint32_t x5 = ((v3 << 8) | (v3 >> 24)) & 4294967295;
            uint32_t x6 = x4 ^ x5;
            *x7 = (x3 + x6) & 4294967295;
            uint32_t x8 = ((x6 << 13) | (x6 >> 19)) & 4294967295;
            *x9 = *x7 ^ x8;
            uint32_t x10 = (x2 + x4) & 4294967295;
            uint32_t x11 = ((x2 << 7) | (x2 >> 25)) & 4294967295;
            *x12 = x10 ^ x11;
            *x13 = ((x10 << 16) | (x10 >> 16)) & 4294967295;
            *x7 = *x7;
        };
        static unsigned long get_num_valid_pairs(uint32_t input_diff[], uint32_t output_diff[], unsigned long pair_samples, unsigned int seed){
            if (seed == 0) srand((unsigned int) time (NULL));
            else srand(seed);
            unsigned long num_valid_pairs = 0;
            unsigned long i = 0;
            unsigned int j = 0;
            uint32_t input1[4];
            uint32_t input2[4];
            uint32_t output1[4];
            uint32_t output2[4];
            for (; i < pair_samples; ++i) {
                for (j = 0; j < 4; ++j) {
                    input1[j] = rand();
                    input2[j] = input1[j] ^ input_diff[j];
                }
                eval1(input1[0], input1[1], input1[2], input1[3], &output1[0], &output1[1], &output1[2], &output1[3]);
                eval2(input2[0], input2[1], input2[2], input2[3], &output2[0], &output2[1], &output2[2], &output2[3]);
                for (j = 0; j < 4; ++j) {
                    if ( (output1[j] ^ output2[j]) != output_diff[j] ) break;
                    if (j == 4 - 1) num_valid_pairs += 1;
                }
            }
            return num_valid_pairs;
        }

    """
    assert ssa1 != ssa2
    assert ssa1["input_vars"] == ssa2["input_vars"]
    assert ssa1["output_vars"] == ssa2["output_vars"]

    name_foo = "eval"  # fixed in verify_ccode

    width2type = {
        8: "uint8_t",
        16: "uint16_t",
        32: "uint32_t",
        64: "uint64_t"
    }

    input_vars = ssa1["input_vars"]
    output_vars = ssa1["output_vars"]

    input_vars_c = ["{} {}".format(width2type[v.width], v.name) for v in input_vars]
    output_vars_c = ["{}* {}".format(width2type[v.width], v.name) for v in output_vars]

    outvar2outvar_c = {v: type(v)("*" + v.name, v.width) for v in output_vars}

    list_eval_ccode = []
    for i, ssa in enumerate([ssa1, ssa2]):
        eval_ccode = "void {}({}, {}){{\n".format(name_foo + str(i+1), ', '.join(input_vars_c), ', '.join(output_vars_c))
        for var, expr in ssa["assignments"]:
            expr = expr.xreplace(outvar2outvar_c)
            if var in output_vars:
                eval_ccode += "\t*{} = {};\n".format(var, bv2ccode(expr))
            else:
                eval_ccode += "\t{} {} = {};\n".format(width2type[var.width], var, bv2ccode(expr))
        eval_ccode += "};"
        list_eval_ccode.append(eval_ccode)

    eval_ccode = "\n".join(list_eval_ccode)

    # no need to include <stdint.h>

    assert all(input_vars[0].width == v.width for v in input_vars)
    assert all(output_vars[0].width == v.width for v in output_vars)

    if diff_type == difference.XorDiff:
        input_diff_ccode = "input1[j] ^ input_diff[j]"
        output_diff_ccode = "output1[j] ^ output2[j]"
    elif diff_type == difference.RXDiff:
        input_diff_ccode = "( ((input1[j] << 1) | (input1[j] >> {})) ^ input_diff[j] ) & {}".format(
            input_vars[0].width - 1, 2**input_vars[0].width - 1)
        output_diff_ccode = "( ((output1[j] << 1) | (output1[j] >> {})) ^ output2[j] ) & {}".format(
            output_vars[0].width - 1, 2**output_vars[0].width - 1)
    else:
        raise ValueError("invalid diff_type")

    get_num_valid_pairs_source_formatted = get_num_valid_pairs_source.format(
        ctype_input=width2type[input_vars[0].width],
        ctype_output=width2type[output_vars[0].width],
        len_input=len(input_vars),
        len_output=len(output_vars),
        eval1=name_foo+"1",
        eval2=name_foo+"2",
        input1=', '.join(["input1[{}]".format(i) for i in range(len(input_vars))]),
        output1=', '.join(["&output1[{}]".format(i) for i in range(len(output_vars))]),
        input2=', '.join(["input2[{}]".format(i) for i in range(len(input_vars))]),
        output2=', '.join(["&output2[{}]".format(i) for i in range(len(output_vars))]),
        input_diff_ccode=input_diff_ccode,
        output_diff_ccode=output_diff_ccode,
    )

    header_ccode = get_num_valid_pairs_header.format(
        ctype_input=width2type[input_vars[0].width],
        ctype_output=width2type[output_vars[0].width],
    )
    source_ccode = eval_ccode + get_num_valid_pairs_source_formatted

    return header_ccode, source_ccode


def compile_run_empirical_weight(ccode, module_name, input_diff, output_diff, target_weight, verbose=False):
    """Compile and execute the C code to compute the empirical weight

        >>> from arxpy.differential.difference import XorDiff, RXDiff
        >>> from arxpy.primitives.chaskey import ChaskeyPi
        >>> from arxpy.smt.verification import ssa2ccode, compile_run_empirical_weight
        >>> ChaskeyPi.set_rounds(1)
        >>> ssa = ChaskeyPi.ssa(["v0", "v1", "v2", "v3"], "x")  # doctest: +NORMALIZE_WHITESPACE
        >>> header, source = ssa2ccode(ssa, XorDiff)
        >>> ind, outd, tw = [0, 0, 0, 0], [0, 0, 0, 0], 2
        >>> compile_run_empirical_weight([header, source], "_libverChaskeyPi", ind, outd, tw)
        0.0
        >>> ind, outd, tw = [0, 0, 0, 0], [0, 0, 0, 1], 2
        >>> compile_run_empirical_weight([header, source], "_libverChaskeyPi", ind, outd, tw)
        inf
        >>> header, source = ssa2ccode(ssa, RXDiff)
        >>> ind, outd, tw = [0, 0, 0, 0], [0, 0, 0, 0], 8
        >>> 4 <= compile_run_empirical_weight([header, source], "_libverChaskeyPi", ind, outd, tw) <= 8
        True
        >>> ind, outd, tw = [0, 0, 0, 0], [1, 1, 1, 1], 8
        >>> compile_run_empirical_weight([header, source], "_libverChaskeyPi", ind, outd, tw)
        inf

    """

    ffibuilder = cffi.FFI()
    ffibuilder.cdef(ccode[0])
    ffibuilder.set_source(module_name, ccode[1])

    with tempfile.TemporaryDirectory() as tmpdirname:
        libver_path = ffibuilder.compile(tmpdir=tmpdirname, verbose=verbose)

        spec = importlib.util.spec_from_file_location(module_name, libver_path)
        libver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(libver)

        pair_samples = max(5 * (2**(int(target_weight) + 1)), 100)

        seed = tuple(input_diff + output_diff + [pair_samples])

        num_pairs = libver.lib.get_num_valid_pairs(input_diff, output_diff, pair_samples, hash(seed) % (2**32))

    return math.inf if num_pairs == 0 else abs(-math.log2(num_pairs * 1.0 / pair_samples))


def fast_empirical_weight(ch_found, verbose_lvl=0, debug=False, filename=None):
    """Computes the empirical weight of the model using C code.

    If ``filename`` is not ``None``, the output will be printed
    to the given file rather than the to stdout.

    The argument ``verbose_lvl`` can take an integer between
    ``0`` (no verbose) and ``3`` (full verbose).

        >>> from arxpy.differential.difference import XorDiff, RXDiff
        >>> from arxpy.differential.characteristic import BvCharacteristic
        >>> from arxpy.primitives.chaskey import ChaskeyPi
        >>> from arxpy.smt.search import SearchCh
        >>> from arxpy.smt.verification import fast_empirical_weight
        >>> ChaskeyPi.set_rounds(2)
        >>> ch = BvCharacteristic(ChaskeyPi, XorDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> search_problem = SearchCh(ch)
        >>> ch_found = search_problem.solve(0)
        >>> ch_found.ch_weight
        0x04
        >>> 3 <= fast_empirical_weight(ch_found) <= 5
        True
        >>> ChaskeyPi.set_rounds(1)
        >>> ch = BvCharacteristic(ChaskeyPi, RXDiff, ["dv0", "dv1", "dv2", "dv3"])
        >>> ic = [operation.BvComp(0, d.val) for d in ch.input_diff]
        >>> ic += [operation.BvComp(0, d[1].val) for d in ch.output_diff]
        >>> ch_found = SearchCh(ch, allow_zero_input_diff=True, initial_constraints=ic).solve(5)
        >>> ch_found.ch_weight
        0x05
        >>> 4 - 1 <= fast_empirical_weight(ch_found) <= 8
        True

    """
    from arxpy.smt.search import _get_smart_print  # avoid cyclic imports

    smart_print = _get_smart_print(filename)

    exact_weight = ch_found.get_exact_weight()

    if debug:
        smart_print("Symbolic characteristic:")
        smart_print(ch_found.ch)
        smart_print("Characteristic found:")
        smart_print(ch_found)
        smart_print()

    der_weights = []
    for i, (diff, der) in enumerate(ch_found.ch.nonlinear_diffs.items()):
        actual_diff = ch_found.nonlinear_diffs[i][1]
        new_input_diff = [(d.xreplace(ch_found._diff_model)) for d in der.input_diff]
        der_weights.append(der._replace_input_diff(new_input_diff).exact_weight(actual_diff))

    max_subch_weight = exact_weight if exact_weight < MAX_WEIGHT else exact_weight / (exact_weight / MAX_WEIGHT)
    max_subch_weight = max(1, max_subch_weight, *der_weights)
    if debug:
        smart_print("max_subch_weight:", max_subch_weight)
        smart_print()

    subch_listdiffder = [[]]  # for each subch, a list of [diff, der] pairs
    subch_index = 0
    current_subch_weight = 0   # exact_weight
    subch_weight = [] # the weight of each subch
    assert len(ch_found.ch.nonlinear_diffs.items()) > 0
    for i, (diff, der) in enumerate(ch_found.ch.nonlinear_diffs.items()):
        der_weight = der_weights[i]
        if current_subch_weight + der_weight > max_subch_weight:
            subch_weight.append(current_subch_weight)
            current_subch_weight = 0
            subch_index += 1
            subch_listdiffder.append([])
        current_subch_weight += der_weight
        subch_listdiffder[subch_index].append([diff, der])

    subch_weight.append(current_subch_weight)
    assert len(subch_weight) == len(subch_listdiffder)

    num_subch = len(subch_listdiffder)

    if verbose_lvl >= 3:
        smart_print("- characteristic decomposed into {} subcharacteristics with exact weights {}".format(
            num_subch, subch_weight
        ))

    def subch_listdiffder2subch_ssa(listdiffder, first_var_next_subch, first_subch=False):
        first_var = listdiffder[0][0].val

        input_vars = []
        inter_vars = set()
        assignments = []
        add_assignment = first_subch
        for var, expr in ch_found.ch.ssa["assignments"]:
            if var == first_var:
                add_assignment = True
            elif var == first_var_next_subch:
                break

            if add_assignment:
                input_vars.extend([atom for atom in expr.atoms(core.Variable) if atom not in input_vars])
                inter_vars.add(var)
                assignments.append([var, expr])

        subch_ssa = {}
        subch_ssa["input_vars"] = [var for var in input_vars if var not in inter_vars]
        subch_ssa["output_vars"] = []
        subch_ssa["inter_vars"] = inter_vars
        subch_ssa["assignments"] = assignments
        return subch_ssa

    subch_ssa = [None for _ in range(num_subch)]
    for i in reversed(range(num_subch)):
        if i == num_subch - 1:
            subch_ssa[i] = subch_listdiffder2subch_ssa(subch_listdiffder[i], None, i == 0)
            subch_ssa[i]["output_vars"] = list(ch_found.ch.ssa["output_vars"])
        else:
            first_var_next_ssa = subch_listdiffder[i + 1][0][0].val
            subch_ssa[i] = subch_listdiffder2subch_ssa(subch_listdiffder[i], first_var_next_ssa, i == 0)
            subch_ssa[i]["output_vars"] = subch_ssa[i + 1]["input_vars"][:]

        for diff_var in subch_ssa[i]["output_vars"]:
            if diff_var not in subch_ssa[i]["inter_vars"] and diff_var not in subch_ssa[i]["input_vars"]:
                subch_ssa[i]["input_vars"].append(diff_var)

        del subch_ssa[i]["inter_vars"]
        subch_ssa[i]["weight"] = subch_weight[i]

    var2diffval = {}
    for diff_var, diff_value in itertools.chain(ch_found.input_diff, ch_found.nonlinear_diffs, ch_found.output_diff):
        var2diffval[diff_var.val] = diff_value.val
    for var, diff in ch_found.ch._var2diff.items():
        if var not in var2diffval:
            var2diffval[var] = diff.val.xreplace(var2diffval)

    # fixing duplicate var problem
    for ssa in subch_ssa:
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

    total_empirical_weight = 0
    for i, ssa in enumerate(subch_ssa):
        ccode = ssa2ccode(ssa, ch_found.ch.diff_type)

        if verbose_lvl >= 2:
            smart_print("- sub-characteristic {}".format(i))
        if verbose_lvl >= 3:
            smart_print("  - ssa:", subch_ssa[i])  # pprint.pformat(list_ssa[i], width=100))
            smart_print("  - listdiffder:", subch_listdiffder[i])  # pprint.pformat(ssa_ders[i], width=100))
        if debug:
            smart_print(ccode[0])
            smart_print(ccode[1])
            smart_print()

        input_diff_c = [v.xreplace(var2diffval) for v in ssa["input_vars"]]
        output_diff_c = [v.xreplace(var2diffval) for v in ssa["output_vars"]]

        if verbose_lvl >= 2:
            smart_print("  - checking {} -> {} with weight {}".format(
                '|'.join([str(d) for d in input_diff_c]), '|'.join([str(d) for d in output_diff_c]),
                ssa["weight"]))

        input_diff_c = [int(d.val) for d in input_diff_c]
        output_diff_c = [int(d.val) for d in output_diff_c]

        assert all(isinstance(d, (int, core.Constant)) for d in input_diff_c), "{}".format(input_diff_c)
        assert all(isinstance(d, (int, core.Constant)) for d in output_diff_c), "{}".format(output_diff_c)

        current_empirical_weight = compile_run_empirical_weight(
            ccode,
            "_libver" + ch_found.ch.func.__name__ + str(i),
            input_diff_c,
            output_diff_c,
            ssa["weight"],
            verbose=verbose_lvl >= 4)

        if verbose_lvl >= 2:
            smart_print("  - exact/empirical weight: {}, {}".format(ssa["weight"], current_empirical_weight))

        if current_empirical_weight == math.inf:
            return math.inf

        total_empirical_weight += current_empirical_weight

    return total_empirical_weight


def _fast_empirical_weight_distribution(ch_found, cipher, rk_dict_diffs=None,
                                        verbose_lvl=0, debug=False, filename=None, precision=0):
    """

        >>> from arxpy.differential.difference import XorDiff
        >>> from arxpy.differential.characteristic import SingleKeyCh
        >>> from arxpy.smt.search import SearchSkCh
        >>> from arxpy.primitives import speck
        >>> from arxpy.smt.verification import _fast_empirical_weight_distribution
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> Speck32.set_rounds(1)
        >>> ch = SingleKeyCh(Speck32, XorDiff)
        >>> search_problem = SearchSkCh(ch)
        >>> ch_found = search_problem.solve(0)
        >>> _fast_empirical_weight_distribution(ch_found, Speck32)
        Counter({0: 256})

    """
    # similar to _empirical_distribution_weight of characteristic module

    from arxpy.smt.search import _get_smart_print  # avoid cyclic imports

    smart_print = _get_smart_print(filename)

    exact_weight = ch_found.get_exact_weight()

    if rk_dict_diffs is not None:
        assert "nonlinear_diffs" in rk_dict_diffs and "output_diff" in rk_dict_diffs

    if debug:
        smart_print("Symbolic characteristic:")
        smart_print(ch_found.ch)
        smart_print("Characteristic found:")
        smart_print(ch_found)
        if rk_dict_diffs is not None:
            smart_print("rk_dict_diffs:", rk_dict_diffs)
        smart_print()

    der_weights = []
    for i, (diff, der) in enumerate(ch_found.ch.nonlinear_diffs.items()):
        actual_diff = ch_found.nonlinear_diffs[i][1]
        new_input_diff = [(d.xreplace(ch_found._diff_model)) for d in der.input_diff]
        der_weights.append(der._replace_input_diff(new_input_diff).exact_weight(actual_diff))

    max_subch_weight = exact_weight if exact_weight < MAX_WEIGHT else exact_weight / (exact_weight / MAX_WEIGHT)
    max_subch_weight = max(1, max_subch_weight, *der_weights)
    if debug:
        smart_print("max_subch_weight:", max_subch_weight)
        smart_print()

    subch_listdiffder = [[]]  # for each subch, a list of [diff, der] pairs
    subch_index = 0
    current_subch_weight = 0  # exact_weight
    subch_weight = []  # the weight of each subch
    assert len(ch_found.ch.nonlinear_diffs.items()) > 0
    for i, (diff, der) in enumerate(ch_found.ch.nonlinear_diffs.items()):
        der_weight = der_weights[i]
        if current_subch_weight + der_weight > max_subch_weight:
            subch_weight.append(current_subch_weight)
            current_subch_weight = 0
            subch_index += 1
            subch_listdiffder.append([])
        current_subch_weight += der_weight
        subch_listdiffder[subch_index].append([diff, der])

    subch_weight.append(current_subch_weight)
    assert len(subch_weight) == len(subch_listdiffder)

    num_subch = len(subch_listdiffder)

    if verbose_lvl >= 3:
        smart_print("- characteristic decomposed into {} subcharacteristics with exact weights {}".format(
            num_subch, subch_weight
        ))

    if rk_dict_diffs is not None:
        rk_var = [var.val for var, _ in rk_dict_diffs["output_diff"]]
    else:
        rk_var = []
        for i, width in enumerate(cipher.key_schedule.output_widths):
            rk_var.append(core.Variable("k" + str(i), width))

    var2diffval = {}
    for diff_var, diff_value in itertools.chain(ch_found.input_diff, ch_found.nonlinear_diffs, ch_found.output_diff):
        var2diffval[diff_var.val] = diff_value.val
    if rk_dict_diffs is not None:
        for var, diff in rk_dict_diffs["output_diff"]:
            var2diffval[var.val] = diff.val
        for var, diff in rk_dict_diffs["nonlinear_diffs"]:
            var2diffval[var.val] = diff.val
    for var, diff in ch_found.ch._var2diff.items():
        if var not in var2diffval:
            if isinstance(diff, core.Term):
                # e.g., symbolic computations with the key
                var2diffval[var] = diff.xreplace(var2diffval)
            else:
                var2diffval[var] = diff.val.xreplace(var2diffval)

    # for each related-key pair, we associated a pair of subch_ssa
    rkey2pair_subchssa = [None for _ in range(KEY_SAMPLES)]
    for key_index in range(KEY_SAMPLES):
        master_key = []
        for width in cipher.key_schedule.input_widths:
            master_key.append(core.Constant(random.randrange(2 ** width), width))
        rk_val = cipher.key_schedule(*master_key)
        if rk_dict_diffs is not None:
            rk_other_val = tuple([d.get_pair_element(r) for r, (_, d) in zip(rk_val, rk_dict_diffs["output_diff"])])
        else:
            rk_other_val = rk_val
        assert len(rk_var) == len(rk_other_val)
        assert all(isinstance(rk, core.Constant) for rk in rk_val)
        assert all(isinstance(rk, core.Constant) for rk in rk_other_val)

        def subch_listdiffder2subch_ssa(listdiffder, first_der_var_next_subch, var2val, first_subch=False):
            first_der_var = listdiffder[0][0].val

            input_vars = []
            inter_vars = set()
            assignments = []
            add_assignment = first_subch
            for var, expr in ch_found.ch.ssa["assignments"]:
                if var == first_der_var:
                    add_assignment = True
                elif var == first_der_var_next_subch:
                    break

                expr = expr.xreplace(var2val)

                if add_assignment:
                    input_vars.extend([atom for atom in expr.atoms(core.Variable) if atom not in input_vars])
                    inter_vars.add(var)
                    assignments.append([var, expr])

            subch_ssa = {}
            subch_ssa["input_vars"] = [var for var in input_vars if var not in inter_vars]
            subch_ssa["output_vars"] = []
            subch_ssa["inter_vars"] = inter_vars
            subch_ssa["assignments"] = assignments
            return subch_ssa

        pair_subchssa = []
        for index_pair in range(2):
            current_rk_val = rk_val if index_pair == 0 else rk_other_val
            rkvar2rkval = {var: val for var, val in zip(rk_var, current_rk_val)}
            subch_ssa = [None for _ in range(num_subch)]
            for i in reversed(range(num_subch)):
                if i == num_subch - 1:
                    subch_ssa[i] = subch_listdiffder2subch_ssa(subch_listdiffder[i], None, rkvar2rkval, i == 0)
                    subch_ssa[i]["output_vars"] = list(ch_found.ch.ssa["output_vars"])
                else:
                    first_var_next_ssa = subch_listdiffder[i + 1][0][0].val
                    subch_ssa[i] = subch_listdiffder2subch_ssa(subch_listdiffder[i], first_var_next_ssa, rkvar2rkval, i == 0)
                    subch_ssa[i]["output_vars"] = subch_ssa[i + 1]["input_vars"][:]

                for var in subch_ssa[i]["output_vars"]:
                    if var not in subch_ssa[i]["inter_vars"] and var not in subch_ssa[i]["input_vars"]:
                        subch_ssa[i]["input_vars"].append(var)

                del subch_ssa[i]["inter_vars"]
                subch_ssa[i]["weight"] = subch_weight[i]

            for _, ssa in enumerate(subch_ssa):
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

            pair_subchssa.append(subch_ssa)

        assert len(pair_subchssa[0]) == len(pair_subchssa[1]) == num_subch
        rkey2pair_subchssa[key_index] = pair_subchssa

    # for each related-key pair, we associated the list of the weight of each subch
    rkey2subch_ew = [[0 for _ in range(num_subch)] for _ in range(KEY_SAMPLES)]

    # start multiprocessing
    with multiprocessing.Pool() as pool:
        for i in range(num_subch):
            for key_index in range(KEY_SAMPLES):
                ssa1 = rkey2pair_subchssa[key_index][0][i]
                ssa2 = rkey2pair_subchssa[key_index][1][i]

                if key_index <= 1:
                    if verbose_lvl >= 2 and key_index == 0:
                        smart_print("- sub-characteristic {}".format(i))
                    if verbose_lvl >= 3 and key_index == 0:
                        smart_print("  - listdiffder:", subch_listdiffder[i])
                    if verbose_lvl >= 3:
                        smart_print("  - related-key pair index", key_index)
                        smart_print("  - ssa1:", ssa1)
                        if ssa1 == ssa2:
                            smart_print("  - ssa2: (same as ssa1)")
                        else:
                            smart_print("  - ssa2:", ssa2)

                if i > 0 and rkey2subch_ew[key_index][i-1] == math.inf:
                    rkey2subch_ew[key_index][i] = math.inf
                    if key_index <= 1 and verbose_lvl >= 2:
                        smart_print("  - rk{} | skipping since invalid sub-ch[{}]".format(key_index, i-1))
                    continue

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
                    smart_print("  - rk{} | checking {} -> {} with weight {}".format(
                        key_index,
                        '|'.join([str(d) for d in input_diff_c]), '|'.join([str(d) for d in output_diff_c]),
                        ssa1["weight"]))

                assert ssa1["weight"] == ssa2["weight"]

                assert all(isinstance(d, (int, core.Constant)) for d in input_diff_c), "{}".format(input_diff_c)
                assert all(isinstance(d, (int, core.Constant)) for d in output_diff_c), "{}".format(output_diff_c)

                input_diff_c = [int(d) for d in input_diff_c]
                output_diff_c = [int(d) for d in output_diff_c]

                rkey2subch_ew[key_index][i] = pool.apply_async(
                    compile_run_empirical_weight,
                    (
                        ccode,
                        "_libver" + ch_found.ch.func.__name__ + str(i),
                        input_diff_c,
                        output_diff_c,
                        ssa1["weight"],
                        False
                    )
                )

            # wait until all i-subch have been compiled and run
            # and replace the Async object by the result
            for key_index in range(KEY_SAMPLES):
                if isinstance(rkey2subch_ew[key_index][i], multiprocessing.pool.AsyncResult):
                    rkey2subch_ew[key_index][i] = rkey2subch_ew[key_index][i].get()

                if key_index <= 1 and verbose_lvl >= 2:
                    smart_print("  - rk{} | exact/empirical weight: {}, {}".format(
                        key_index, subch_weight[i], rkey2subch_ew[key_index][i]))

    # end multiprocessing

    empirical_weight_distribution = collections.Counter()
    all_rkey_weights = []
    for key_index in range(KEY_SAMPLES):
        rkey_weight = sum(rkey2subch_ew[key_index])

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

