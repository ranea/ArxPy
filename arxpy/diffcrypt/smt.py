"""The SMT module creates and solve SMT problems of characteristic weights."""
import collections
import functools
import itertools

from pysmt import logics
from pysmt import shortcuts as sc
from pysmt import typing

from arxpy.bitvector import core
from arxpy.bitvector import operation

from arxpy.diffcrypt import characteristic
from arxpy.diffcrypt import difference
from arxpy.diffcrypt import differential


def bv2pysmt(bv):
    """Convert a bit-vector type to a pySMT type.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.diffcrypt.smt import bv2pysmt
        >>> bv2pysmt(Constant(0b00000001, 8))
        1_8
        >>> x, y = Variable("x", 8), Variable("y", 8)
        >>> bv2pysmt(x)
        x
        >>> bv2pysmt(x +  y)
        (x + y)
        >>> bv2pysmt(x <=  y)
        (x u<= y)
        >>> bv2pysmt(x[4: 2])
        x[2:4]

    """
    msg = "unknown conversion of {} to a pySMT type".format(type(bv).__name__)

    if isinstance(bv, int):
        return bv

    if isinstance(bv, core.Variable):
        return sc.Symbol(bv.name, typing.BVType(bv.width))

    if isinstance(bv, core.Constant):
        return sc.BV(bv.val, bv.width)

    if isinstance(bv, operation.Operation):
        args = [bv2pysmt(a) for a in bv.args]

        if type(bv) == operation.BvNot:
            if args[0].is_equals():
                return sc.Not(*args)
            else:
                return sc.BVNot(*args)

        if type(bv) == operation.BvAnd:
            return sc.BVAnd(*args)

        if type(bv) == operation.BvOr:
            return sc.BVOr(*args)

        if type(bv) == operation.BvXor:
            return sc.BVXor(*args)

        if type(bv) == operation.BvComp:
            # return sc.BVComp(*args)
            return sc.Equals(*args)

        if type(bv) == operation.BvUlt:
            return sc.BVULT(*args)

        if type(bv) == operation.BvUle:
            return sc.BVULE(*args)

        if type(bv) == operation.BvUgt:
            return sc.BVUGT(*args)

        if type(bv) == operation.BvUge:
            return sc.BVUGE(*args)

        if type(bv) == operation.BvShl:
            # Left hand side width must be a power of 2
            if (args[0].bv_width() & (args[0].bv_width() - 1)) == 0:
                return sc.BVLShl(*args)
            else:
                x, r = bv.args
                offset = 0
                while (x.width & (x.width - 1)) != 0:
                    x = operation.ZeroExtend(x, 1)
                    r = operation.ZeroExtend(r, 1)
                    offset += 1

                shift = bv2pysmt(x << r)
                return sc.BVExtract(shift, end=shift.bv_width() - offset - 1)
            # width = args[0].bv_width()
            # assert (width & (width - 1)) == 0  # power of 2
            # return sc.BVLShl(*args)

        if type(bv) == operation.BvLshr:
            # Left hand side width must be a power of 2
            if (args[0].bv_width() & (args[0].bv_width() - 1)) == 0:
                return sc.BVLShr(*args)
            else:
                x, r = bv.args
                offset = 0
                while (x.width & (x.width - 1)) != 0:
                    x = operation.ZeroExtend(x, 1)
                    r = operation.ZeroExtend(r, 1)
                    offset += 1

                shift = bv2pysmt(x >> r)
                return sc.BVExtract(shift, end=shift.bv_width() - offset - 1)
            # width = args[1].bv_width()
            # assert (width & (width - 1)) == 0  # power of 2
            # return sc.BVLShr(*args)

        if type(bv) == operation.RotateLeft:
            # Left hand side width must be a power of 2
            if (args[0].bv_width() & (args[0].bv_width() - 1)) == 0:
                return sc.BVRol(*args)
            else:
                x, r = bv.args
                n = x.width
                return bv2pysmt(operation.Concat(x[n - r - 1:], x[n - 1: n - r]))

        if type(bv) == operation.RotateRight:
            # Left hand side width must be a power of 2
            if (args[0].bv_width() & (args[0].bv_width() - 1)) == 0:
                return sc.BVRor(*args)
            else:
                x, r = bv.args
                n = x.width
                return bv2pysmt(operation.Concat(x[r - 1:], x[n - 1: r]))

        if type(bv) == operation.Ite:
            if args[0].is_equals():
                a0 = args[0]
            else:
                a0 = sc.Equals(args[0], bv2pysmt(core.Constant(1, 1)))

            return sc.Ite(a0, *args[1:])

        if type(bv) == operation.Extract:
            return sc.BVExtract(args[0], args[2], args[1])

        if type(bv) == operation.Concat:
            return sc.BVConcat(*args)

        if type(bv) == operation.ZeroExtend:
            return sc.BVZExt(*args)

        if type(bv) == operation.Repeat:
            return args[0].BVRepeat(args[1])

        if type(bv) == operation.BvNeg:
            return sc.BVNeg(*args)

        if type(bv) == operation.BvAdd:
            return sc.BVAdd(*args)

        if type(bv) == operation.BvSub:
            return sc.BVSub(*args)

        if type(bv) == operation.BvMul:
            return sc.BVMul(*args)

        if type(bv) == operation.BvMul:
            return sc.BVMul(*args)

        if type(bv) == operation.BvUdiv:
            return sc.BVUDiv(*args)

        if type(bv) == operation.BvUrem:
            return sc.BVURem(*args)

        raise NotImplementedError(msg)


def pysmt2bv(ps):
    """Transform a pySMT type to a bit-vector type.

    Currently, only conversion from bv_constants and symbols are supported.

        >>> from arxpy.diffcrypt.smt import sc, typing
        >>> pysmt2bv(sc.Symbol("x", typing.BVType(8))).vrepr()
        "DiffVar('x', width=8)"
        >>> pysmt2bv(sc.BV(1, 8)).vrepr()
        'Constant(0b00000001, width=8)'

    """
    class_name = type(ps).__name__
    msg = "unknown conversion of {} to a bit-vector type".format(class_name)

    if ps.is_symbol():
        return difference.DiffVar(ps.symbol_name(), ps.bv_width())

    if ps.is_bv_constant():
        return core.Constant(ps.constant_value(), ps.bv_width())

    raise NotImplementedError(msg)


class SmtProblem(object):
    """Create and solve SMT problems related to characteristic weights.

    Given a characteristic of a bit-vector function, SmtProblem creates
    the decision problem of whether the weight of a characteristic is equal
    to a given weight (called target_weight). It is also possible
    to check whether the weight is below the target weight by setting
    equality=False.

        >>> from arxpy.bitvector.function import Function
        >>> from arxpy.diffcrypt.difference import XorDiff, DiffVar
        >>> from arxpy.diffcrypt.characteristic import Characteristic
        >>> from arxpy.diffcrypt.smt import SmtProblem
        >>> class MyFunction(Function):
        ...     input_widths = [8, 8, 8]
        ...     output_widths = [8, 8]
        ...     @classmethod
        ...     def eval(cls, x, y, k):
        ...         return (y + k, (y + k) ^ x)
        >>> x, y, k = DiffVar("x", 8), DiffVar("y", 8), DiffVar("k", 8)
        >>> ch = Characteristic(MyFunction, XorDiff, [x, y, k])
        >>> smt_problem = SmtProblem(ch, 0)
        >>> print(smt_problem)  # doctest:+ELLIPSIS
        assert ~(0x000000 == ((x ∘ y) ∘ k))
        assert 0x00 == (((~(k << 0x01) ^ (d0 << 0x01)) & ...
        assert w_ky_d0 == (((0x0f & ((0x33 & ((0x55 & ...
        assert w_ky_d0 == w_xyk_d0d1
        assert 0x0 == w_xyk_d0d1

    SmtProblems of composite characteristics are handled by CompositeSmtProblem.
    """

    def __init__(self, ch, target_weight, equality=True, parent_ch=None):
        """Initialize the SMT problem."""
        assert isinstance(ch, characteristic.Characteristic)
        assert target_weight < sum(ch.func.input_widths)

        self.ch = ch
        self.parent_ch = parent_ch
        self.target_weight = target_weight
        self.equality = equality
        self.assertions = None

        self._generate()

    def _generate(self):
        """Generate the SMT problem."""
        self.assertions = []

        # Forbid zero input difference with XOR difference

        if self.ch.diff_type == difference.XorDiff:
            if self.parent_ch is not None and self.parent_ch.outer_ch == self.ch:
                inner_noutputs = len(self.parent_ch.inner_ch.output_diff)
                non_zero_input_diff = self.ch.input_diff[:-inner_noutputs]
            else:
                non_zero_input_diff = self.ch.input_diff
            non_zero_input_diff = functools.reduce(operation.Concat,
                                                   non_zero_input_diff)
            zero = core.Constant(0, non_zero_input_diff.width)
            self.assertions.append(
                operation.BvNot(operation.BvComp(non_zero_input_diff, zero))
            )

        # Assertions of the weights of the non-deterministic steps

        self.op_weights = []
        for var, propagation in self.ch.items():
            if isinstance(propagation, differential.Differential):
                self.assertions.append(propagation.is_valid())
                weight_value = propagation.weight()
                weight_var = core.Variable(propagation._weight_var_name(),
                                           weight_value.width)
                self.assertions.append(operation.BvComp(weight_var, weight_value))
                self.op_weights.append(weight_var)
            else:
                self.assertions.append(operation.BvComp(var, propagation))

        # Characteristic weight assignment

        max_value = 0
        for ow in self.op_weights:
            max_value += (2 ** ow.width) - 1
        width = max(max_value.bit_length(), 1)  # for trivial characteristic
        ext_op_weights = []
        for ow in self.op_weights:
            ext_op_weights.append(operation.ZeroExtend(ow, width - ow.width))

        name_ch_weight = "w_{}_{}".format(
            ''.join([str(i) for i in self.ch.input_diff]),
            ''.join([str(i) for i in self.ch.output_diff]))
        ch_weight = core.Variable(name_ch_weight, width)

        self.assertions.append(operation.BvComp(ch_weight, sum(ext_op_weights)))

        # Condition between the weight and the target weight

        weight_function = self.ch.get_weight_function()
        target_weight = int(weight_function(self.target_weight))

        width = max(ch_weight.width, target_weight.bit_length())
        self.ch_weight = operation.ZeroExtend(ch_weight, width - ch_weight.width)

        if self.equality:
            self.assertions.append(operation.BvComp(self.ch_weight, target_weight))
        else:
            self.assertions.append(operation.BvUlt(self.ch_weight, target_weight))

        self.assertions = tuple(self.assertions)

    @property
    def formula_size(self):
        """The size of the underlying bit-vector formula."""
        size = 0
        for a in self.assertions:
            size += a.formula_size

        for d in itertools.chain(self.ch.sequence, self.op_weights):
            size += d.formula_size

        size += self.ch_weight.formula_size

        return size

    @property
    def pysmt_formula_size(self):
        """The size of the underlying bit-vector formula according to pySMT."""
        sc.reset_env()
        pysmt_formula = sc.And(*[bv2pysmt(a) for a in self.assertions])

        return sc.get_formula_size(pysmt_formula)

    def __str__(self):
        representation = ""
        for assertion in self.assertions:
            representation += "assert {}\n".format(assertion)
        return representation[:-1]  # remove \n

    def to_smtlib(self):
        """Return a SMT_LIB string representation of the formula.

            >>> from arxpy.bitvector.function import Function
            >>> from arxpy.diffcrypt.difference import XorDiff, DiffVar
            >>> from arxpy.diffcrypt.characteristic import Characteristic
            >>> from arxpy.diffcrypt.smt import SmtProblem
            >>> class MyFunction(Function):
            ...     input_widths = [8, 8, 8]
            ...     output_widths = [8, 8]
            ...     @classmethod
            ...     def eval(cls, x, y, k):
            ...         return (y + k, (y + k) ^ x)
            >>> x, y, k = DiffVar("x", 8), DiffVar("y", 8), DiffVar("k", 8)
            >>> ch = Characteristic(MyFunction, XorDiff, [x, y, k])
            >>> smt_problem = SmtProblem(ch, 0)
            >>> print(smt_problem.to_smtlib())  # doctest:+ELLIPSIS
            (and (not (= #b000000000000000000000000 (concat (concat x y) k))) ...

        Note that a more human-readable from can be obtained by
        printing the SmtProblem directly (print(smt_problem)).
        """
        sc.reset_env()
        pysmt_formula = sc.And(*[bv2pysmt(a) for a in self.assertions])
        return sc.to_smtlib(pysmt_formula, daggify=False)

    def _get_assignment(self, model):
        assig = {}
        assig["differences"] = collections.OrderedDict()
        for d in self.ch.sequence:
            assig["differences"][d] = pysmt2bv(model[bv2pysmt(d)])

        inv = self.ch.get_inverse_weight_function()
        assig["weight"] = inv(int(pysmt2bv(model[bv2pysmt(self.ch_weight)])))

        assig["op_weights"] = collections.OrderedDict()
        for ow in self.op_weights:
            assig["op_weights"][ow] = inv(int(pysmt2bv(model[bv2pysmt(ow)])))

        return assig

    def solve(self, solver_name=None, get_assignment=False):
        """Solve the SMT problem.

        Return whether the decision problem is satisfiable. If get_assignment
        is set to True, solve() returns an assignment of the variables
        that makes the SMT problem satisfiable (if the problem is
        unsatisfiable, None is returned).

        This assignment is returned as a dictionary with the following entries:

        - differences: an ordered dictionary containing the sequence of
          differences.
        - weight: the weight of the characteristic.
        - op_weights: the weights of each operation with non-deterministic
          propagation.

            >>> from arxpy.bitvector.function import Function
            >>> from arxpy.diffcrypt.difference import XorDiff, DiffVar
            >>> from arxpy.diffcrypt.characteristic import Characteristic
            >>> from arxpy.diffcrypt.smt import SmtProblem
            >>> class MyFunction(Function):
            ...     input_widths = [8, 8, 8]
            ...     output_widths = [8, 8]
            ...     @classmethod
            ...     def eval(cls, x, y, k):
            ...         return (y + k, (y + k) ^ x)
            >>> x, y, k = DiffVar("x", 8), DiffVar("y", 8), DiffVar("k", 8)
            >>> ch = Characteristic(MyFunction, XorDiff, [x, y, k])
            >>> smt_problem = SmtProblem(ch, 0)
            >>> smt_problem.solve()
            True
            >>> smt_problem.solve(get_assignment=True)  # doctest:+NORMALIZE_WHITESPACE
            {'differences': OrderedDict([(x, 0x80), (y, 0x00), (k, 0x80),
            (d0, 0x80), (d1, 0x00)]), 'weight': 0,
            'op_weights': OrderedDict([(w_ky_d0, 0)])}

        """
        sc.reset_env()
        pysmt_formula = sc.And(*[bv2pysmt(a) for a in self.assertions])

        if not get_assignment:
            return sc.is_sat(pysmt_formula, solver_name, logic=logics.QF_BV)
        else:
            model = sc.get_model(pysmt_formula, solver_name, logic=logics.QF_BV)

            if model is None:
                return None
            else:
                return self._get_assignment(model)


class CompositeSmtProblem(object):
    """Create and solve SMT problems related to composite characteristic weights.

    Given a composite characteristic, CompositeSmtProblem creates
    the pair of SMT problem related to the weight of the inner characteristic
    and the weight of the outer characteristic. See SmtProblem
    for more information.

        >>> from arxpy.bitvector.operation import RotateLeft
        >>> from arxpy.bitvector.function import Function, CompositeFunction
        >>> from arxpy.diffcrypt.difference import XorDiff, DiffVar
        >>> from arxpy.diffcrypt.characteristic import Characteristic, CompositeCh
        >>> class MyInner(Function):
        ...     input_widths = [8]
        ...     output_widths = [8, 8]
        ...     @classmethod
        ...     def eval(cls, k):
        ...         return (k, RotateLeft(k, 1))
        >>> class MyOuter(Function):
        ...     input_widths = [8, 8, 8, 8]
        ...     output_widths = [8, 8]
        ...     @classmethod
        ...     def eval(cls, x, y, k0, k1):
        ...         for ki in [k0, k1]:
        ...             x, y = y + ki, (y + ki) ^ x
        ...         return x, y
        >>> class MyComposite(CompositeFunction):
        ...     input_widths = [8, 8, 8]
        ...     output_widths = [8, 8]
        ...     inner_func = MyInner
        ...     outer_func = MyOuter
        >>> x, y, k = DiffVar("x", 8), DiffVar("y", 8), DiffVar("k", 8)
        >>> ch = CompositeCh(MyComposite, XorDiff, [x, y, k])
        >>> smt_problem = CompositeSmtProblem(ch, [0, 0])
        >>> print(smt_problem)  # doctest:+ELLIPSIS
        assert ~(0x00 == k)
        assert i0 == (k <<< 1)
        assert 0b0 == w_k_ki0
        assert w_k_ki0 < 0b0
        <BLANKLINE>
        assert ~(0x0000 == (x ∘ y))
        assert 0x00 == (((~(k << 0x01) ^ (o0 << 0x01)) & (~(k << 0x01) ^ ...
        assert w_ky_o0 == (((0x0f & ((0x33 & ((0x55 & ((~((o0 ^ ~k) & ...
        assert o1 == (o0 ^ x)
        assert 0x00 == (((~(i0 << 0x01) ^ (o1 << 0x01)) & (~(i0 << 0x01) ...
        assert w_i0o1_o2 == (((0x0f & ((0x33 & ((0x55 & ((~((o1 ^ ~i0) & ...
        assert o3 == (o0 ^ o2)
        assert w_xyki0_o2o3 == ((0b0 ∘ w_i0o1_o2) + (0b0 ∘ w_ky_o0))
        assert 0b00000 == w_xyki0_o2o3

    """

    def __init__(self, ch, target_weights, inner_equality=False,
                 outer_equality=True):
        """Initialize the pair of SMT problems."""
        assert isinstance(ch, characteristic.CompositeCh)
        assert len(target_weights) == 2
        inner_widths = sum(ch.func.inner_func.input_widths)
        outer_widths = sum(ch.func.input_widths)
        assert target_weights[0] < inner_widths
        assert target_weights[1] < outer_widths - inner_widths

        itw, otw = target_weights
        self.ch = ch
        self.inner_problem = SmtProblem(ch.inner_ch, itw, inner_equality, ch)
        self.outer_problem = SmtProblem(ch.outer_ch, otw, outer_equality, ch)

    @property
    def formula_size(self):
        """The size of the underlying bit-vector formula."""
        size = self.inner_problem.formula_size + self.outer_problem.formula_size

        for d in self.ch.inner_ch.sequence:
            if d in self.ch.outer_ch.sequence:
                size -= d.formula_size  # don't count twice

        return size

    @property
    def pysmt_formula_size(self):
        """The size of the underlying bit-vector formula according to pySMT."""
        sc.reset_env()
        i_f = sc.And(*[bv2pysmt(a) for a in self.inner_problem.assertions])
        o_f = sc.And(*[bv2pysmt(a) for a in self.outer_problem.assertions])
        pysmt_formula = sc.And(i_f, o_f)

        return sc.get_formula_size(pysmt_formula)

    def __str__(self):
        return "{}\n\n{}".format(self.inner_problem, self.outer_problem)

    def to_smtlib(self):
        """Return a SMT_LIB string representation of the formula."""
        sc.reset_env()
        i_f = sc.And(*[bv2pysmt(a) for a in self.inner_problem.assertions])
        o_f = sc.And(*[bv2pysmt(a) for a in self.outer_problem.assertions])
        pysmt_formula = sc.And(i_f, o_f)
        return sc.to_smtlib(pysmt_formula, daggify=False)

    def solve(self, solver_name=None, get_assignment=False):
        """Solve the pair of SMT problems simultaneously.

        Return whether the pair of decision problems are satisfiable.
        If get_assignment is set to True, solve() returns an assignment of
        the variables that makes the pair of SMT problems satisfiable
        (if one of the problems is unsatisfiable, None is returned).

        This assignment is returned as a pair of dictionaries, where
        the first dictionary is the assignment of the inner SMT problem
        and the second dictionary is the assignment of the outer SMT problem.

            >>> from arxpy.bitvector.operation import RotateLeft
            >>> from arxpy.bitvector.function import Function, CompositeFunction
            >>> from arxpy.diffcrypt.difference import XorDiff, DiffVar
            >>> from arxpy.diffcrypt.characteristic import Characteristic, CompositeCh
            >>> class MyInner(Function):
            ...     input_widths = [8]
            ...     output_widths = [8, 8]
            ...     @classmethod
            ...     def eval(cls, k):
            ...         return (k, RotateLeft(k, 1))
            >>> class MyOuter(Function):
            ...     input_widths = [8, 8, 8, 8]
            ...     output_widths = [8, 8]
            ...     @classmethod
            ...     def eval(cls, x, y, k0, k1):
            ...         for ki in [k0, k1]:
            ...             x, y = y + ki, (y + ki) ^ x
            ...         return x, y
            >>> class MyComposite(CompositeFunction):
            ...     input_widths = [8, 8, 8]
            ...     output_widths = [8, 8]
            ...     inner_func = MyInner
            ...     outer_func = MyOuter
            >>> x, y, k = DiffVar("x", 8), DiffVar("y", 8), DiffVar("k", 8)
            >>> ch = CompositeCh(MyComposite, XorDiff, [x, y, k])
            >>> smt_problem = CompositeSmtProblem(ch, [1, 1])
            >>> smt_problem.solve()
            True
            >>> inner_assig, outer_assig = smt_problem.solve(get_assignment=True)
            >>> inner_assig  # doctest:+NORMALIZE_WHITESPACE
            {'differences': OrderedDict([(k, 0x40), (i0, 0x80)]),
            'weight': 0, 'op_weights': OrderedDict()}
            >>> outer_assig  # doctest:+NORMALIZE_WHITESPACE
            {'differences': OrderedDict([(x, 0x40), (y, 0x00), (k, 0x40),
            (i0, 0x80), (o0, 0x40), (o1, 0x00), (o2, 0x80), (o3, 0xc0)]),
            'weight': 1, 'op_weights': OrderedDict([(w_ky_o0, 1),
            (w_i0o1_o2, 0)])}

        """
        sc.reset_env()
        i_f = sc.And(*[bv2pysmt(a) for a in self.inner_problem.assertions])
        o_f = sc.And(*[bv2pysmt(a) for a in self.outer_problem.assertions])
        pysmt_formula = sc.And(i_f, o_f)

        if not get_assignment:
            return sc.is_sat(pysmt_formula, solver_name, logic=logics.QF_BV)
        else:
            model = sc.get_model(pysmt_formula, solver_name, logic=logics.QF_BV)

            if model is None:
                return None
            else:
                inner_assig = self.inner_problem._get_assignment(model)
                outer_assig = self.outer_problem._get_assignment(model)
                return inner_assig, outer_assig
