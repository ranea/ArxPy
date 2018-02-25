"""The Operation module provides the common bit-vector operators."""
import collections
import enum
import functools
import itertools
import math

from sympy.core import cache
from sympy.core import compatibility
from sympy.printing import precedence as preced

from arxpy.bitvector import context
from arxpy.bitvector import core


def _cacheit(func):
    """Cache functions if the CacheContext is enabled."""
    cfunc = cache.cacheit(func)

    def cached_func(*args, **kwargs):
        if context.Cache.current_context:
            return cfunc(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return cached_func


class OperatorType(enum.Enum):
    """Enumeration for specifying the type of an bit-vector operator."""

    Bitwise = enum.auto()
    Relational = enum.auto()
    Shift = enum.auto()
    Rotation = enum.auto()
    Arithmetic = enum.auto()
    Other = enum.auto()


class Operation(core.Term):
    """Base class for bit-vector operators.

    A bit-vector operator represents an operation that takes some bit-vector
    operands (i.e. terms) and and scalar operands (i.e. integer),
    and computes a single bit-vector term. The application of a
    bit-vector operator to a particular set of operands is called
    a bit-vector operation (e.g. bvxor is a bit-vector operator and
    bvxor(Constant(1, 8), Variable("x", 8)) is a bit-vector operation
    with operand Constant(1, 8) and Variable("x", 8)).

    A operator has to specify a signature (its syntantic rules):

    - arity: pair of number specifying the number of bit-vector operands
      and scalar
    - condition (optional): restriction on the bit-widths and the scalar
      values
    - output_width: the bit-width of the resulting bit-vector

    In addition, a operator has also to specify:

    - is_symmetric: whether the operator is symmetric with respect to
      its operand.
    - operator_type (optional): a value from the enumeration OperatorType.
    - operand_types (optional): a list specifying the types of the operands
    in order.
    - simple_operator (optional): an operator is *simple* if it has no scalar
    operands and all the bit-vector operands have the same width. If that
    is the case and simple_operator is True, the operator provides
    *Automatic Constant Conversion*.
    - short_name (optional): a string that will replace the name of the
      operator when printing.
    - unary_symbol/infix_symbool (optional): a string with the operator symbol
      that will be used for printing.

    Automatic Constant Conversion allows to pass integers as bit-vector
    operands as long as one operand is passed as a bit-vector term
    (to deduce the bit-width). For example, the modular addition is a
    simple operator and provides Automatic Constant Conversion as shown:

        >>> from arxpy.bitvector.core import Variable
        >>> (Variable("x", 8) + 1).vrepr()
        "BvAdd(Constant(0b00000001, width=8), Variable('x', width=8), width=8)"

    Many operators can be used with the usual Python operators (+, ^, <=, ...).
    See each operator for more information.

    """

    is_Atom = False
    precedence = preced.PRECEDENCE["Func"]

    operator_type = OperatorType.Other

    @classmethod
    def _parse_args(cls, *args):
        # Automatic Constant Conversion
        if getattr(cls, "simple_operator", False):
            for a in args:
                if isinstance(a, core.Term):
                    w = a.width
                    break
            else:
                msg = "{} expects at least 1 term operand"
                raise TypeError(msg.format(cls.__name__))

            args = [core.Constant(a, w) if isinstance(a, int) else a for a in args]

        if getattr(cls, "operand_types", False):
            if cls.is_symmetric:
                op_types = collections.Counter(cls.operand_types)
                arg_types = collections.Counter([type(a) for a in args])
                assert op_types == arg_types
            else:
                for arg_type, arg in zip(cls.operand_types, args):
                    assert isinstance(arg, arg_type)

        num_terms = 0
        num_scalars = 0
        for a in args:
            if isinstance(a, core.Term):
                num_terms += 1
            elif isinstance(a, int):
                num_scalars += 1
            else:
                assert False
        assert tuple(cls.arity) == (num_terms, num_scalars)

        if cls.is_symmetric:
            args = sorted(args, key=compatibility.default_sort_key)

        if hasattr(cls, "condition"):
            assert cls.condition(*args)

        return args

    @_cacheit
    def __new__(cls, *args, **options):
        """Create the object."""
        val_op = options.pop("validate_operands",
                             context.Validation.current_context)
        evaluate = options.pop("evaluate", context.Evaluation.current_context)
        simplify = options.pop("simplify", context.Simplification.current_context)
        st = options.pop("state", context.StatefulExecution.current_context)

        if val_op:
            args = cls._parse_args(*args)

        if st is not None:
            newargs = []
            for arg in args:
                if isinstance(arg, Operation) and st.contain_op(arg):
                    newargs.append(st.get_id(arg))
                else:
                    newargs.append(arg)
            args = newargs

        width = cls.output_width(*args)

        with context.StatefulExecution(None):
            if evaluate:
                result = cls.eval(*args)
            else:
                result = None

        if result is not None:
            return result

        obj = super().__new__(cls, *args, **options, width=width)

        if isinstance(obj, Operation) and st is not None:
            if st.contain_op(obj):
                return st.get_id(obj)
            else:
                return st.add_op(obj)

        if isinstance(obj, Operation) and simplify and evaluate:
            with context.Simplification(False):
                while True:
                    obj, modified = obj._simplify()
                    if not modified or not isinstance(obj, Operation):
                        break

        return obj

    def _simplify(self):
        """Simplify the bit-vector operation depending on the operation."""
        return self, False

    def _binary_symmetric_simplification(self, compatible_terms):
        """Basic simplification for binary symmetric operator.

        The arguments are assumed to be already simplified.

        Performs the following simplification:

            - Given two compatible terms connected, it replaces them by
              the operation with operands these two terms.

        Two terms are connected if they are arguments of the same operator
        node when the whole expression is flattened. For example, in
        ((x ^ y) + z) + t, {x, t} are not connected by {z, t} are.

        Two terms are compatible if they can be simplified. For example,
        two constants are always compatible or {x, ~x} are compatible
        if the root operation is bitwise.

        The argument compatible_terms is a sequence of lambda functions
        that computes each compatible terms. For example, BvXor has the
        following compatible_terms sequence:

            [lambda x: BvNot(x), lambda x: x]

        """
        op = type(self)
        assert isinstance(compatible_terms, collections.Sequence)

        def replace_constant(cte, expr):
            modified = False
            newargs = []

            for arg in expr.args:
                if not modified:
                    if isinstance(arg, core.Constant):
                        arg = op(x, arg)
                        modified = True
                    elif isinstance(arg, op):
                        arg, modified = replace_constant(x, arg)

                    newargs.append(arg)
                else:
                    newargs.append(arg)

            assert len(newargs) in [1, 2]
            if len(newargs) == 1:
                new_expr = newargs[0]
            elif len(newargs) == 2:
                new_expr = op(*newargs)

            return new_expr, modified

        def replace_term(x, compatible_terms, expr):
            modified = False
            newargs = []

            for arg in expr.args:
                if not modified:
                    if arg in compatible_terms:
                        arg = op(x, arg)
                        modified = True
                    elif isinstance(arg, op):
                        arg, modified = replace_term(x, compatible_terms, arg)

                    newargs.append(arg)
                else:
                    newargs.append(arg)

            assert len(newargs) in [1, 2]
            if len(newargs) == 1:
                new_expr = newargs[0]
            elif len(newargs) == 2:
                new_expr = op(*newargs)

            return new_expr, modified

        x, y = self.args

        mod = False  # modified

        if isinstance(x, core.Constant) and isinstance(y, op):
            new_expr, mod = replace_constant(x, y)
        elif isinstance(y, core.Constant) and isinstance(x, op):
            new_expr, mod = replace_constant(y, x)

        if not mod and not isinstance(x, core.Constant) and isinstance(y, op):
            new_expr, mod = replace_term(x, [f(x) for f in compatible_terms], y)

        if not mod and not isinstance(y, core.Constant) and isinstance(x, op):
            new_expr, mod = replace_term(y, [f(y) for f in compatible_terms], x)

        if not mod and isinstance(x, op) and isinstance(y, op):
            x1, x2 = x.args

            if op(x1, y, evaluate=False) != op(x1, y):
                new_expr = op(x2, op(x1, y))
                mod = True

            if not mod and op(x2, y, evaluate=False) != op(x2, y):
                new_expr = op(x1, op(x2, y))
                mod = True

            if not mod:
                new_expr, mod = op(x1, y)._simplify()
                new_expr = op(x2, new_expr)

            if not mod:
                new_expr, mod = op(x2, y)._simplify()
                new_expr = op(x1, new_expr)

        if mod:
            return new_expr, True
        else:
            return self, False

    @classmethod
    def eval(cls, *args):
        """Evaluate the bit-vector operator with the given arguments.

        Note that the operands in args have been already processed
        so they are either Term or int.
        """
        return

    @classmethod
    def class_key(cls):
        """Return the identifier used for sorting."""
        return 3, 0, cls.__name__

    @property
    def formula_size(self):
        """The formula size of the operation."""
        def L(n):
            return int(math.ceil(math.log(n, 2)))

        def bin_enc(n):
            return 1 + L(n + 1)

        size = 1 + bin_enc(self.width)
        for arg in self.args:
            if isinstance(arg, int):
                size += bin_enc(arg)
            else:
                size += arg.formula_size

        return size


# Bitwise operators

class BvNot(Operation):
    """Bitwise negation operation.

    It overrides the operator ~. See Operation for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvNot
        >>> BvNot(Constant(0b1010101, 7))
        0b0101010
        >>> ~Constant(0b1010101, 7)
        0b0101010
        >>> ~Variable("x", 8)
        ~x

    """

    arity = [1, 0]
    is_symmetric = False
    operator_type = OperatorType.Bitwise
    unary_symbol = "~"

    @classmethod
    def output_width(cls, x):
        return x.width

    @classmethod
    def eval(cls, x):
        def doit(x, width):
            """NOT operation when the operand is int."""
            return ~x % (2 ** width)

        if isinstance(x, core.Constant):
            return core.Constant(doit(int(x), x.width), x.width)
        elif isinstance(x, BvNot):
            return x.args[0]
        # # De Morgan's laws (disabled, all op equal precedence)
        # if isinstance(x, BvAnd):
        #     return BvOr(BvNot(x.args[0]), BvNot(x.args[1]))
        # elif isinstance(x, BvOr):
        #     return BvAnd(BvNot(x.args[0]), BvNot(x.args[1]))


class BvAnd(Operation):
    """Bitwise AND (logical conjunction) operation.

    It overrides the operator & and provides Automatic Constant Conversion.
    See Operation for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvAnd
        >>> BvAnd(Constant(5, 8), Constant(3, 8))
        0x01
        >>> BvAnd(Constant(5, 8), 3)
        0x01
        >>> Constant(5, 8) & 3
        0x01
        >>> Variable("x", 8) & Variable("y", 8)
        x & y

    """

    arity = [2, 0]
    is_symmetric = True
    operator_type = OperatorType.Bitwise
    simple_operator = True
    infix_symbol = "&"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """AND operation when both operands are int."""
            return x & y

        zero = core.Constant(0, x.width)
        allones = BvNot(zero)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)
        elif x == zero or y == zero:
            return zero
        elif x == allones:
            return y
        elif y == allones:
            return x
        elif x == y:
            return x
        elif x == BvNot(y):
            return zero

    def _simplify(self, *args, **kwargs):
        compatible_terms = [
            lambda x: x,
            lambda x: BvNot(x)
        ]

        return self._binary_symmetric_simplification(compatible_terms)


class BvOr(Operation):
    """Bitwise OR (logical disjunction) operation.

    It overrides the operator | and provides Automatic Constant Conversion.
    See Operation for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvOr
        >>> BvOr(Constant(5, 8), Constant(3, 8))
        0x07
        >>> BvOr(Constant(5, 8), 3)
        0x07
        >>> Constant(5, 8) | 3
        0x07
        >>> Variable("x", 8) | Variable("y", 8)
        x | y

    """

    arity = [2, 0]
    is_symmetric = True
    operator_type = OperatorType.Bitwise
    simple_operator = True
    infix_symbol = "|"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """OR operation when both operands are int."""
            return x | y

        zero = core.Constant(0, x.width)
        allones = BvNot(zero)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)
        elif x == allones or y == allones:
            return allones
        elif x == zero:
            return y
        elif y == zero:
            return x
        elif x == y:
            return x
        elif x == BvNot(y):
            return allones

    def _simplify(self, *args, **kwargs):
        compatible_terms = [
            lambda x: x,
            lambda x: BvNot(x)
        ]

        return self._binary_symmetric_simplification(compatible_terms)


class BvXor(Operation):
    """Bitwise XOR (exclusive-or) operation.

    It overrides the operator ^ and provides Automatic Constant Conversion.
    See Operation for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvXor
        >>> BvXor(Constant(5, 8), Constant(3, 8))
        0x06
        >>> BvXor(Constant(5, 8), 3)
        0x06
        >>> Constant(5, 8) ^ 3
        0x06
        >>> Variable("x", 8) ^ Variable("y", 8)
        x ^ y

    """

    arity = [2, 0]
    is_symmetric = True
    operator_type = OperatorType.Bitwise
    simple_operator = True
    infix_symbol = "^"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """XOR operation when both operands are int."""
            return x ^ y

        zero = core.Constant(0, x.width)
        allones = BvNot(zero)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)
        elif x == zero:
            return y
        elif y == zero:
            return x
        elif x == allones:
            return BvNot(y)
        elif y == allones:
            return BvNot(x)
        elif x == y:
            return zero
        elif x == BvNot(y):
            return allones

    def _simplify(self, *args, **kwargs):
        compatible_terms = [
            lambda x: x,
            lambda x: BvNot(x)
        ]

        return self._binary_symmetric_simplification(compatible_terms)

# Relational operators


class BvComp(Operation):
    """Equality operator.

    Provides Automatic Constant Conversion. See Operation for more
    information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvComp
        >>> BvComp(Constant(1, 8), Constant(2, 8))
        0b0
        >>> BvComp(Constant(1, 8), 2)
        0b0
        >>> BvComp(Constant(1, 8), Variable("y", 8))
        0x01 == y

    The operator == is used for exact structural equality testing and
    it returns either True or False. On the other hand, BvComp
    performs symbolic equality testing and if it cannot prove the objects
    are equal (or unequal), it leaves the relation unevaluated.

        >>> Variable("x", 8) == Variable("y", 8)
        False
        >>> BvComp(Variable("x", 8), Variable("y", 8))  # symbolic equality
        x == y

    """

    arity = [2, 0]
    is_symmetric = True
    operator_type = OperatorType.Relational
    simple_operator = True
    infix_symbol = "=="

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return 1

    @classmethod
    def eval(cls, x, y):
        zero = core.Constant(0, 1)
        one = core.Constant(1, 1)

        if x is y:
            return one
        elif isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return one if x.val == y.val else zero


class BvUlt(Operation):
    """Unsigned less than operator.

    It overrides < and provides Automatic Constant Conversion.
    See Operation for moreinformation.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvUlt
        >>> BvUlt(Constant(1, 8), Constant(2, 8))
        0b1
        >>> BvUlt(Constant(1, 8), 2)
        0b1
        >>> Constant(1, 8) < 2
        0b1
        >>> Constant(1, 8) < Variable("y", 8)
        0x01 < y

    """

    arity = [2, 0]
    is_symmetric = False
    operator_type = OperatorType.Relational
    simple_operator = True
    infix_symbol = "<"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return 1

    @classmethod
    def eval(cls, x, y):
        zero = core.Constant(0, 1)
        one = core.Constant(1, 1)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return one if x.val < y.val else zero


class BvUle(Operation):
    """Unsigned less than or equal operator.

    It overrides <= and provides Automatic Constant Conversion.
    See Operation for moreinformation.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvUle
        >>> BvUle(Constant(2, 8), Constant(2, 8))
        0b1
        >>> BvUle(Constant(2, 8), 2)
        0b1
        >>> Constant(2, 8) <= 2
        0b1
        >>> Constant(2, 8) <= Variable("y", 8)
        0x02 <= y

    """

    arity = [2, 0]
    is_symmetric = False
    operator_type = OperatorType.Relational
    simple_operator = True
    infix_symbol = "<="

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return 1

    @classmethod
    def eval(cls, x, y):
        zero = core.Constant(0, 1)
        one = core.Constant(1, 1)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return one if x.val <= y.val else zero


class BvUgt(Operation):
    """Unsigned greater than operator.

    It overrides > and provides Automatic Constant Conversion.
    See Operation for moreinformation.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvUgt
        >>> BvUgt(Constant(1, 8), Constant(2, 8))
        0b0
        >>> BvUgt(Constant(1, 8), 2)
        0b0
        >>> Constant(1, 8) > 2
        0b0
        >>> Constant(1, 8) > Variable("y", 8)
        0x01 > y

    """

    arity = [2, 0]
    is_symmetric = False
    operator_type = OperatorType.Relational
    simple_operator = True
    infix_symbol = ">"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return 1

    @classmethod
    def eval(cls, x, y):
        zero = core.Constant(0, 1)
        one = core.Constant(1, 1)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return one if x.val > y.val else zero


class BvUge(Operation):
    """Unsigned greater than or equal operator.

    It overrides >= and provides Automatic Constant Conversion.
    See Operation for moreinformation.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvUgt
        >>> BvUge(Constant(2, 8), Constant(2, 8))
        0b1
        >>> BvUge(Constant(2, 8), 2)
        0b1
        >>> Constant(2, 8) >= 2
        0b1
        >>> Constant(2, 8) >= Variable("y", 8)
        0x02 >= y

    """

    arity = [2, 0]
    is_symmetric = False
    operator_type = OperatorType.Relational
    simple_operator = True
    infix_symbol = ">="

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return 1

    @classmethod
    def eval(cls, x, y):
        zero = core.Constant(0, 1)
        one = core.Constant(1, 1)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return one if x.val >= y.val else zero


# Shifts operators

class BvShl(Operation):
    """Shift left operation.

    It overrides << and provides Automatic Constant Conversion.
    See Operation for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvShl
        >>> BvShl(Constant(0b10001, 5), Constant(1, 5))
        0b00010
        >>> BvShl(Constant(0b10001, 5), 1)
        0b00010
        >>> Constant(0b10001, 5) << 1
        0b00010
        >>> Variable("x", 8) << Variable("y", 8)
        x << y

    """

    arity = [2, 0]
    is_symmetric = False
    operator_type = OperatorType.Shift
    simple_operator = True
    infix_symbol = "<<"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """Shift left operation when both operands are int."""
            return (x << y) % (2 ** width)

        zero = core.Constant(0, x.width)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)
        elif isinstance(y, core.Constant) and y >= x.width:
            return zero
        elif x == zero or y == zero:
            return x
        elif isinstance(x, BvShl) and isinstance(x.args[1], core.Constant) \
                and isinstance(y, core.Constant):
            # prevent out of bound
            r = min(x.args[0].width, int(x.args[1]) + int(y))
            return BvShl(x.args[0], r)


class BvLshr(Operation):
    """Logical right shift operation.

    It overrides >> and provides Automatic Constant Conversion.
    See Operation for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvLshr
        >>> BvLshr(Constant(0b10001, 5), Constant(1, 5))
        0b01000
        >>> BvLshr(Constant(0b10001, 5), 1)
        0b01000
        >>> Constant(0b10001, 5) >> 1
        0b01000
        >>> Variable("x", 8) >> Variable("y", 8)
        x >> y

    """

    arity = [2, 0]
    is_symmetric = False
    operator_type = OperatorType.Shift
    simple_operator = True
    infix_symbol = ">>"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """Logical right shift operation when both operands are int."""
            return x >> y

        zero = core.Constant(0, x.width)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)
        elif isinstance(y, core.Constant) and y >= x.width:
            return zero
        elif x == zero or y == zero:
            return x
        elif isinstance(x, BvLshr) and isinstance(x.args[1], core.Constant) \
                and isinstance(y, core.Constant):
            r = min(x.args[0].width, int(x.args[1]) + int(y))
            return BvLshr(x.args[0], r)


class RotateLeft(Operation):
    """Circular left rotation operation.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import RotateLeft
        >>> RotateLeft(Constant(150, 8), 2)
        0x5a
        >>> RotateLeft(Variable("x", 8), 2)
        x <<< 2

    """

    arity = [1, 1]
    is_symmetric = False
    operator_type = OperatorType.Rotation
    infix_symbol = "<<<"

    @classmethod
    def condition(cls, x, r):
        return x.width > r >= 0

    @classmethod
    def output_width(cls, x, r):
        return x.width

    @classmethod
    def eval(cls, x, r):
        # if isinstance(x, core.Constant):
        #     if x.width == 1 or r == 0:
        #         return x
        #     else:
        #         return Concat(x[x.width - r - 1:], x[x.width - 1: x.width - r])

        def doit(val, r, width):
            """Left cyclic rotation operation when both operands are int."""
            mask = 2 ** width - 1
            return ((val << r) & mask) | ((val & mask) >> (width) - r)

        if isinstance(x, core.Constant):
            return core.Constant(doit(int(x), r, x.width), x.width)
        elif r == 0:
            return x
        elif isinstance(x, RotateLeft):
            return RotateLeft(x.args[0], (x.args[1] + r) % x.args[0].width)
        elif isinstance(x, RotateRight):
            return RotateRight(x.args[0], (x.args[1] - r) % x.args[0].width)


class RotateRight(Operation):
    """Circular right rotation operation.

    It provides Automatic Constant Conversion. See Operation for more
    information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import RotateRight
        >>> RotateRight(Constant(150, 8), 3)
        0xd2
        >>> RotateRight(Variable("x", 8), 3)
        x >>> 3

    """

    arity = [1, 1]
    is_symmetric = False
    operator_type = OperatorType.Rotation
    infix_symbol = ">>>"

    @classmethod
    def condition(cls, x, r):
        return x.width > r >= 0

    @classmethod
    def output_width(cls, x, r):
        return x.width

    @classmethod
    def eval(cls, x, r):
        # if isinstance(x, core.Constant):
        #     if x.width == 1 or r == 0:
        #         return x
        #     else:
        #         return Concat(x[r - 1:], x[x.width - 1: r])

        def doit(val, r, width):
            """Right cyclic rotation operation when both operands are int."""
            mask = 2 ** width - 1
            r = r % width
            return ((val & mask) >> r) | (val << (width - r) & mask)

        if isinstance(x, core.Constant):
            return core.Constant(doit(int(x), r, x.width), x.width)
        elif r == 0:
            return x
        elif isinstance(x, RotateRight):
            return RotateRight(x.args[0], (x.args[1] + r) % x.args[0].width)
        elif isinstance(x, RotateLeft):
            return RotateLeft(x.args[0], (x.args[1] - r) % x.args[0].width)


# Others

class Ite(Operation):
    """If-then-else operator.

    Ite(b, x, y) returns x if b is 0b0 (True) and y if b is 0b1 (False).

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import Ite
        >>> Ite(Constant(0, 1), Constant(0b11, 2), Constant(0b00, 2))
        0b00
        >>> Ite(Constant(1, 1), Constant(0x1, 4), Constant(0x0, 4))
        0x1

    """

    arity = [3, 0]
    is_symmetric = False

    @classmethod
    def condition(cls, b, x, y):
        return b.width == 1 and x.width == y.width

    @classmethod
    def output_width(cls, b, x, y):
        return x.width

    @classmethod
    def eval(cls, b, x, y):
        if b == core.Constant(1, 1):
            return x
        elif b == core.Constant(0, 1):
            return y


class Extract(Operation):
    """Extraction of bits.

    Given the bit-vector (t[n-1], ...., t[1], t[0]), extract(t, i, j)
    extracts the bits from t[i] down t[j] (end points included). This
    can also be done by t[i:j] since the operator [] is overriden
    for bit-vector types. In particular, t[i] = t[i:i].

    Note that the indices can be omitted when they reference the MSB or the LSB
    (i.e. t[n-1:j] = t[:j] and t[i:0] = t[i:]).

    .. Warning:

        In Python, the operator [] has a different meaning for other
        data types. In general, given a sequence l (e.g. a list),
        l[i:j] extracts the elements from the position i up to (but no
        included) the position j.

        For example, for a given list l and bit-vector t, l[0:1] equals
        to l[0] and t[1:0] equals to (t[0], t[1]).

    Usage:

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import Extract
        >>> Extract(Constant(0b11100, 5), 4, 2,)
        0b111
        >>> Constant(0b11100, 5)[4:2]
        0b111
        >>> Variable("x", 8)[4:2]
        x[4:2]
        >>> Variable("x", 8)[7:0]
        x

    """

    arity = [1, 2]
    is_symmetric = False
    operand_types = [core.Term, int, int]

    @classmethod
    def condition(cls, t, i, j):
        return t.width > i >= j

    @classmethod
    def output_width(cls, t, i, j):
        return i - j + 1

    @classmethod
    def eval(cls, x, i, j):
        def doit(x, i, j):
            """Extract from x[i] down x[j] from a constant x."""
            bin_repr = x.bin()
            prefix, value = bin_repr[:2], bin_repr[2:]
            n = x.width
            value = value[n - 1 - i:n - j]  # e.g.: i=n-1, j=0
            return int(prefix + value, 2)

        if isinstance(x, core.Constant):
            return core.Constant(doit(x, i, j), cls.output_width(x, i, j))
        elif i == x.width - 1 and j == 0:
            return x
        elif isinstance(x, Extract):
            # x[3:1][2] = (x3 x2 x1)[2] = x3 = x[3]
            offset = x.args[2]
            return Extract(x.args[0], i + offset, j + offset)
        elif isinstance(x, Concat):
            if i <= x.args[1].width - 1:
                # 4-bit x, y: concat(x, y)[3:] = y[3:]
                return Extract(x.args[1], i, j)
            elif j >= x.args[1].width:
                # 4-bit x, y: concat(x, y)[:5] = x[:1]
                offset = x.args[1].width
                return Extract(x.args[0], i - offset, j - offset)
        elif isinstance(x, (BvShl, RotateLeft)) and x.args[1] <= j:
            # (x << 1)[:2] = x[n-2: 1]
            offset = int(x.args[1])
            return Extract(x.args[0], i - offset, j - offset)
        elif isinstance(x, (BvLshr, RotateRight)) and i < x.width - x.args[1]:
            # (x >> 1)[n-3:] = x[n-2: 1]
            offset = int(x.args[1])
            return Extract(x.args[0], i + offset, j + offset)

        # # disabled (all op equal precedence)
        # if isinstance(x, (BvAnd, BvOr, BvXor, BvNot)):
        #     args = [Extract(end, start, a) for a in x.args]
        #     return x.func(*args)


class Concat(Operation):
    """Concatenation operation.

    Given the bit-vectors (x[n-1], ...., x[1], x[0]) and
    (y[m-1], ...., y[1], y[0]), concat(x, y) returns the bit-vector
    (x[n-1], ...., x[1], x[0], y[m-1], ...., y[1], y[0]).

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import Concat
        >>> Concat(Constant(0x12, 8), Constant(0x345, 12))
        0x12345
        >>> Concat(Variable("x", 8), Variable("y", 8))
        x ∘ y

    """

    arity = [2, 0]
    is_symmetric = False
    infix_symbol = "∘"

    @classmethod
    def output_width(cls, x, y):
        return x.width + y.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """Concatenation when both operands are int."""
            return int(x.bin() + y.bin()[2:], 2)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(x, y), cls.output_width(x, y))
        # TODO: add test
        elif isinstance(x, core.Constant) and isinstance(y, Concat) and \
                isinstance(y.args[0], core.Constant):
            return Concat(Concat(x, y.args[0]), y.args[1])
        elif isinstance(y, core.Constant) and isinstance(x, Concat) and \
                isinstance(x.args[1], core.Constant):
            return Concat(x.args[0], Concat(x.args[1], y))
        elif isinstance(x, Extract) and isinstance(y, Extract):
            # x[5:4] concat x[3:2] = x[5:2]
            if x.args[0] == y.args[0] and x.args[2] == y.args[1] + 1:
                return Extract(x.args[0], x.args[1], y.args[2])


class ZeroExtend(Operation):
    """Extend with zeroes preserving the unsigned value.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import ZeroExtend
        >>> ZeroExtend(Constant(0x12, 8), 4)
        0x012
        >>> ZeroExtend(Variable("x", 8), 4)
        0x0 ∘ x

    """

    arity = [1, 1]
    is_symmetric = False
    short_name = "ext"

    @classmethod
    def condition(cls, x, i):
        return i >= 0

    @classmethod
    def output_width(cls, x, i):
        return x.width + i

    @classmethod
    def eval(cls, x, i):
        if i == 0:
            return x
        else:
            return Concat(core.Constant(0, i), x)


class Repeat(Operation):
    """Repeat n-times a given bit-vector.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import Repeat
        >>> Repeat(Constant(0x1, 4), 4)
        0x1111
        >>> Repeat(Variable("x", 8), 4)
        ((x ∘ x) ∘ x) ∘ x

    """

    arity = [1, 1]
    is_symmetric = False

    @classmethod
    def condition(cls, x, i):
        return i >= 1

    @classmethod
    def output_width(cls, x, i):
        return i * x.width

    @classmethod
    def eval(cls, x, i):
        if i == 1:
            return x
        else:
            return functools.reduce(Concat, itertools.repeat(x, i))


# Arithmetic operators

class BvNeg(Operation):
    """Unary minus operation.

    It overrides the unary operator -. See Operation for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvNeg
        >>> BvNeg(Constant(1, 8))
        0xff
        >>> -Constant(1, 8)
        0xff
        >>> BvNeg(Variable("x", 8))
        -x

    """

    arity = [1, 0]
    is_symmetric = False
    operator_type = OperatorType.Arithmetic
    unary_symbol = "-"

    @classmethod
    def output_width(cls, x):
        return x.width

    @classmethod
    def eval(cls, x):
        def doit(x, width):
            """Unary minus operation when the operand is int."""
            return ((2 ** width) - x) % (2 ** width)

        if isinstance(x, core.Constant):
            return core.Constant(doit(int(x), x.width), x.width)
        elif isinstance(x, BvNeg):
            return x.args[0]
        # # disabled (all op equal precedence)
        # elif isinstance(x, BvAdd):
        #     return BvAdd(BvNeg(x.args[0]), BvNeg(x.args[1]))
        # elif isinstance(x, (BvMul, BvDiv, BvMod)):
        #     return x.func(BvNeg(x.args[0]), x.args[1])


class BvAdd(Operation):
    """Modular addition operation.

    It overrides the operator + and provides Automatic Constant Conversion.
    See Operation for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvAdd
        >>> BvAdd(Constant(1, 8), Constant(2, 8))
        0x03
        >>> BvAdd(Constant(1, 8), 2)
        0x03
        >>> Constant(1, 8) + 2
        0x03
        >>> Variable("x", 8) + Variable("y", 8)
        x + y

    """

    arity = [2, 0]
    is_symmetric = True
    operator_type = OperatorType.Arithmetic
    simple_operator = True
    infix_symbol = "+"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """Modular addition when both operands are integers."""
            return (x + y) % (2 ** width)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)

        zero = core.Constant(0, x.width)

        if x == zero:
            return y
        elif y == zero:
            return x
        elif x == BvNeg(y):
            return zero

    def _simplify(self, *args, **kwargs):
        compatible_terms = [
            lambda x: BvNeg(x)
        ]

        return self._binary_symmetric_simplification(compatible_terms)


class BvSub(Operation):
    """Modular subtraction operation.

    It overrides the operator - and provides Automatic Constant Conversion.
    See Operation for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvSub
        >>> BvSub(Constant(1, 8), Constant(2, 8))
        0xff
        >>> BvSub(Constant(1, 8), 2)
        0xff
        >>> Constant(1, 8) - 2
        0xff
        >>> Variable("x", 8) - Variable("y", 8)
        x + -y

    """

    arity = [2, 0]
    is_symmetric = False
    operator_type = OperatorType.Arithmetic
    simple_operator = True
    infix_symbol = "-"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        return BvAdd(x, BvNeg(y))


class BvMul(Operation):
    """Modular multiplication operation.

    It overrides the operator * and provides Automatic Constant Conversion.
    See Operation for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvMul
        >>> BvMul(Constant(4, 8), Constant(3, 8))
        0x0c
        >>> BvMul(Constant(4, 8), 3)
        0x0c
        >>> Constant(4, 8) * 3
        0x0c
        >>> Variable("x", 8) * Variable("y", 8)
        x * y

    """

    arity = [2, 0]
    is_symmetric = True
    operator_type = OperatorType.Arithmetic
    simple_operator = True
    infix_symbol = "*"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """Modular multiplication when both operands are int."""
            return (x * y) % (2 ** width)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)

        zero = core.Constant(0, x.width)
        one = core.Constant(1, x.width)

        if x == zero or y == zero:
            return zero
        elif x == one:
            return y
        elif y == one:
            return x


class BvUdiv(Operation):
    """Unsigned and truncated division operation.

    It overrides the operator / and provides Automatic Constant Conversion.
    See Operation for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvUdiv
        >>> BvUdiv(Constant(0x0c, 8), Constant(3, 8))
        0x04
        >>> BvUdiv(Constant(0x0c, 8), 3)
        0x04
        >>> Constant(0x0c, 8) / 3
        0x04
        >>> Variable("x", 8) / Variable("y", 8)
        x / y
        >>> Constant(0x01, 8) / 0  # special case
        0x01 / 0x00

    """

    arity = [2, 0]
    is_symmetric = False
    operator_type = OperatorType.Arithmetic
    simple_operator = True
    infix_symbol = "/"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """Division operation (truncated) when both operands are int."""
            assert y != 0
            return x // y

        zero = core.Constant(0, x.width)
        one = core.Constant(1, x.width)

        if y == zero:
            return None

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)
        elif x == y:
            return one
        elif x == zero:
            return zero
        elif y == one:
            return x


class BvUrem(Operation):
    """Unsigned remainder (modulus) operation.

    Usage:

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvUrem
        >>> BvUrem(Constant(0x0d, 8), Constant(3, 8))
        0x01
        >>> BvUrem(Constant(0x0d, 8), 3)
        0x01
        >>> Constant(0x0d, 8) % 3
        0x01
        >>> Variable("x", 8) % Variable("y", 8)
        x % y
        >>> Constant(0x01, 8) % 0  # special case
        0x01 % 0x00

    """

    arity = [2, 0]
    is_symmetric = False
    operator_type = OperatorType.Arithmetic
    simple_operator = True
    infix_symbol = "%"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """Remainder operation when both operands are int."""
            assert y != 0
            return x % y

        zero = core.Constant(0, x.width)
        one = core.Constant(1, x.width)

        if y == zero:
            return None

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)
        elif x == y or x == zero or y == one:
            return zero
