"""Provide the common bit-vector operators."""
import collections
import functools
import itertools
import math

from sympy.core import cache
from sympy import default_sort_key
from sympy.printing import precedence as sympy_precedence

from arxpy.bitvector import context
from arxpy.bitvector import core


def _cacheit(func):
    """Cache functions if `CacheContext` is enabled."""
    cfunc = cache.cacheit(func)

    def cached_func(*args, **kwargs):
        if context.Cache.current_context:
            return cfunc(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return cached_func


def _tuplify(seq):
    if isinstance(seq, collections.abc.Sequence):
        return tuple(seq)
    else:
        return tuple([seq])


# noinspection PyUnresolvedReferences
class Operation(core.Term):
    """Represent bit-vector operations.

    A bit-vector operation takes some bit-vector operands (i.e. `Term`)
    and some scalar operands (i.e. `int`), and returns a single
    bit-vector term. Often, *operator* is used to denote the operation
    as a function (without operands) and *operation* is used to denote
    the application of a operator to some operands.

    This class is not meant to be instantiated but to provide a base
    class for the different types of bit-vector operations.

    Attributes:
        arity: a pair of number specifying the number of bit-vector operands
            (at least one) and scalar operands.
        is_symmetric: True if the operator is symmetric with respect to
            its operands. Operators with scalar operands cannot be symmetric.
        is_simple: True if the operator is *simple*, that is, all its
            operands are bit-vector of the same width. Simple operators allow
            *Automatic Constant Conversion*, that is, instead of passing
            all arguments as bit-vector types, it is possible to pass
            arguments as plain integers.

            ::

                >>> from arxpy.bitvector.core import Constant
                >>> (Constant(1, 8) + 1).vrepr()
                'Constant(0b00000010, width=8)'

        operand_types: a list specifying the types of the operands (optional
            if all operands are bit-vectors)
        alt_name: an alternative name used when printing (optional)
        unary_symbol: a symbol used when printing (optional)
        infix_symbol: a symbol used when printing (optional)
    """

    is_Atom = False
    precedence = sympy_precedence.PRECEDENCE["Func"]

    is_simple = False

    @_cacheit
    def __new__(cls, *args, **options):
        val_op = options.pop("validate_operands",
                             context.Validation.current_context)
        evaluate = options.pop("evaluate", context.Evaluation.current_context)
        simplify = options.pop("simplify",
                               context.Simplification.current_context)
        st = options.pop("state", context.Memoization.current_context)
        noteval = _tuplify(options.pop("notevaluate",
                                       context.NotEvaluation.current_context))

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

        if noteval is None:
            noteval = []

        with context.Memoization(None):
            if evaluate and not(cls in noteval):
                result = cls.eval(*args)
            else:
                result = None

        if result is not None:
            # result is already a Term/Operation (possibly simplified)
            obj = result
        else:
            obj = super().__new__(cls, *args, width=width)

            if isinstance(obj, Operation) and simplify and evaluate:
                with context.Simplification(False), context.Memoization(None):
                    while True:
                        obj, modified = obj._simplify()
                        if not modified or not isinstance(obj, Operation):
                            break

        if isinstance(obj, Operation) and st is not None:
            for arg in obj.args:
                if isinstance(arg, Operation):
                    raise ValueError("arg {} of {} was not memoized".format(arg, obj))
            if st.contain_op(obj):
                return st.get_id(obj)
            else:
                return st.add_op(obj)

        return obj

    @classmethod
    def _parse_args(cls, *args):
        # Automatic Constant Conversion
        if cls.is_simple:
            for a in args:
                if isinstance(a, core.Term):
                    w = a.width
                    break
            else:
                msg = "{} expects at least 1 term operand"
                raise TypeError(msg.format(cls.__name__))

            args = [core.Constant(a, w) if isinstance(a, int) else a for a in args]

        if hasattr(cls, "operand_types"):
            operand_types = cls.operand_types
        else:
            operand_types = [core.Term for _ in args]
        for arg_type, arg in zip(operand_types, args):
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
        assert num_terms >= 1

        if cls.is_symmetric:
            args = sorted(args, key=default_sort_key)

        assert cls.condition(*args), "{}.condition({}) did not hold".format(cls, args)

        return args

    def _simplify(self):
        """Simplify the bit-vector operation.

        Return the simplified value and a boolean flag depending on
        whether or not the expression was reduced.
        """
        return self, False

    def _binary_symmetric_simplification(self, compatible_terms):
        """Simplify a binary symmetric operation.

        Replace pair of *compatible connected terms* by their resulting value.
        * Two terms are connected if they are arguments of the same operator
          node when the bit-vector expression is flattened (e.g. ``z``
          and ``t`` are connected in ``((x ^ y) + z) + t``.
        * Two connected terms are compatible if they can be simplified.
          For example, two constants are always compatible.

        Note that this function assumed the arguments of the operation
        are already simplified.

        Args:
            compatible_terms: a list of lambda functions specifying
            the compatible terms for a particular operator.

        """
        op = type(self)
        assert isinstance(compatible_terms, collections.abc.Sequence)

        # noinspection PyShadowingNames
        def replace_constant(cte, expr):
            modified = False
            newargs = []

            for arg in expr.args:
                if not modified:
                    if isinstance(arg, core.Constant):
                        arg = op(cte, arg)
                        modified = True
                    elif isinstance(arg, op):
                        arg, modified = replace_constant(cte, arg)

                    newargs.append(arg)
                else:
                    newargs.append(arg)

            assert len(newargs) in [1, 2]
            if len(newargs) == 1:
                new_expr = newargs[0]
            elif len(newargs) == 2:
                new_expr = op(*newargs)
            else:
                raise ValueError("invalid newargs length: {}".format(newargs))

            return new_expr, modified

        # noinspection PyShadowingNames
        def replace_term(term, compatible_terms, expr):
            modified = False
            newargs = []

            for arg in expr.args:
                if not modified:
                    if arg in compatible_terms:
                        arg = op(term, arg)
                        modified = True
                    elif isinstance(arg, op):
                        arg, modified = replace_term(term, compatible_terms, arg)

                    newargs.append(arg)
                else:
                    newargs.append(arg)

            assert len(newargs) in [1, 2]
            if len(newargs) == 1:
                new_expr = newargs[0]
            elif len(newargs) == 2:
                new_expr = op(*newargs)
            else:
                raise ValueError("invalid newargs length: {}".format(newargs))

            return new_expr, modified

        x, y = self.args

        modified = False  # modified

        if isinstance(x, core.Constant) and isinstance(y, op):
            new_expr, modified = replace_constant(x, expr=y)
        elif isinstance(y, core.Constant) and isinstance(x, op):
            new_expr, modified = replace_constant(y, expr=x)

        if not modified and not isinstance(x, core.Constant) and isinstance(y, op):
            new_expr, modified = replace_term(x, [f(x) for f in compatible_terms], y)

        if not modified and not isinstance(y, core.Constant) and isinstance(x, op):
            new_expr, modified = replace_term(y, [f(y) for f in compatible_terms], x)

        if not modified and isinstance(x, op) and isinstance(y, op):
            x1, x2 = x.args

            if op(x1, y, evaluate=False) != op(x1, y):
                new_expr = op(x2, op(x1, y))
                modified = True

            if not modified and op(x2, y, evaluate=False) != op(x2, y):
                new_expr = op(x1, op(x2, y))
                modified = True

            if not modified:
                new_expr, modified = op(x1, y)._simplify()
                new_expr = op(x2, new_expr)

            if not modified:
                new_expr, modified = op(x2, y)._simplify()
                new_expr = op(x1, new_expr)

        if modified:
            # noinspection PyUnboundLocalVariable
            return new_expr, True
        else:
            return self, False

    @classmethod
    def condition(cls, *args):
        """Check if the operands verify the restrictions of the operator."""
        return True

    def output_width(*args):
        """Return the bit-width of the resulting bit-vector."""
        raise NotImplementedError("subclasses need to override this method")

    @classmethod
    def eval(cls, *args):
        """Evaluate the operator with given operands.

        This is an internal method. To evaluate a bit-vector operation,
        use the operator ``()``.
        """
        raise NotImplementedError("subclasses need to override this method")

    @classmethod
    def class_key(cls):
        """Return the key (identifier) of the class for sorting."""
        return 3, 0, cls.__name__

    @property
    def formula_size(self):
        """The formula size of the operation."""
        def log2(n):
            return int(math.ceil(math.log(n, 2)))

        def bin_enc(n):
            return 1 + log2(n + 1)

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

    It overrides the operator ~. See `Operation` for more information.

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
    See `Operation` for more information.

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
    is_simple = True
    infix_symbol = "&"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """AND operation when both operands are int."""
            return x & y

        zero = core.Constant(0, x.width)
        allones = BvNot(zero)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
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
    See `Operation` for more information.

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
    is_simple = True
    infix_symbol = "|"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """OR operation when both operands are int."""
            return x | y

        zero = core.Constant(0, x.width)
        allones = BvNot(zero)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
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
    See `Operation` for more information.

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
    is_simple = True
    infix_symbol = "^"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """XOR operation when both operands are int."""
            return x ^ y

        zero = core.Constant(0, x.width)
        allones = BvNot(zero)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
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

    Provides Automatic Constant Conversion. See `Operation` for more
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
    performs symbolic equality testing and it leaves the relation unevaluated
    if it cannot prove the objects are equal (or unequal).

        >>> Variable("x", 8) == Variable("y", 8)
        False
        >>> BvComp(Variable("x", 8), Variable("y", 8))  # symbolic equality
        x == y

    """

    arity = [2, 0]
    is_symmetric = True
    is_simple = True
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
    See `Operation` for more information.

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
    is_simple = True
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
    See `Operation` for more information.

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
    is_simple = True
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
    See `Operation` for more information.

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
    is_simple = True
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
    See `Operation` for more information.

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
    is_simple = True
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
    See `Operation` for more information.

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
    is_simple = True
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
        # y >= x.width doesn't evaluate even if y is a Constant
        elif isinstance(y, core.Constant) and int(y) >= x.width:
            return zero
        elif x == zero or y == zero:
            return x
        elif isinstance(x, BvShl) and isinstance(x.args[1], core.Constant) \
                and isinstance(y, core.Constant):
            # prevent out of bound
            r = min(x.args[0].width, int(x.args[1]) + int(y))
            return BvShl(x.args[0], core.Constant(r, x.width))


class BvLshr(Operation):
    """Logical right shift operation.

    It overrides >> and provides Automatic Constant Conversion.
    See `Operation` for more information.

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
    is_simple = True
    infix_symbol = ">>"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """Logical right shift operation when both operands are int."""
            return x >> y

        zero = core.Constant(0, x.width)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
        elif isinstance(y, core.Constant) and int(y) >= x.width:
            return zero
        elif x == zero or y == zero:
            return x
        elif isinstance(x, BvLshr) and isinstance(x.args[1], core.Constant) \
                and isinstance(y, core.Constant):
            r = min(x.args[0].width, int(x.args[1]) + int(y))
            return BvLshr(x.args[0], core.Constant(r, x.width))


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
    infix_symbol = "<<<"
    operand_types = [core.Term, int]

    @classmethod
    def condition(cls, x, r):
        return x.width > r >= 0

    @classmethod
    def output_width(cls, x, r):
        return x.width

    @classmethod
    def eval(cls, x, r):
        def doit(val, r, width):
            """Left cyclic rotation operation when both operands are int."""
            mask = 2 ** width - 1
            r = r % width
            return ((val << r) & mask) | ((val & mask) >> (width - r))

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

    It provides Automatic Constant Conversion. See `Operation` for more
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
    infix_symbol = ">>>"
    operand_types = [core.Term, int]

    @classmethod
    def condition(cls, x, r):
        return x.width > r >= 0

    @classmethod
    def output_width(cls, x, r):
        return x.width

    @classmethod
    def eval(cls, x, r):
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

    ``Ite(b, x, y)`` returns ``x`` if ``b == 0b1`` and ``y`` otherwise.

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

    ``Extract(t, i, j)`` extracts the bits from position ``i`` down
    position ``j`` (end points included, position 0 corresponding
    to the least significant bit).

    It overrides the operation [], that is, ``Extract(t, i, j)``
    is equivalent to ``t[i:j]``.

    Note that the indices can be omitted when they point the most
    significant bit or the least significant bit.
    For example, if ``t`` is a bit-vector of length ``n``,
    then ``t[n-1:j] = t[:j]`` and ``t[i:0] = t[i:]``

    Warning:
        In python, given a list ``l``, ``l[i:j]`` denotes the elements
        from position ``i`` up to (but no included) position ``j``.
        Note that with bit-vectors, the order of the arguments is
        swapped and both end points are included.

        For example, for a given list ``l`` and bit-vector ``t``,
        ``l[0:1] == l[0]`` and ``t[1:0] == (t[0], t[1])``.

    ::

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
        elif isinstance(x, (BvShl, RotateLeft)) and \
                isinstance(x.args[1], (int, core.Constant)) and x.args[1] <= j:
            # (x << 1)[:2] = x[n-2: 1]
            offset = int(x.args[1])
            return Extract(x.args[0], i - offset, j - offset)
        elif isinstance(x, (BvLshr, RotateRight)) and \
                isinstance(x.args[1], (int, core.Constant)) and i < x.width - x.args[1]:
            # (x >> 1)[n-3:] = x[n-2: 1]
            offset = int(x.args[1])
            return Extract(x.args[0], i + offset, j + offset)


class Concat(Operation):
    """Concatenation operation.

    Given the bit-vectors :math:`(x_{n-1}, \dots, x_0)` and
    :math:`(y_{m-1}, \dots, y_0)`, ``Concat(x, y)`` returns the bit-vector
    :math:`(x_{n-1}, \dots, x_0, y_{m-1}, \dots, y_0)`.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import Concat
        >>> Concat(Constant(0x12, 8), Constant(0x345, 12))
        0x12345
        >>> Concat(Variable("x", 8), Variable("y", 8))
        x :: y

    """

    arity = [2, 0]
    is_symmetric = False
    infix_symbol = "::"

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
        0x0 :: x

    """

    arity = [1, 1]
    is_symmetric = False
    alt_name = "ext"
    operand_types = [core.Term, int]

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
    """Concatenate a bit-vector with itself a given number of times.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import Repeat
        >>> Repeat(Constant(0x1, 4), 4)
        0x1111
        >>> Repeat(Variable("x", 8), 4)
        x :: x :: x :: x

    """

    arity = [1, 1]
    is_symmetric = False
    operand_types = [core.Term, int]

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
            # noinspection PyTypeChecker
            return functools.reduce(Concat, itertools.repeat(x, i))


# Arithmetic operators

class BvNeg(Operation):
    """Unary minus operation.

    It overrides the unary operator -. See `Operation` for more information.

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
    See `Operation` for more information.

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
    is_simple = True
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
        elif isinstance(x, BvSub):  # (x0 - x1) + y
            if x.args[1] == y:
                return x.args[0]
        elif isinstance(y, BvSub):  # x + (y0 - y1)
            if y.args[1] == x:
                return y.args[0]

    def _simplify(self, *args, **kwargs):
        compatible_terms = [
            lambda x: BvNeg(x)
        ]

        return self._binary_symmetric_simplification(compatible_terms)


class BvSub(Operation):
    """Modular subtraction operation.

    It overrides the operator - and provides Automatic Constant Conversion.
    See `Operation` for more information.

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.bitvector.operation import BvSub
        >>> BvSub(Constant(1, 8), Constant(2, 8))
        0xff
        >>> BvSub(Constant(1, 8), 2)
        0xff
        >>> Constant(1, 8) - 2
        0xff
        >>> Variable("x", 8) - Variable("y", 8)
        x - y

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = "-"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y, width):
            """Modular subtraction when both operands are integers."""
            return (x - y) % (2 ** width)

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y), x.width), x.width)

        zero = core.Constant(0, x.width)

        if x == zero:
            return BvNeg(y)
        elif y == zero:
            return x
        elif x == y:
            return zero
        elif isinstance(x, BvAdd):  # (x0 + x1) - y
            if x.args[0] == y:
                return x.args[1]
            elif x.args[1] == y:
                return x.args[0]
        elif isinstance(y, BvAdd):  # x - (y0 + y1)
            if y.args[0] == x:
                return BvNeg(y.args[1])
            elif y.args[1] == x:
                return BvNeg(y.args[0])


class BvMul(Operation):
    """Modular multiplication operation.

    It overrides the operator * and provides Automatic Constant Conversion.
    See `Operation` for more information.

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
    is_simple = True
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
    See `Operation` for more information.

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

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = "/"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """Division operation (truncated) when both operands are int."""
            assert y != 0
            return x // y

        zero = core.Constant(0, x.width)
        one = core.Constant(1, x.width)

        assert y != zero

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
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

    """

    arity = [2, 0]
    is_symmetric = False
    is_simple = True
    infix_symbol = "%"

    @classmethod
    def condition(cls, x, y):
        return x.width == y.width

    @classmethod
    def output_width(cls, x, y):
        return x.width

    @classmethod
    def eval(cls, x, y):
        def doit(x, y):
            """Remainder operation when both operands are int."""
            assert y != 0
            return x % y

        zero = core.Constant(0, x.width)
        one = core.Constant(1, x.width)

        assert y != zero

        if isinstance(x, core.Constant) and isinstance(y, core.Constant):
            return core.Constant(doit(int(x), int(y)), x.width)
        elif x == y or x == zero or y == one:
            return zero
