"""Provide context managers to modify the default behaviour."""
import collections
import contextlib

import bidict

from arxpy.bitvector import core


class StatefulContext(contextlib.AbstractContextManager):
    """Base class for context managers with history."""

    current_context = None

    def __init__(self, new_context):
        """Initialize the context."""
        self.new_context = new_context

    def __enter__(self):
        self.previous_context = type(self).current_context
        type(self).current_context = self.new_context
        # return self.new_context

    def __exit__(self, *args):
        type(self).current_context = self.previous_context


class Cache(StatefulContext):
    """Control the Cache context.

    Control whether or not the cache is used operating with bit-vectors.
    By default, the cache is enabled.

    Note that the Cache context cannot be enabled when the
    `Simplification` or `Evaluation` context are disabled.
    """

    current_context = True

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context in [True, False]
        super().__init__(new_context)

    def __enter__(self):
        if self.new_context is True:
            assert Simplification.current_context is True
            assert Evaluation.current_context is True
            assert Memoization.current_context is False
        super().__enter__()


class Simplification(StatefulContext):
    """Control the Simplification context.

    Control whether or not bit-vector expressions are automatically simplified.
    By default, automatic simplification is enabled.

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.bitvector.context import Simplification
        >>> x, y = Variable("x", 8), Variable("y", 8)
        >>> (x | y) | x
        x | y
        >>> with Simplification(False):
        ...     expr = (x | y) | x
        >>> expr
        x | x | y

    When the Simplification context is disabled, the `Cache` context is
    also disabled.

    Note:
        Disabling `Simplification` and `Validation` speeds up
        non-symbolic computations with bit-vectors.
    """

    current_context = True

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context in [True, False]
        super().__init__(new_context)

    def __enter__(self):
        if self.new_context is False:
            self.cache_context = Cache(False)
            self.cache_context.__enter__()
        super().__enter__()

    def __exit__(self, *args):
        if self.new_context is False:
            self.cache_context.__exit__()
        super().__exit__()


class Evaluation(StatefulContext):
    """Control the Evaluation context.

    Control whether or not bit-vector operations are evaluated.
    By default, bit-vector expressions are evaluated.

        >>> from arxpy.bitvector.core import Constant
        >>> from arxpy.bitvector.context import Evaluation
        >>> Constant(1, 8) + Constant(1, 8)
        0x02
        >>> with Evaluation(False):
        ...     expr = Constant(1, 8) + Constant(1, 8)
        >>> expr
        0x01 + 0x01
        >>> expr.doit()
        0x02

    When the Evaluation context is disabled, the `Simplification` and `Cache`
    contexts are also disabled.
    """

    current_context = True

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context in [True, False]
        super().__init__(new_context)

    def __enter__(self):
        if self.new_context is False:
            self.simplify_context = Simplification(False)
            self.simplify_context.__enter__()
        super().__enter__()

    def __exit__(self, *args):
        if self.new_context is False:
            self.simplify_context.__exit__()
        super().__exit__()


class Validation(StatefulContext):
    """Control the Validation context.

    Control whether or not arguments of bit-vector operators are validated.
    By default, validation of arguments is enabled.

    Note that when it is disabled,  Automatic Constant Conversion is no longer
    available (see `Operation`).

        >>> from arxpy.bitvector.core import Constant
        >>> from arxpy.bitvector.context import Validation
        >>> Constant(1, 8) + 1
        0x02
        >>> with Validation(False):
        ...     Constant(1, 5) + 2
        Traceback (most recent call last):
         ...
        AttributeError: 'int' object has no attribute 'width'

    Note:
        Disabling `Simplification` and `Validation` speeds up
        non-symbolic computations with bit-vectors.
    """

    current_context = True

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context in [True, False]
        super().__init__(new_context)


class Memoization(StatefulContext):
    """Control the Memoization context.

    Control whether or not bit-vector operations are evaluated in the
    *memoization mode*. By default, it is disabled.

    In the memoization mode, the result of each bit-vector operation is
    stored in a table (with an unique identifier). When the same inputs
    occurs again, the result is retrieved from the table. See also
    `Memoization <https://en.wikipedia.org/wiki/Memoization>`_.

    Note that in the memoization mode, bit-vector operations don't return
    the actual values but their identifiers in the memoization table.
    The actual values can be obtained from the `MemoizationTable`.

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.bitvector.context import Memoization, MemoizationTable
        >>> x, y, z = Variable("x", 8), Variable("y", 8), Variable("z", 8),
        >>> ~((x + y) ^ (z & y))
        ~((x + y) ^ (y & z))
        >>> lut = MemoizationTable()
        >>> with Memoization(lut):
        ...     expr = ~((x + y) ^ (z & y))
        >>> expr
        x3
        >>> lut
        MemoizationTable([(x0, x + y), (x1, y & z), (x2, x0 ^ x1), (x3, ~x2)])

    The Memoization context is useful to efficiently compute large symbolic
    expressions since the identifiers are used instead of the full expressions.

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.bitvector.context import Memoization, MemoizationTable
        >>> x = Variable("x", 8)
        >>> expr = x
        >>> for i in range(3): expr += expr
        >>> expr
        x + x + x + x + x + x + x + x
        >>> lut = MemoizationTable()
        >>> with Memoization(lut):
        ...     expr = x
        ...     for i in range(3): expr += expr
        >>> expr
        x2
        >>> lut  # doctest: +NORMALIZE_WHITESPACE
        MemoizationTable([(x0, x + x), (x1, x0 + x0), (x2, x1 + x1)])

    When the Memoization context is enabled, the `Simplification` and `Cache`
    contexts are disabled.
    """

    current_context = None

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context is None or isinstance(new_context, MemoizationTable)
        super().__init__(new_context)

    def __enter__(self):
        if self.new_context is not None:
            self.simplify_context = Simplification(False)
            self.simplify_context.__enter__()
        super().__enter__()

    def __exit__(self, *args):
        if self.new_context is not None:
            self.simplify_context.__exit__()
        super().__exit__()


class MemoizationTable(collections.abc.MutableMapping):
    """Store bit-vector expressions with unique identifiers.

    The MemoizationTable is a dictionary-like structure
    (implementing the usual methods of a dictionary and
    some additional methods) used for evaluating bit-vector operations
    in the *memoization mode* (see `Memoization`).

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.bitvector.context import Memoization, MemoizationTable
        >>> x, y = Variable("x", 8), Variable("y", 8)
        >>> lut = MemoizationTable()
        >>> with Memoization(lut):
        ...     expr = ~(x + y)
        >>> lut
        MemoizationTable([(x0, x + y), (x1, ~x0)])
        >>> lut[Variable("x0", 8)]
        x + y
        >>> lut.get_id(x + y)
        x0
        >>> lut.add_op(Variable("x1", 8) & Variable("z", 8))
        x2
        >>> lut
        MemoizationTable([(x0, x + y), (x1, ~x0), (x2, x1 & z)])
        >>> lut.replace_id(Variable("x0", 8), Variable("x_0", 8))
        >>> lut
        MemoizationTable([(x_0, x + y), (x1, ~x_0), (x2, x1 & z)])
    """

    def __init__(self, id_prefix="x"):
        """Initialize an MemoizationTable."""
        self.table = bidict.OrderedBidict()
        self.counter = 0
        self.id_prefix = id_prefix

    def __getitem__(self, key):
        return self.table.__getitem__(key)

    def __setitem__(self, key, expr):
        raise AttributeError("use add_op and replace_id instead")

    def __delitem__(self, key):
        assert all(key not in op.atoms() for op in self.table.values())
        return self.table.__delitem__(key)

    def __len__(self):
        return self.table.__len__()

    def __iter__(self):
        return self.table.__iter__()

    def __str__(self):
        return '{0}({1})'.format(type(self).__name__, list(self.table.items()))

    __repr__ = __str__

    def add_op(self, expr):
        """Add an bit-vector expression and return its identifier."""
        from arxpy.bitvector import operation
        assert isinstance(expr, operation.Operation)
        assert not self.contain_op(expr)
        name = "{}{}".format(self.id_prefix, self.counter)
        self.counter += 1
        identifier = core.Variable(name, expr.width)
        self.table[identifier] = expr

        return identifier

    def get_id(self, expr):
        """Return the identifier of a bit-vector expression."""
        return self.table.inv[expr]

    def contain_op(self, expr):
        """Check if the bit-vector expression is stored."""
        return expr in self.table.inv

    def replace_id(self, old_id, new_id):
        """Replace the old identifier by the given new identifier."""
        assert isinstance(old_id, core.Variable)
        assert isinstance(new_id, core.Variable)
        assert old_id in self.table and new_id not in self.table

        table = list(self.table.items())

        for i, (key, op) in enumerate(table):
            if key == old_id:
                new_key = new_id
            else:
                new_key = key

            table[i] = (new_key, op.xreplace({old_id: new_id}))

        self.table = bidict.OrderedBidict(table)

    def clear(self):
        """Empty the table."""
        self.__init__()


class NotEvaluation(StatefulContext):
    """Control the NotEvaluation context.

    Control whether or not some operations are not evaluated.
    By default, all operations are evaluated.

        >>> from arxpy.bitvector.core import Constant
        >>> from arxpy.bitvector.extraop import PopCount
        >>> from arxpy.bitvector.context import NotEvaluation
        >>> PopCount(Constant(0b010, 3) + Constant(0b001, 3))
        0b10
        >>> with NotEvaluation(PopCount):
        ...     expr = PopCount(Constant(0b010, 3) + Constant(0b001, 3))
        >>> expr
        PopCount(0b011)
        >>> expr.doit()
        0b10

    When the NotEvaluation context is enable, the `Cache` is disabled.
    """

    current_context = None

    def __init__(self, new_context):
        """Initialize the context."""
        super().__init__(new_context)

    def __enter__(self):
        if self.new_context is not None:
            self.cache_context = Cache(False)
            self.cache_context.__enter__()
        super().__enter__()

    def __exit__(self, *args):
        if self.new_context is not None:
            self.cache_context.__exit__()
        super().__exit__()
