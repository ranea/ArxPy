"""The Context module provides context managers to modify the default behaviour."""
import collections
import contextlib

import bidict

from arxpy.bitvector import core


class StatefulContext(contextlib.AbstractContextManager):
    """Base class for context managers with history (i.e. stateful)."""

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

    This context manager controls whether or not the cache is used in
    bit-vector operations. By default, it is True.

    Note that the Cache context cannot be enabled when the Simplification/Evaluation
    context are disabled.

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
            assert StatefulExecution.current_context is False
        super().__enter__()


class Simplification(StatefulContext):
    """Control the Simplification context.

    This context manager controls whether or not bit-vector operations
    are simplified. By default, it is True.

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.bitvector.context import Simplification
        >>> x, y = Variable("x", 8), Variable("y", 8)
        >>> (x + y) - x
        y
        >>> with Simplification(False):
        ...     expr = (x + y) - x
        >>> expr
        (x + y) + -x

    When the Simplification context is disabled, the Cache context is
    also disabled.

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

    This context manager controls whether or not bit-vector operations
    are evaluated by default. By default, it is True.

        >>> from arxpy.bitvector.core import Constant
        >>> Constant(1, 8) + Constant(1, 8)
        0x02
        >>> with Evaluation(False):
        ...     expr = Constant(1, 8) + Constant(1, 8)
        >>> expr
        0x01 + 0x01
        >>> expr.doit()
        0x02

    When the Evaluation context is disabled, the Simplification and Cache
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

    This context manager controls whether or not arguments of
    bit-vector operators are validated.

    If this context is disabled, Automatic Constant Conversion is no longer
    available (see Operation).

        >>> from arxpy.bitvector.core import Constant
        >>> Constant(1, 8) + 1
        0x02
        >>> with Validation(False):
        ...     Constant(1, 5) + 2
        Traceback (most recent call last):
         ...
        AttributeError: 'int' object has no attribute 'width'

    """

    current_context = True

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context in [True, False]
        super().__init__(new_context)


class StatefulExecution(StatefulContext):
    """Control the StatefulExecution context.

    This context manager controls whether or not bit-vector operations
    are evaluated in the "stateful execution mode".

    In the "stateful execution mode", intermediate operations
    are stored in a look-up table (each one with an unique identifier) and
    the result of an operation returns its identifier instead of its value.
    In the "stateful execution mode", complex and huge symbolic
    expressions can be computed efficiently since the identifiers are
    used instead of the full expressions. By default, it is disabled.

    The  intermediate operations (with their identifiers) can
    be obtained from the ExecutionState object.

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.bitvector.context import StatefulExecution, ExecutionState
        >>> x, y, z = Variable("x", 8), Variable("y", 8), Variable("z", 8),
        >>> (x + y) ^ (z & (x | y))
        (x + y) ^ (z & (x | y))
        >>> st = ExecutionState()
        >>> with StatefulExecution(st):
        ...     expr = (x + y) ^ (z & (x | y))
        >>> expr
        x3
        >>> st
        ExecutionState([(x0, x + y), (x1, x | y), (x2, x1 & z), (x3, x0 ^ x2)])

    Another example:

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.bitvector.context import StatefulExecution, ExecutionState
        >>> x = Variable("x", 8)
        >>> expr = x
        >>> for i in range(5): expr += expr
        >>> expr  # doctest: +NORMALIZE_WHITESPACE
        ((((x + x) + (x + x)) + ((x + x) + (x + x))) + (((x + x) + (x + x)) +
        ((x + x) + (x + x)))) + ((((x + x) + (x + x)) + ((x + x) + (x + x))) +
        (((x + x) + (x + x)) + ((x + x) + (x + x))))

        >>> st = ExecutionState()
        >>> with StatefulExecution(st):
        ...     expr = x
        ...     for i in range(5): expr += expr
        >>> expr
        x4
        >>> st  # doctest: +NORMALIZE_WHITESPACE
        ExecutionState([(x0, x + x), (x1, x0 + x0), (x2, x1 + x1),
        (x3, x2 + x2), (x4, x3 + x3)])

    When StatefulExecution is enabled, the Simplification and Cache
    contexts are disabled.
    """

    current_context = None

    def __init__(self, new_context):
        """Initialize the context."""
        assert new_context is None or isinstance(new_context, ExecutionState)
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


class ExecutionState(collections.abc.MutableMapping):
    """Manage a ExecutionState.

    The ExecutionState stores all the intermediate operations
    with their identifier symbols.

    This class implements the usual methods of a dictionary and
    some additional methods.

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.bitvector.context import StatefulExecution, ExecutionState
        >>> x, y = Variable("x", 8), Variable("y", 8)
        >>> st = ExecutionState()
        >>> with StatefulExecution(st):
        ...     expr = ~((x + y) ^ x)
        >>> st
        ExecutionState([(x0, x + y), (x1, x ^ x0), (x2, ~x1)])
        >>> st[Variable("x0", 8)]
        x + y
        >>> list(st.items())
        [(x0, x + y), (x1, x ^ x0), (x2, ~x1)]
        >>> st.add_op(Variable("x2", 8) & Variable("x0", 8))
        x3
        >>> st.get_id(x + y)
        x0
        >>> st.replace_id(Variable("x0", 8), Variable("x_0", 8))
        >>> st
        ExecutionState([(x_0, x + y), (x1, x ^ x_0), (x2, ~x1), (x3, x2 & x_0)])

    """

    def __init__(self, id_prefix="x"):
        """Initialize an ExecutionState."""
        self.table = bidict.orderedbidict()
        self.counter = 0
        self.id_prefix = id_prefix

    def __getitem__(self, key):
        return self.table.__getitem__(key)

    def __setitem__(self, key, operation):
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

    def add_op(self, op):
        """Add an operation to the state and return its identifier."""
        from arxpy.bitvector import operation
        assert isinstance(op, operation.Operation)
        assert not self.contain_op(op)
        name = "{}{}".format(self.id_prefix, self.counter)
        self.counter += 1
        identifier = core.Variable(name, op.width)
        self.table[identifier] = op

        return identifier

    def get_id(self, operation):
        """Return the identifier of an intermediate operation."""
        return self.table.inv[operation]

    def contain_op(self, operation):
        """Check whether the operation is stored in the state."""
        return operation in self.table.inv

    def replace_id(self, old_id, new_id):
        """Replace the old id by the given new id."""
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

        self.table = bidict.orderedbidict(table)

    def clear(self):
        """Clear the state."""
        self.__init__()
