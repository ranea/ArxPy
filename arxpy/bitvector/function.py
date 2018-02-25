"""The Function module provides the data types to define bit-vector functions."""
import collections

from arxpy.bitvector import context
from arxpy.bitvector import core


class Function(object):
    """Base class to define *bit-vector functions*.

    A *bit-vector function* is a function with a fixed number of
    fixed-width bit-vector inputs and a fixed number of fixed-width
    bit-vector outputs.

    In other words, a bit-vector function specifies the following signature:

    - input_widths: a list containing the widths of the inputs.
    - output_widths:  a list containing the widths of the outputs.

    Bit-vector functions provides Automatic Constant Conversion;
    all integers arguments will be converted to bit-vectors automatically.

    Using the operator (), a bit-vector function is only evaluated
    if all the inputs are bit-vector contansts. Otherwise, an Exception
    is raised. To execute the bit-vector function with symbolic inputs,
    use the method symbolic_execution().

    To define a new bit-vector function, subclass Function, define
    the input/output widths and override the method eval().

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.bitvector.function import Function
        >>> class MyFunction(Function):
        ...     input_widths = [8, 8]
        ...     output_widths = [8, 8]
        ...     @classmethod
        ...     def eval(cls, x, y): return (x ^ y, x)
        >>> MyFunction(0, 0)
        (0x00, 0x00)
        >>> MyFunction(Variable("x", 8), Variable("y", 8))
        Traceback (most recent call last):
        ...
        TypeError: expected bit-vector constant arguments


    """

    def __new__(cls, *args, **options):
        assert len(cls.input_widths) == len(args)
        newargs = []
        for arg, width in zip(args, cls.input_widths):
            newargs.append(core.bitvectify(arg, width))
        args = newargs

        if all(isinstance(arg, core.Constant) for arg in args) or \
                options.pop("symbolic_inputs", False):
            result = cls.eval(*args)
        else:
            raise TypeError("expected bit-vector constant arguments")

        output = list(core.tuplify(result))
        assert len(cls.output_widths) == len(output)
        for i in range(len(output)):
            output[i] = core.bitvectify(output[i], cls.output_widths[i])

        if isinstance(result, collections.Sequence):
            return tuple(output)
        else:
            return output[0]

    @classmethod
    def symbolic_execution(cls, *args, id_prefix="x", st=None):
        """Evaluate symbolically the function.

        It evalutes the bit-vector function in the "stateful execution mode"
        (see StatefulExecution).

        It returns a pair (output, st) where output is the return
        value of the function and st is the ExecutionState containing
        the intermediate operations (see StatefulExecution).

        The ExecutionState can be also provided. Otherwise, a new
        ExecutionState will be used.

            >>> from arxpy.bitvector.core import Variable
            >>> from arxpy.bitvector.function import Function
            >>> class MyFunction(Function):
            ...     input_widths = [8, 8]
            ...     output_widths = [8, 8]
            ...     @classmethod
            ...     def eval(cls, a, b): return (a ^ b, b)
            >>> a, b = Variable("a", 8), Variable("b", 8)
            >>> MyFunction.symbolic_execution(a, b)
            ((x0, b), ExecutionState([(x0, a ^ b)]))

        """
        if st is None:
            st = context.ExecutionState(id_prefix=id_prefix)

        with context.StatefulExecution(st):
            output = cls(*args, symbolic_inputs=True)

        return output, st

    @classmethod
    def _symbolic_input(cls, symbol_prefix="i"):
        """Return a tuple of variables with proper input widths."""
        in_vars = []

        for i, width in enumerate(cls.input_widths):
            name = "{}{}".format(symbol_prefix, i)
            in_vars.append(core.Variable(name, width))

        return tuple(in_vars)


class CompositeFunction(Function):
    """Base class to define the composite functions.

    A composite function is the composition of two bit-vector functions.
    Given two bit-vector functions *outer* and *inner*,
    the composite function evaluates outer by fixing the last inputs
    of outer with the outputs of inner.

    For example, given outer(·, ·, ·) and inner(·) and assuming
    inner returns two bit-vectors, the resulting composite function
    is F(·, ·) = outer(·, inner(·)).

    A composite function has to specify:

     - outer_func: the "outer" bit-vector function
     - inner_func: the "inner" bit-vector function

    To define a new composite function, subclass CompositeFunction, define
    the outer/inner functions and the input/output widths of
    the composite function.

        >>> from arxpy.bitvector.core import Variable
        >>> from arxpy.bitvector.function import Function, CompositeFunction
        >>> class MyInner(Function):
        ...     input_widths = [8]
        ...     output_widths = [8, 8]
        ...     @classmethod
        ...     def eval(cls, x): return (x, x)
        >>> MyInner(1)
        (0x01, 0x01)
        >>> class MyOuter(Function):
        ...     input_widths = [8, 8, 8]
        ...     output_widths = [8, 8, 8]
        ...     @classmethod
        ...     def eval(cls, x, y, z): return (x, y, z)
        >>> MyOuter(1, 2, 3)
        (0x01, 0x02, 0x03)
        >>> class MyComposite(CompositeFunction):
        ...     input_widths = [8, 8]
        ...     output_widths = [8, 8, 8]
        ...     inner_func = MyInner
        ...     outer_func = MyOuter
        >>> MyComposite(0, 1)
        (0x00, 0x01, 0x01)

    """

    @classmethod
    def eval(cls, *args):
        inner_ninputs = len(cls.inner_func.input_widths)
        evaluated_inner = cls.inner_func(*args[-inner_ninputs:])
        evaluated_inner = core.tuplify(evaluated_inner)
        return cls.outer_func(*args[:-inner_ninputs], *evaluated_inner)

    @classmethod
    def _symbolic_input(cls, outer_symbol_prefix="o", inner_symbol_prefix="i"):
        """Return a tuple of variables with proper input widths."""
        inner_symbols = cls.inner_func._symbolic_input(inner_symbol_prefix)
        outer_symbols = cls.outer_func._symbolic_input(outer_symbol_prefix)
        inner_noutputs = len(cls.inner_func.output_widths)
        return outer_symbols[:-inner_noutputs] + inner_symbols

    @classmethod
    def symbolic_execution(cls, *args, id_prefix="x", st=None):
        """Evaluate symbolically the function.

        It evalutes the composite function in the "stateful execution mode"
        (see symbolic_execution of Function).

        It returns two pairs (inner_output, inner_st), (outer_outer, outer_st)
        where each pair is the result of calling symbolic_execution
        on inner and outer respectively.

            >>> from arxpy.bitvector.core import Variable
            >>> from arxpy.bitvector.function import Function, CompositeFunction
            >>> class MyInner(Function):
            ...     input_widths = [8, 8]
            ...     output_widths = [8]
            ...     @classmethod
            ...     def eval(cls, a, b): return (a ^ b)
            >>> MyInner(1, 1)
            0x00
            >>> class MyOuter(Function):
            ...     input_widths = [8, 8]
            ...     output_widths = [8]
            ...     @classmethod
            ...     def eval(cls, c, d): return (c + d)
            >>> MyOuter(1, 1)
            0x02
            >>> class MyComposite(CompositeFunction):
            ...     input_widths = [8, 8, 8]
            ...     output_widths = [8]
            ...     inner_func = MyInner
            ...     outer_func = MyOuter
            >>> MyComposite(1, 1, 1)
            0x01
            >>> a, b, cd = Variable("a", 8), Variable("b", 8), Variable("cd", 8)
            >>> MyComposite.symbolic_execution(a, b, cd)  # doctest: +NORMALIZE_WHITESPACE
            ((x0, ExecutionState([(x0, b ^ cd), (x1, a + x0)])),
            (x1, ExecutionState([(x0, b ^ cd), (x1, a + x0)])))

        """
        if st is None:
            st = context.ExecutionState(id_prefix=id_prefix)

        inner_ninputs = len(cls.inner_func.input_widths)
        inner_args = args[-inner_ninputs:]
        inner_exec = cls.inner_func.symbolic_execution(*inner_args, st=st)

        new_args = args[:-inner_ninputs] + core.tuplify(inner_exec[0])

        self_exec = cls.outer_func.symbolic_execution(*new_args, st=st)

        return inner_exec, self_exec
