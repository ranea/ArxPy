"""Provide the basic bit-vector types."""
import math

from sympy import preorder_traversal, Basic, Atom


class Term(Basic):
    """Represent bit-vector terms.

    Bit-vector terms are constants, variables and operations applied to terms.

    Bit-vector terms support many operations with the standard
    operators symbols (<=, +, ^, etc.). See `operation`
    for more information.

    This class is not meant to be instantiated but to provide a base
    class for the different types of terms.

    Note that Term inherits the methods of the SymPy class `Basic
    <http://docs.sympy.org/latest/modules/core.html#module-sympy.core.basic>`_;
    many of these methods work with bit-vector terms out of the box.

    .. Implementation details:

        Subclasses must implement the following methods:

        - __hash__() if eq is overridden.
        - _hashable_content() if new object attributes are defined.
        - class_key(), current order

            Constant: 1, 0, cls.__name__
            Variable: 2, 0, cls.__name__
            Function: 3, 0, cls.__name__

        Sympy's order:

            Number: 1, 0, cls.__name__
            Atom: 2, 0, cls.__name__
            Mul: 3, 0, cls.__name__
            Add: 3, 1, cls.__name__
            Pow: 3, 2, cls.__name__
            Function: 4, i, name
            Core: 5, 0, cls.__name__

    """

    # Bitwise operators

    def __invert__(self):
        """Override ~ operator."""
        from arxpy.bitvector import operation
        return operation.BvNot(self)

    def __and__(self, other):
        """Override & operator."""
        from arxpy.bitvector import operation
        return operation.BvAnd(self, other)

    __rand__ = __and__

    def __or__(self, other):
        """Override | operator."""
        from arxpy.bitvector import operation
        return operation.BvOr(self, other)

    __ror__ = __or__

    def __xor__(self, other):
        """Override ^ operator."""
        from arxpy.bitvector import operation
        return operation.BvXor(self, other)

    __rxor__ = __xor__

    # Relational operators

    def __lt__(self, other):
        """Override < operator."""
        from arxpy.bitvector import operation
        return operation.BvUlt(self, other)

    def __le__(self, other):
        """Override <= operator."""
        from arxpy.bitvector import operation
        return operation.BvUle(self, other)

    def __gt__(self, other):
        """Override > operator."""
        from arxpy.bitvector import operation
        return operation.BvUgt(self, other)

    def __ge__(self, other):
        """Override >= operator."""
        from arxpy.bitvector import operation
        return operation.BvUge(self, other)

    # Shifts

    def __lshift__(self, other):
        """Override << operator."""
        from arxpy.bitvector import operation
        return operation.BvShl(self, other)

    def __rlshift__(self, other):
        """Override reflected << operator."""
        from arxpy.bitvector import operation
        return operation.BvShl(other, self)

    def __rshift__(self, other):
        """Override >> operator."""
        from arxpy.bitvector import operation
        return operation.BvLshr(self, other)

    def __rrshift__(self, other):
        """Override reflected >> operator."""
        from arxpy.bitvector import operation
        return operation.BvLshr(other, self)

    # Arithmetic operators

    def __neg__(self):
        """Override unary minus - operator."""
        from arxpy.bitvector import operation
        return operation.BvNeg(self)

    def __add__(self, other):
        """Override + operator."""
        from arxpy.bitvector import operation
        return operation.BvAdd(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        """Override - operator."""
        from arxpy.bitvector import operation
        return operation.BvSub(self, other)

    def __rsub__(self, other):
        """Override other - operator."""
        from arxpy.bitvector import operation
        return operation.BvSub(other, self)

    def __mul__(self, other):
        """Override * operator."""
        from arxpy.bitvector import operation
        return operation.BvMul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Override / operator."""
        from arxpy.bitvector import operation
        return operation.BvUdiv(self, other)

    def __rtruediv__(self, other):
        """Override reflected / operator."""
        from arxpy.bitvector import operation
        return operation.BvUdiv(other, self)

    def __mod__(self, other):
        """Override % operator."""
        from arxpy.bitvector import operation
        return operation.BvUrem(self, other)

    def __rmod__(self, other):
        """Override reflected % operator."""
        from arxpy.bitvector import operation
        return operation.BvUrem(other, self)

    # end Boolean methods (relational methods from Expr)

    __slots__ = ["_width"]

    def __new__(cls, *args, width):
        assert isinstance(width, int) and 0 < width
        obj = Basic.__new__(cls, *args)
        obj._width = width
        return obj

    def __getitem__(self, key):
        """Override [] operator."""
        from arxpy.bitvector import operation

        if isinstance(key, slice):
            assert key.step is None or key.step == 1

            i = key.start if key.start is not None else self.width - 1
            if i < 0 or i >= self.width:
                raise IndexError("first index out of range")

            j = key.stop if key.stop is not None else 0
            if j < 0 or j >= self.width or j > i:
                raise IndexError("second index out of range")

            return operation.Extract(self, i, j)
        elif isinstance(key, int):
            if key < 0 or key >= self.width:
                raise IndexError("index out of range")
            return operation.Extract(self, key, key)
        else:
            raise TypeError("invalid index")

    def __iter__(self):
        # Necessary since __getitem__ is defined
        raise AttributeError("Term is not iterable")

    def __str__(self):
        """Return the non-verbose string representation."""
        from arxpy.bitvector import printing
        return (printing.BvStrPrinter()).doprint(self)

    __repr__ = __str__

    def _hashable_content(self):
        """Return the information of the object to compute its hash."""
        return self.args + (self.width, )

    @property
    def width(self):
        """The bit-width of the term."""
        return self._width

    @property
    def formula_size(self):
        """The formula size of the bit-vector term.

        As defined in `Complexity of Fixed-Size Bit-Vector Logics
        <https://doi.org/10.1007/s00224-015-9653-1>`_.
        """
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

    def vrepr(self):
        """Return a verbose string representation."""
        from arxpy.bitvector import printing
        return (printing.BvReprPrinter()).doprint(self)

    def srepr(self):
        """Return a short string representation."""
        from arxpy.bitvector import printing
        return (printing.BvShortPrinter()).doprint(self)

    def doit(self):
        """Evaluate the term.

        Terms are evaluated by default, but this behaviour can be disabled.
        See `Evaluation` for more information.
        """
        newargs = []
        for arg in self.args:
            if isinstance(arg, Term):
                newargs.append(arg.doit())
            else:
                newargs.append(arg)
        return type(self)(*newargs)

    # def is_subexpression(self, t):
    #     """Return True if the term is contained in the given expression.
    #
    #         >>> from arxpy.bitvector.core import Constant, Variable
    #         >>> t = Constant(1, 4) + Variable("v", 4)
    #         >>> Variable("v", 4).is_subexpression(t)
    #         True
    #         >>> Constant(2, 4).is_subexpression(t)
    #         False
    #
    #     """
    #     assert isinstance(t, Term)
    #     for sub in preorder_traversal(t):
    #         if self == sub:
    #             return True
    #     else:
    #         return False

    def class_key(self):
        """Return the key (identifier) of the class for sorting."""
        raise NotImplementedError("subclasses need to override this method")

    def atoms(self, *types):
        """Returns the atoms that form the current object.

        Similar to SymPy atoms() method, but this method
        doesn't throw an exception when one of the arguments
        is of type 'int'.
        """
        if types:
            types = tuple(
                [t if isinstance(t, type) else type(t) for t in types])
        nodes = preorder_traversal(self)
        if types:
            result = {node for node in nodes if isinstance(node, types)}
        else:
            result = {node for node in nodes if not isinstance(node, int) and not node.args}
        return result


class Constant(Atom, Term):
    """Represent bit-vector constants.

    Bit-vector constants are interpreted as unsigned integers in base 2,
    that is, a bit-vector :math:`(x_{n-1}, \dots, x_1, x_0)` represents
    the non-negative integer :math:`x_0 + 2 x_1 + \dots + 2^{n-1} x_{n-1}`.

    Args:
        val: the integer value.
        width: the bit-width.

    ::

        >>> from arxpy.bitvector.core import Constant
        >>> Constant(3, 12)
        0x003
        >>> Constant(0b11, 12)
        0x003
        >>> Constant(0x003, 12)
        0x003
        >>> Constant(3, 12).vrepr()
        'Constant(0b000000000011, width=12)'

    """

    def __int__(self):
        return self.val

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        """Override == operator."""
        if isinstance(other, int):
            return self.val == other
        elif isinstance(other, Constant) and self.width == other.width:
            return self.val == other.val
        else:
            return False

    # def __index__(self):
    #     """Return an int to be used inside a slice [ : : ]."""
    #     return self.int

    def _hashable_content(self):
        """Return a tuple of information about self to compute its hash."""
        return self.val, self.width

    @classmethod
    def class_key(cls):
        """Return the key (identifier) of the class for sorting."""
        return 1, 0, cls.__name__

    # end Integer

    __slots__ = ["_val"]

    def __new__(cls, val, width):
        assert isinstance(val, int) and 0 <= val < 2 ** width
        obj = Term.__new__(cls, width=width)
        obj._val = val
        return obj

    def __bool__(self):
        if self.width == 1:
            return self == Constant(1, 1)
        else:
            raise AttributeError("only 1-bit constants implement bool()")

    @property
    def val(self):
        """The integer represented by the bit-vector constant."""
        return self._val

    @property
    def formula_size(self):
        """The formula size of the constant."""
        def log2(n):
            return int(math.ceil(math.log(n, 2)))

        def bin_enc(n):
            return 1 + log2(n + 1)

        return 1 + log2(int(self) + 1) + bin_enc(self.width)

    def bin(self):
        """Return the binary representation.

            >>> from arxpy.bitvector.core import Constant
            >>> print(Constant(3, 4).bin())
            0b0011
            >>> print(Constant(4, 6).bin())
            0b000100

        """
        width = self.width + 2  # 2 due to '0b'
        return format(self.val, r'0=#{}b'.format(width))

    def hex(self):
        """Return the hexadecimal representation.

            >>> from arxpy.bitvector.core import Constant
            >>> print(Constant(3, 4).hex())
            0x3

        """
        assert self.width % 4 == 0
        width = (self.width // 4) + 2
        return format(self.val, '0=#{}x'.format(width))

    def oct(self):
        """Return the octal representation.

            >>> from arxpy.bitvector.core import Constant
            >>> print(Constant(4, 6).oct())
            0o04

        """
        assert self.width % 3 == 0
        width = (self.width // 3) + 2
        return format(self.val, '0=#{}o'.format(width))


class Variable(Atom, Term):
    """Represent bit-vector variables.

    Args:
        name: the name of the variable.
        width: the bit-width.

    ::

        >>> from arxpy.bitvector.core import Variable
        >>> Variable("x", 12)
        x
        >>> Variable("x", 12).vrepr()
        "Variable('x', width=12)"

    """

    def _hashable_content(self):
        """Return a tuple of information about self to compute hash."""
        return self.name, self.width

    # def __call__(self, *args):
    #     from sympy.core.function as function
    #     return function.UndefinedFunction(self.name, self.width)(*args)

    # end Symbol

    __slots__ = ['_name']

    def __new__(cls, name, width):
        assert isinstance(name, str)
        obj = Term.__new__(cls, width=width)
        obj._name = name

        return obj

    @property
    def name(self):
        """The name of the variable."""
        return self._name

    @property
    def formula_size(self):
        """The formula size of the variable."""
        def log2(n):
            return int(math.ceil(math.log(n, 2)))

        def bin_enc(n):
            return 1 + log2(n + 1)

        return 1 + bin_enc(self.width)


def bitvectify(t, width):
    """Convert the argument *t* to a bit-vector of bit-width *width*.

        >>> from arxpy.bitvector.core import bitvectify
        >>> print(bitvectify(0, 8).vrepr())
        Constant(0b00000000, width=8)
        >>> print(bitvectify("x", 8).vrepr())
        Variable('x', width=8)

    """
    if isinstance(t, int):
        return Constant(t, width)
    elif isinstance(t, str):
        return Variable(t, width)
    elif isinstance(t, Term):
        assert t.width == width
        return t
    else:
        msg = "cannot convert '{}' to a bit-vector"
        raise TypeError(msg.format(type(t).__name__))
