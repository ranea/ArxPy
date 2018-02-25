"""The Core module provides the basic bit-vector types."""
import collections
import math

from sympy.core import basic
from sympy.core import symbols


class Term(basic.Basic):
    """Base class to define bit-vector terms.

    Bit-vector terms are constants, variables and operations and
    uninterpreted functions applied to terms. The simplest terms
    (constans and variables) are also called atoms.

    For these data types,  the standard comparison operators (==, <=, etc.)
    and arithmetic operators (+, <<, &, ^, etc.) have been overriden with
    the common bit-vector operators. See the module operators for the
    full list.

    Using vector notation, a bit-vector term t of width n is represented as
    (t[n-1], t[n-2], ..., t[1], t[0]), where the most significant bit (MSB)
    standing on the left-hand side and the least significant bit (LSB)
    on the right-hand side. The operator [] can be used to extract
    the bits of a bit-vector term. See the operator extract for
    more information.

    The concepts bit-vector term, term and bit-vector are used
    interchangeably.

    .. Implementation details:

        Subclasses must implement the following methods:

        - __hash__() if eq is overriden.
        - _hashable_content() if new object attributes are defined.
        - class_key() similarly to SymPy's order:

            Number: 1, 0, cls.__name__
            Atom: 2, 0, cls.__name__
            Mul: 3, 0, cls.__name__
            Add: 3, 1, cls.__name__
            Pow: 3, 2, cls.__name__
            Function: 4, i, name
            Core: 5, 0, cls.__name__

        __getitem__ is overriden to support slices but len() isn't to
        prevent side effects.

        Operations apply to terms (the most general object) so their
        python operators are overriden here.

    """

    # Bitwise operators

    def __invert__(self):
        """Overriding for ~ operator."""
        from arxpy.bitvector import operation
        return operation.BvNot(self)

    def __and__(self, other):
        """Overriding for & operator."""
        from arxpy.bitvector import operation
        return operation.BvAnd(self, other)

    __rand__ = __and__

    def __or__(self, other):
        """Overriding for | operator."""
        from arxpy.bitvector import operation
        return operation.BvOr(self, other)

    __ror__ = __or__

    def __xor__(self, other):
        """Overriding for ^ operator."""
        from arxpy.bitvector import operation
        return operation.BvXor(self, other)

    __rxor__ = __xor__

    # Relational operators

    def __lt__(self, other):
        """Overriding for < operator."""
        from arxpy.bitvector import operation
        return operation.BvUlt(self, other)

    def __le__(self, other):
        """Overriding for <= operator."""
        from arxpy.bitvector import operation
        return operation.BvUle(self, other)

    def __gt__(self, other):
        """Overriding for > operator."""
        from arxpy.bitvector import operation
        return operation.BvUgt(self, other)

    def __ge__(self, other):
        """Overriding for >= operator."""
        from arxpy.bitvector import operation
        return operation.BvUge(self, other)

    # Shifts

    def __lshift__(self, other):
        """Overriding for << operator."""
        from arxpy.bitvector import operation
        return operation.BvShl(self, other)

    def __rlshift__(self, other):
        """Overriding for reflected << operator."""
        from arxpy.bitvector import operation
        return operation.BvShl(other, self)

    def __rshift__(self, other):
        """Overriding for >> operator."""
        from arxpy.bitvector import operation
        return operation.BvLshr(self, other)

    def __rrshift__(self, other):
        """Overriding for reflected >> operator."""
        from arxpy.bitvector import operation
        return operation.BvLshr(other, self)

    # Arithmetic operators

    def __neg__(self):
        """Overriding for unary minus - operator."""
        from arxpy.bitvector import operation
        return operation.BvNeg(self)

    def __add__(self, other):
        """Overriding for + operator."""
        # local import to prevent circular import
        from arxpy.bitvector import operation
        return operation.BvAdd(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        """Overriding for - operator."""
        from arxpy.bitvector import operation
        return operation.BvSub(self, other)

    def __rsub__(self, other):
        """Overriding for other - operator."""
        from arxpy.bitvector import operation
        return operation.BvSub(other, self)

    def __mul__(self, other):
        """Overriding for * operator."""
        from arxpy.bitvector import operation
        return operation.BvMul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        """Overriding for / operator."""
        from arxpy.bitvector import operation
        return operation.BvUdiv(self, other)

    def __rtruediv__(self, other):
        """Overriding for reflected / operator."""
        from arxpy.bitvector import operation
        return operation.BvUdiv(other, self)

    def __mod__(self, other):
        """Overriding for % operator."""
        from arxpy.bitvector import operation
        return operation.BvUrem(self, other)

    def __rmod__(self, other):
        """Overriding for reflected % operator."""
        from arxpy.bitvector import operation
        return operation.BvUrem(other, self)

    # end Boolean methods (relational were in Expr)

    __slots__ = ["_width"]

    def __new__(cls, *args, width):
        """Create the object."""
        assert isinstance(width, int) and 0 < width
        obj = basic.Basic.__new__(cls, *args)
        obj._width = width
        return obj

    @property
    def width(self):
        """The bit-width of the term."""
        return self._width

    def _hashable_content(self):
        """Return the information of the object to compute its hash."""
        return self.args + (self.width, )

    def __str__(self):
        """Return the non-verbose representation as a string."""
        from arxpy.bitvector import printing
        return (printing.BvStrPrinter()).doprint(self)

    __repr__ = __str__

    def vrepr(self):
        """Return the verbose representation as a string."""
        from arxpy.bitvector import printing
        return (printing.BvReprPrinter()).doprint(self)

    # _sorted_args = NotImplemented

    def doit(self):
        """Evaluate objects that are not evaluated."""
        newargs = []
        for arg in self.args:
            if isinstance(arg, Term):
                newargs.append(arg.doit())
            else:
                newargs.append(arg)
        return type(self)(*newargs)

    # methods of Core disabled to prevent side effects

    def _sympy_method_not_implemented(self, *args, **kwargs):
        return NotImplementedError("SymPy method not supported.")

    __reduce_ex__ = _sympy_method_not_implemented

    __getnewargs__ = _sympy_method_not_implemented

    __getstate__ = _sympy_method_not_implemented

    __setstate__ = _sympy_method_not_implemented

    copy = _sympy_method_not_implemented

    dummy_eq = _sympy_method_not_implemented

    free_symbols = _sympy_method_not_implemented

    expr_free_symbols = _sympy_method_not_implemented

    canonical_variables = _sympy_method_not_implemented

    rcall = _sympy_method_not_implemented

    _recursive_call = _sympy_method_not_implemented

    is_hypergeometric = _sympy_method_not_implemented

    is_comparable = _sympy_method_not_implemented

    as_poly = _sympy_method_not_implemented

    as_content_primitive = _sympy_method_not_implemented

    find = _sympy_method_not_implemented

    matches = _sympy_method_not_implemented

    match = _sympy_method_not_implemented

    count_ops = _sympy_method_not_implemented

    _eval_rewrite = _sympy_method_not_implemented

    rewrite = _sympy_method_not_implemented

    count = _sympy_method_not_implemented

    # end Core methods

    def __getitem__(self, key):
        """Overriding for [] operator."""
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
            raise TypeError("invalid tindex")

    def __iter__(self):
        raise AttributeError("Term is not iterable")

    def is_subexpression(self, t):
        """Return True if the term is contained in the given expression.

            >>> from arxpy.bitvector.core import Constant, Variable
            >>> t = Constant(1, 4) + Variable("v", 4)
            >>> Variable("v", 4).is_subexpression(t)
            True
            >>> Constant(2, 4).is_subexpression(t)
            False

        """
        assert isinstance(t, Term)
        for sub in basic.preorder_traversal(t):
            if self == sub:
                return True
        else:
            return False

    @property
    def formula_size(self):
        """The formula size of the bit-vector term.

        The formula size of a term is the size of the term in a
        bit-vector formula.
        """
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


class Constant(basic.Atom, Term):
    """Represent and handles bit-vector constants.

        >>> from arxpy.bitvector.core import Constant
        >>> Constant(3, 12)
        0x003
        >>> Constant(0b11, 12)
        0x003
        >>> Constant(0x003, 12)
        0x003
        >>> Constant(3, 12).vrepr()
        'Constant(0b000000000011, width=12)'

    See the parent class Term for the common methods available for terms.
    """

    def _sympy_method_not_implemented(self, *args, **kwargs):
        return NotImplementedError("SymPy method not supported.")

    matches = _sympy_method_not_implemented

    is_Atom = True  # redefined

    # end Atom

    # is_number = True
    # is_integer = True
    # is_Integer = True

    __slots__ = ["_val"]

    @property
    def val(self):
        """Natural number represented by the bitvector."""
        return self._val

    def __new__(cls, val, width):
        """Create the object."""
        assert isinstance(val, int) and 0 <= val and val < 2 ** width
        obj = Term.__new__(cls, width=width)
        obj._val = val
        return obj

    def __int__(self):
        return self.val

    def __hash__(self):
        return super().__hash__()

    # def __index__(self):
    #     """Return an int to be used inside a slice [ : : ]."""
    #     return self.int

    # end Integer

    # is_Number = True

    @classmethod
    def class_key(cls):
        """Return the identifier used for sorting."""
        return 1, 0, cls.__name__

    # end Number

    def _hashable_content(self):
        """Return a tuple of information about self to compute its hash."""
        return self.val, self.width

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

    def __bool__(self):
        if self.width == 1:
            return self == Constant(1, 1)
        else:
            raise AttributeError("only 1-bit constants overrides bool()")

    def __eq__(self, other):
        """Overriding for == operator."""
        if isinstance(other, int):
            return self.val == other
        elif isinstance(other, Constant) and self.width == other.width:
            return self.val == other.val
        else:
            return False

    @property
    def formula_size(self):
        """The formula size of the constant."""
        def L(n):
            return int(math.ceil(math.log(n, 2)))

        def bin_enc(n):
            return 1 + L(n + 1)

        return 1 + L(int(self) + 1) + bin_enc(self.width)


class Variable(basic.Atom, Term):
    """Represent and handles bit-vector variables.

        >>> from arxpy.bitvector.core import Variable
        >>> Variable("x", 8)
        x
        >>> Variable("x", 8).vrepr()
        "Variable('x', width=8)"

    See the parent class Term for the common methods available for terms.

    .. Implementation details:

        No assumptions supported.
        By default, commutative = False.
        A Variable is identified by name and width.
    """

    def _sympy_method_not_implemented(self, *args, **kwargs):
        return NotImplementedError("SymPy method not supported.")

    matches = _sympy_method_not_implemented

    is_Atom = True

    # end basic.Atom

    # is_comparable = False

    __slots__ = ['_name']

    @property
    def name(self):
        """Name or identifier of the symbol."""
        return self._name

    # is_Symbol = True
    # is_symbol = True

    def __new__(cls, name, width):
        """Create the object."""
        assert isinstance(name, str)
        obj = Term.__new__(cls, width=width)
        obj._name = name

        return obj

    def _hashable_content(self):
        """Return a tuple of information about self to compute hash."""
        return (self.name, self.width)

    # def __call__(self, *args):
    #     from sympy.core.function as spfun
    #     return spfun.UndefinedFunction(self.name, self.width)(*args)

    # end Symbol

    @property
    def formula_size(self):
        """The formula size of the variable."""
        def L(n):
            return int(math.ceil(math.log(n, 2)))

        def bin_enc(n):
            return 1 + L(n + 1)

        return 1 + bin_enc(self.width)


def bitvectify(t, width):
    """Convert the argument to a bit-vector of given width.

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


def _vars(names, width, cls=Variable):
    """Return a list of variables of given names and common width.

        >>> from arxpy.bitvector.core import _vars
        >>> l = _vars('x y z', 8)
        >>> l
        (x, y, z)
        >>> l[0].vrepr()
        "Variable('x', width=8)"
        >>> _vars("x0:10", width=8)
        (x0, x1, x2, x3, x4, x5, x6, x7, x8, x9)
    """
    return symbols(names, cls=cls, width=width)


def tuplify(seq):
    """Return seq as a tuple if it wasn't a sequence.

        >>> from arxpy.bitvector.core import tuplify, Constant
        >>> tuplify(Constant(3, 8))
        (0x03,)
        >>> tuplify([Constant(3, 8)])
        (0x03,)

    """
    if isinstance(seq, collections.Sequence):
        return tuple(seq)
    else:
        return (seq, )
