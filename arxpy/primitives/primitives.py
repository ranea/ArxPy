"""Represent symmetric primitives."""
import collections
import warnings

from arxpy.bitvector import context
from arxpy.bitvector import core
from arxpy.bitvector import operation


class BvFunction(object):
    """Represent (iterated) fixed-width bit-vector functions.

    A `BvFunction` takes fixed-width `Constant` operands and return a
    tuple of fixed-width `Constant`. An iterated bit-vector function
    contains a subroutine that is iterated a certain number of *rounds*,
    which can be changed using `set_rounds`.

    Similar to `Operation`, `BvFunction` is evaluated
    using the operator ``()`` and provides *Automatic Constant Conversion*.
    Note that `BvFunction` only accepts `Constant` operands and
    always return a tuple, as opposed to `Operation` that accepts
    `Term` and scalar operands and returns a single `Term`.

        >>> from arxpy.primitives.primitives import BvFunction
        >>> from arxpy.primitives.chaskey import ChaskeyPi
        >>> issubclass(ChaskeyPi, BvFunction)
        True
        >>> ChaskeyPi(0, 0, 0, 0)  # automatic conversion from int to Constant
        (0x00000000, 0x00000000, 0x00000000, 0x00000000)

    Attributes:
        input_widths: a list containing the widths of the inputs
        output_widths: a list containing the widths of the outputs
        rounds: the number of iterations

    """
    input_widths = None
    output_widths = None
    rounds = None

    def __new__(cls, *args, **options):
        if len(cls.input_widths) != len(args):
            raise ValueError("{} requires {} inputs but {} were given: {}".format(
                cls.__name__, len(cls.input_widths), len(args), args))
        newargs = []
        for arg, width in zip(args, cls.input_widths):
            newargs.append(core.bitvectify(arg, width))
        args = newargs

        if all(isinstance(arg, core.Constant) for arg in args) or \
                options.pop("symbolic_inputs", False):
            result = cls.eval(*args)
        else:
            raise TypeError("expected bit-vector constant arguments")

        assert isinstance(result, collections.abc.Sequence)
        assert len(cls.output_widths) == len(result)

        output = []
        for r, width in zip(result, cls.output_widths):
            output.append(core.bitvectify(r, width))

        return tuple(output)

    @classmethod
    def eval(cls, *args):
        """Evaluate the function (internal method)."""
        raise NotImplementedError("subclasses need to override this method")

    @classmethod
    def set_rounds(cls, new_rounds):
        """Change the number of rounds and adjust the input/output widths."""
        raise NotImplementedError("subclasses need to override this method")

    # noinspection PyArgumentList
    @classmethod
    def ssa(cls, input_names, id_prefix):
        """Return a static single assignment program representing the function.

        Args:
            input_names: the names  for the input variables
            id_prefix: the prefix to denote the intermediate variables

        Return:
            : a dictionary with three keys

            - *input_vars*: a list of `Variable` representing the inputs
            - *output_vars*: a list of `Variable` representing the outputs
            - *assignments*: an ordered sequence of pairs
              (`Variable`, `Operation`) representing each assignment
              of the SSA program.

        ::

                >>> from arxpy.primitives.chaskey import ChaskeyPi
                >>> ChaskeyPi.set_rounds(1)
                >>> ChaskeyPi.ssa(["v0", "v1", "v2", "v3"], "x")  # doctest: +NORMALIZE_WHITESPACE
                {'input_vars': (v0, v1, v2, v3),
                'output_vars': (x7, x12, x13, x9),
                'assignments': ((x0, v0 + v1), (x1, v1 <<< 5), (x2, x0 ^ x1), (x3, x0 <<< 16), (x4, v2 + v3),
                (x5, v3 <<< 8), (x6, x4 ^ x5), (x7, x3 + x6), (x8, x6 <<< 13), (x9, x7 ^ x8), (x10, x2 + x4),
                (x11, x2 <<< 7), (x12, x10 ^ x11), (x13, x10 <<< 16))}

        """
        input_vars = []
        for name, width in zip(input_names, cls.input_widths):
            input_vars.append(core.Variable(name, width))
        input_vars = tuple(input_vars)

        table = context.MemoizationTable(id_prefix=id_prefix)

        with context.Memoization(table):
            # noinspection PyArgumentList
            output_vars = cls(*input_vars, symbolic_inputs=True)

        ssa_dict = {
            "input_vars": input_vars,
            "output_vars": output_vars,
            "assignments": tuple(table.items())
        }

        for var, expr in ssa_dict["assignments"]:
            for arg in expr.args:
                if isinstance(arg, operation.Operation):
                    raise ValueError("assignment {} <- {} was not decomposed".format(var, expr))

        to_delete = []
        vars_needed = set()
        for var in output_vars:
            vars_needed.add(var)
        for var, expr in reversed(ssa_dict["assignments"]):
            if var in vars_needed:
                for arg in expr.atoms(core.Variable):
                    vars_needed.add(arg)
            else:
                to_delete.append((var, expr))

        input_vars_not_used = [v for v in input_vars if v not in vars_needed]
        if input_vars_not_used:
            warnings.warn("found unused input vars {} in \n{}".format(input_vars_not_used, ssa_dict))

        if to_delete:
            warnings.warn("removing redundant assignments {} in \n{}".format(to_delete, ssa_dict))
            ssa_dict["assignments"] = list(ssa_dict["assignments"])
            for assignment in to_delete:
                ssa_dict["assignments"].remove(assignment)
            ssa_dict["assignments"] = tuple(ssa_dict["assignments"])

        if hasattr(cls, "round_keys") and cls.round_keys is not None:
            rk_not_used = [k for k in cls.round_keys if k not in vars_needed]
            if rk_not_used:
                warnings.warn("found round keys {} not used in {}\n{}".format(rk_not_used, cls.__name__, ssa_dict))

        return ssa_dict


# noinspection PyAbstractClass
class KeySchedule(BvFunction):
    """Represent key schedule functions.

    A key schedule function is a `BvFunction` that takes
    the masterkey as input and returns the round keys.
    See `BvFunction` for more information.
    """


# noinspection PyAbstractClass
class Encryption(BvFunction):
    """Represent encryption functions.

    An encryption function is a `BvFunction` that takes
    the plaintext as input and returns the ciphertext
    for some fixed key.
    See `BvFunction` for more information.

    Attributes:
        round_keys: a list of `Term` representing the round keys

    """
    round_keys = None


class Cipher(object):
    """Represent (iterated) block ciphers.

    A (iterated) block cipher consists of `KeySchedule` function
    that computes round keys from a master key and an `Encryption`
    function that computes a ciphertext from a given plaintext
    and the round keys.

    Given a ``cipher``, it can be evaluated with the operator ``()``
    by passing it as arguments the plaintext and the master key,
    that is, ``cipher(plaintext, masterkey)`` returns the ciphertext.

        >>> from arxpy.primitives.primitives import Cipher
        >>> from arxpy.primitives import speck
        >>> Speck32 = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
        >>> issubclass(Speck32, Cipher)
        True
        >>> plaintext = [0, 0]
        >>> masterkey = [0, 0, 0, 0]
        >>> Speck32(plaintext, masterkey)
        (0x2bb9, 0xc642)

    Attributes:
        key_schedule: the `KeySchedule` function of the cipher
        encryption: the `Encryption` function of the cipher

    """
    key_schedule = None
    encryption = None
    rounds = None

    _minimum_rounds = 1  # for testing

    def __new__(cls, plaintext, masterkey, **options):
        assert isinstance(plaintext, collections.abc.Sequence)
        assert isinstance(masterkey, collections.abc.Sequence)
        assert cls.rounds >= cls._minimum_rounds

        previous_round_keys = cls.encryption.round_keys

        round_keys = cls.key_schedule(*masterkey, **options)
        cls.encryption.round_keys = round_keys
        result = cls.encryption(*plaintext, **options)

        cls.encryption.round_keys = previous_round_keys

        return result

    @classmethod
    def set_rounds(cls, new_rounds):
        """Change the number of rounds and adjust the input/output widths."""
        raise NotImplementedError("subclasses need to override this method")
