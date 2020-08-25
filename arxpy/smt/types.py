"""Convert between pySMT_ types and `bitvector` types.

.. _pySMT: https://github.com/pysmt/pysmt
"""

from pysmt import environment

from arxpy.bitvector import core
from arxpy.bitvector import operation
from arxpy.bitvector import extraop

from arxpy.differential import difference


def _is_power_of_2(x):
    # http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
    return (x & (x - 1)) == 0


def bv2pysmt(bv, boolean=False, strict_shift=False, env=None):
    """Convert a bit-vector type to a pySMT type.

    Args:
        bv: the bit-vector `Term` to convert
        boolean: if True, boolean pySMT types (e.g., `pysmt.shortcuts.Bool`) are used instead of
            bit-vector pySMT types (e.g., `pysmt.shortcuts.BV`).
        strict_shift: if `True`, shifts and rotation by non-power-of-two offsets
            are power of two are translated to pySMT's shifts and
            rotation directly.
        env: a `pysmt.environment.Environment`; if not specified, a new pySMT environment is created.
    ::

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> from arxpy.smt.types import bv2pysmt
        >>> s = bv2pysmt(Constant(0b00000001, 8), boolean=False)
        >>> s, s.get_type()
        (1_8, BV{8})
        >>> x, y = Variable("x", 8), Variable("y", 8)
        >>> s = bv2pysmt(x)
        >>> s, s.get_type()
        (x, BV{8})
        >>> s = bv2pysmt(x +  y)
        >>> s, s.get_type()
        ((x + y), BV{8})
        >>> s = bv2pysmt(x <=  y)
        >>> s, s.get_type()
        ((x u<= y), Bool)
        >>> s = bv2pysmt(x[4: 2])
        >>> s, s.get_type()
        (x[2:4], BV{3})

    """
    msg = "unknown conversion of {} to a pySMT type".format(type(bv).__name__)

    if env is None:
        env = environment.reset_env()
    fm = env.formula_manager

    if isinstance(bv, int):
        return bv

    pysmt_bv = None

    if isinstance(bv, core.Variable):
        if boolean:
            assert bv.width == 1
            pysmt_bv = fm.Symbol(bv.name, env.type_manager.BOOL())
        else:
            pysmt_bv = fm.Symbol(bv.name, env.type_manager.BVType(bv.width))

    elif isinstance(bv, core.Constant):
        if boolean:
            assert bv.width == 1
            pysmt_bv = fm.Bool(bool(bv))
        else:
            pysmt_bv = fm.BV(bv.val, bv.width)

    elif isinstance(bv, operation.Operation):
        # only 1st layer can return a boolean
        # Equals and Ite work well with BV, the rest don't

        if issubclass(type(bv), extraop.PartialOperation):
            raise NotImplementedError("PartialOperation is not yet supported")

        if type(bv) == operation.BvNot:
            if boolean:
                assert bv.width == 1
                args = [bv2pysmt(a, True, strict_shift, env) for a in bv.args]
                pysmt_bv = fm.Not(*args)
            else:
                args = [bv2pysmt(a, False, strict_shift, env) for a in bv.args]
                pysmt_bv = fm.BVNot(*args)

        elif type(bv) == operation.BvAnd:
            if boolean:
                assert bv.width == 1
                args = [bv2pysmt(a, True, strict_shift, env) for a in bv.args]
                pysmt_bv = fm.And(*args)
            else:
                args = [bv2pysmt(a, False, strict_shift, env) for a in bv.args]
                pysmt_bv = fm.BVAnd(*args)

        elif type(bv) == operation.BvOr:
            if boolean:
                assert bv.width == 1
                args = [bv2pysmt(a, True, strict_shift, env) for a in bv.args]
                pysmt_bv = fm.Or(*args)
            else:
                args = [bv2pysmt(a, False, strict_shift, env) for a in bv.args]
                pysmt_bv = fm.BVOr(*args)
        elif type(bv) == operation.BvXor:
            if boolean:
                assert bv.width == 1
                args = [bv2pysmt(a, True, strict_shift, env) for a in bv.args]
                pysmt_bv = fm.Xor(*args)
            else:
                args = [bv2pysmt(a, False, strict_shift, env) for a in bv.args]
                pysmt_bv = fm.BVXor(*args)
        elif type(bv) == operation.Ite:
            args = [None for _ in range(len(bv.args))]
            # fm.Ite requires a Boolean type for args[0] but
            # bv2pysmt(bv.args[0], True, ...)  caused an error
            # (if args[0] is BvComp, it can be further optimized)
            args[0] = bv2pysmt(bv.args[0], False, strict_shift, env)
            if args[0].get_type().is_bv_type():
                args[0] = fm.Equals(args[0], fm.BV(1, 1))
            if boolean:
                assert bv.width == 1
                args[1:] = [bv2pysmt(a, True, strict_shift, env) for a in bv.args[1:]]
            else:
                args[1:] = [bv2pysmt(a, False, strict_shift, env) for a in bv.args[1:]]
            pysmt_bv = fm.Ite(*args)
        else:
            args = [bv2pysmt(a, False, strict_shift, env) for a in bv.args]

            if type(bv) == operation.BvComp:
                if boolean:
                    pysmt_bv = fm.Equals(*args)
                else:
                    pysmt_bv = fm.BVComp(*args)

            elif type(bv) == operation.BvUlt:
                pysmt_bv = fm.BVULT(*args)

            elif type(bv) == operation.BvUle:
                pysmt_bv = fm.BVULE(*args)

            elif type(bv) == operation.BvUgt:
                pysmt_bv = fm.BVUGT(*args)

            elif type(bv) == operation.BvUge:
                pysmt_bv = fm.BVUGE(*args)

            elif boolean:
                raise ValueError("{} cannot return a boolean type".format(type(bv).__name__))

            elif type(bv) in [operation.BvShl, operation.BvLshr]:
                if not strict_shift or _is_power_of_2(args[0].bv_width()):
                    if type(bv) == operation.BvShl:
                        pysmt_bv = fm.BVLShl(*args)
                    elif type(bv) == operation.BvLshr:
                        pysmt_bv = fm.BVLShr(*args)
                else:
                    x, r = bv.args
                    offset = 0
                    while not _is_power_of_2(x.width):
                        x = operation.ZeroExtend(x, 1)
                        r = operation.ZeroExtend(r, 1)
                        offset += 1

                    shift = bv2pysmt(type(bv)(x, r), False, strict_shift, env)
                    pysmt_bv = fm.BVExtract(shift, end=shift.bv_width() - offset - 1)

            elif type(bv) == operation.RotateLeft:
                if not strict_shift or _is_power_of_2(args[0].bv_width()):
                    pysmt_bv = fm.BVRol(*args)
                else:
                    # Left hand side width must be a power of 2
                    x, r = bv.args
                    n = x.width
                    pysmt_bv = bv2pysmt(operation.Concat(x[n - r - 1:], x[n - 1: n - r]),
                                    False, strict_shift, env)

            elif type(bv) == operation.RotateRight:
                if not strict_shift or _is_power_of_2(args[0].bv_width()):
                    pysmt_bv = fm.BVRor(*args)
                else:
                    # Left hand side width must be a power of 2
                    x, r = bv.args
                    n = x.width
                    pysmt_bv = bv2pysmt(operation.Concat(x[r - 1:], x[n - 1: r]),
                                    False, strict_shift, env)

            elif type(bv) == operation.Extract:
                # pySMT Extract(bv, start, end)
                pysmt_bv = fm.BVExtract(args[0], args[2], args[1])

            elif type(bv) == operation.Concat:
                pysmt_bv = fm.BVConcat(*args)

            elif type(bv) == operation.ZeroExtend:
                pysmt_bv = fm.BVZExt(*args)

            elif type(bv) == operation.Repeat:
                pysmt_bv = args[0].BVRepeat(args[1])

            elif type(bv) == operation.BvNeg:
                pysmt_bv = fm.BVNeg(*args)

            elif type(bv) == operation.BvAdd:
                pysmt_bv = fm.BVAdd(*args)

            elif type(bv) == operation.BvSub:
                pysmt_bv = fm.BVSub(*args)

            elif type(bv) == operation.BvMul:
                pysmt_bv = fm.BVMul(*args)

            elif type(bv) == operation.BvUdiv:
                pysmt_bv = fm.BVUDiv(*args)

            elif type(bv) == operation.BvUrem:
                pysmt_bv = fm.BVURem(*args)

            else:
                bv2 = bv.doit()
                assert bv.width == bv2.width, "{} == {}\n{}\n{}".format(bv.width, bv2.width, bv.vrepr(), bv2.vrepr())
                if bv != bv2:  # avoid cyclic loop
                    pysmt_bv = bv2pysmt(bv2, boolean=boolean, strict_shift=strict_shift, env=env)
                else:
                    raise NotImplementedError("(doit) " + msg)

    elif isinstance(bv, difference.Difference):
        pysmt_bv = bv2pysmt(bv.val, boolean, strict_shift, env)

    if pysmt_bv is not None:
        try:
            pysmt_bv_width = pysmt_bv.bv_width()
        except (AssertionError, TypeError):
            pysmt_bv_width = 1  # boolean type

        assert bv.width == pysmt_bv_width
        return pysmt_bv
    else:
        raise NotImplementedError(msg)


def pysmt2bv(ps):
    """Convert a pySMT type to a bit-vector type.

    Currently, only conversion from `pysmt.shortcuts.BV`
    and `pysmt.shortcuts.Symbol` is supported.

        >>> from pysmt import shortcuts, typing  # pySMT shortcuts and typing modules
        >>> from arxpy.smt.types import pysmt2bv
        >>> env = shortcuts.reset_env()
        >>> pysmt2bv(env.formula_manager.Symbol("x", env.type_manager.BVType(8))).vrepr()
        "Variable('x', width=8)"
        >>> pysmt2bv(env.formula_manager.BV(1, 8)).vrepr()
        'Constant(0b00000001, width=8)'

    """
    class_name = type(ps).__name__
    msg = "unknown conversion of {} ({} {}) to a bit-vector type".format(ps, ps.get_type(), class_name)

    if ps.is_symbol():
        if str(ps.get_type()) == "Bool":
            return core.Variable(ps.symbol_name(), 1)
        else:
            return core.Variable(ps.symbol_name(), ps.bv_width())
    elif ps.is_bv_constant():
        return core.Constant(int(ps.constant_value()), ps.bv_width())
    elif ps.is_false():
        return core.Constant(0, 1)
    elif ps.is_true():
        return core.Constant(1, 1)
    else:
        raise NotImplementedError(msg)


def pysmt_model2bv_model(model, differences=None):
    """Convert a `pysmt.solvers.solver.Model` into a `dict` of bit-vector types.

    To return `Difference` values instead of `Term` values,
    a list of symbolic differences can be passed as argument.
    In that case, variables in the model also present in ``differences``
    will be added to the bit-vector model as `Difference` objects.

        >>> from pysmt import shortcuts
        >>> from arxpy.smt.types import pysmt_model2bv_model  # pySMT shortcuts
        >>> env = shortcuts.reset_env()
        >>> fm = env.formula_manager
        >>> formula = fm.Equals(fm.BV(0, 8), fm.Symbol("x", env.type_manager.BVType(8)))
        >>> pysmt_model = env.factory.get_model(formula)
        >>> for var, val in pysmt_model: print(var, val, var.get_type(), val.get_type())
        x 0_8 BV{8} BV{8}
        >>> bv_model = pysmt_model2bv_model(pysmt_model)
        >>> for var, val in bv_model.items(): print(var, val, type(var), type(val))
        x 0x00 <class 'arxpy.bitvector.core.Variable'> <class 'arxpy.bitvector.core.Constant'>

    """
    if differences is not None:
        # diff_dict is a dictionary of name -> diff
        name2diff = {}
        for diff in differences:
            name2diff[diff.val.name] = diff
    else:
        name2diff = {}

    bv_model = {}
    for var, value in model:
        bv_var = pysmt2bv(var)
        bv_value = pysmt2bv(value)
        if isinstance(bv_var, core.Variable):
            if bv_var.name in name2diff:
                bv_var = name2diff[bv_var.name]
                bv_value = type(bv_var)(bv_value)
        bv_model[bv_var] = bv_value
    return bv_model
