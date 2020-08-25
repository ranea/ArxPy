"""Manage the representation of bit-vector expressions."""
from sympy.printing import repr as sympy_repr
from sympy.printing import str as sympy_str


# noinspection PyPep8Naming,PyMethodMayBeStatic
class BvStrPrinter(sympy_str.StrPrinter):
    """Printing class that handles the `str` method of `Term`."""

    def _need_parentheses(self, bv, parent):
        """Return true if bv need parenthesis when used in infix notation."""
        from arxpy.bitvector import core

        assert isinstance(bv, (core.Term, int))

        if isinstance(bv, int):
            return False
        elif isinstance(bv, core.Term) and len(bv.args) in [0, 1]:
            return False
        elif type(bv) == type(parent):
            # avoid a == b == c == d instead of (a == b) == (c == d)
            from arxpy.bitvector import operation
            if type(parent) == operation.BvComp:
                return True
            else:
                return False
        else:
            return True

    def _print_Term(self, bv):
        args = [self._print(a) for a in bv.args]
        return "{}({})".format(type(bv).__name__, ", ".join(args))

    def _print_Constant(self, bv):
        if bv.width % 4 == 0:
            return bv.hex()
        else:
            return bv.bin()

    def _print_Variable(self, bv):
        return bv.name

    def _print_Operation(self, bv):
        has_symbol = hasattr(bv, "unary_symbol") or hasattr(bv, "infix_symbol")
        op_name = getattr(bv, "alt_name", type(bv).__name__)

        args = []
        for a in bv.args:
            if has_symbol and self._need_parentheses(a, bv):
                args.append("({})".format(self._print(a)))
            else:
                args.append(self._print(a))

        if has_symbol:
            assert sum(bv.arity) in [1, 2]
            if sum(bv.arity) == 1:
                return "{}{}".format(bv.unary_symbol, args[0])
            elif sum(bv.arity) == 2:
                return "{} {} {}".format(args[0], bv.infix_symbol, args[1])
        else:
            return "{}({})".format(op_name, ", ".join(args))

    def _print_Extract(self, bv):
        x, i, j = bv.args
        delimiter = ":"

        if i == j:
            delimiter = i = ""
        else:
            if j == 0:
                j = ""

            if i == x.width - 1:
                i = ""

        if self._need_parentheses(x, bv):
            x = "({})".format(self._print(x))
        else:
            x = self._print(x)

        return "{}[{}{}{}]".format(x, i, delimiter, j)


class BvShortPrinter(BvStrPrinter):
    """Printing class that handles the `srepr` method of `Term`."""

    lvl = 0

    max_lvl = 5

    def _print_Term(self, bv):
        assert False
        # args = [self._print(a) for a in bv.args]
        # return "{}({})".format(type(bv).__name__, ", ".join(args))

    def _print_Operation(self, bv):
        has_symbol = hasattr(bv, "unary_symbol") or hasattr(bv, "infix_symbol")
        op_name = getattr(bv, "alt_name", type(bv).__name__)

        from arxpy.bitvector import extraop
        # if all(isinstance(a, (int, core.Constant, core.Variable))
        #        or type(a) == type(bv) for a in bv.args):
        if all(isinstance(a, (extraop.Reverse, extraop.PopCount, extraop.PopCountSum2,
                              extraop.PopCountSum3, extraop.PopCountDiff)) or
               type(a) == type(bv) for a in bv.args):
            next_lvl = False
        else:
            next_lvl = True
        if next_lvl:
            BvShortPrinter.lvl += 1

        args = []
        for a in bv.args:
            if BvShortPrinter.lvl > BvShortPrinter.max_lvl:
                args.append("...")
            elif has_symbol and self._need_parentheses(a, bv):
                args.append("({})".format(self._print(a)))
            else:
                args.append(self._print(a))

        if next_lvl:
            BvShortPrinter.lvl -= 1

        if has_symbol:
            assert sum(bv.arity) in [1, 2]
            if sum(bv.arity) == 1:
                return "{}{}".format(bv.unary_symbol, args[0])
            elif sum(bv.arity) == 2:
                return "{} {} {}".format(args[0], bv.infix_symbol, args[1])
        else:
            return "{}({})".format(op_name, ", ".join(args))


# noinspection PyPep8Naming,PyMethodMayBeStatic
class BvReprPrinter(sympy_repr.ReprPrinter):
    """Printing class that handles the `Term.vrepr` method."""

    def _print_Term(self, bv):
        if bv.args:
            args = ', '.join([self._print(a) for a in bv.args])
            return "{}({}, width={})".format(type(bv).__name__, args, bv.width)
        else:
            return "{}(width={})".format(type(bv).__name__, bv.width)

    def _print_Constant(self, bv):
        return "{}({}, width={})".format(type(bv).__name__, bv.bin(), bv.width)

    def _print_Variable(self, bv):
        name = self._print(bv.name)
        return "{}({}, width={})".format(type(bv).__name__, name, bv.width)


class BvWrapPrinter(BvStrPrinter):
    """Printing class that wrap the representation of `Term`."""

    len_prefix = 0

    max_line_width = 100

    def _print_Operation(self, bv):
        standard_repr = super()._print_Operation(bv)
        if len(standard_repr) + self.__class__.len_prefix < self.__class__.max_line_width:
            return standard_repr

        if hasattr(bv, "unary_symbol"):
            op_name = bv.unary_symbol
        elif hasattr(bv, "infix_symbol"):
            op_name = bv.infix_symbol
        else:
            op_name = getattr(bv, "alt_name", type(bv).__name__)

        old_len_prefix = self.__class__.len_prefix
        self.__class__.len_prefix += len(op_name) + 1

        args = [self._print(bv.args[0])]
        for a in bv.args[1:]:
            args.append("\n" + " "*self.__class__.len_prefix + self._print(a))

        # end = "\n" + " "*old_len_prefix + ")"

        self.__class__.len_prefix = old_len_prefix

        return "{}({}".format(op_name, ",".join(args)) # + end


def dotprinting(bv, vrepr_label=False, vrepr_id=False):
    """Print the given bit-vector expression to graphviz format.

    Args:
        bv: a bit-vector `Term`
        vrepr_label: if True, the verbose representation (`Term.vrepr`) is used
            to label the nodes (instead of the default representation)
        vrepr_id: if True, the verbose representation is used to
            identify the nodes (instead of the hash value)

    ::

        >>> from arxpy.bitvector.core import Constant, Variable
        >>> expr = Constant(1, 8) + ~Variable("x", 8)
        >>> print(dotprinting(expr))  # doctest: +SKIP
        digraph {
            graph [rankdir=TD]
            8318688407297900065 [label=BvAdd color=black shape=ellipse]
            2830213174350589301 [label="0x01" color=black shape=ellipse]
            8318688407297900065 -> 2830213174350589301
            6499762230957827102 [label=BvNot color=black shape=ellipse]
            8990231514331719946 [label=x color=black shape=ellipse]
            6499762230957827102 -> 8990231514331719946
            8318688407297900065 -> 6499762230957827102
        }

    Note:
         This method requires `graphviz <https://www.graphviz.org/>`_
         and its `python interface <https://pypi.org/project/graphviz/>`_
         to be installed.

    .. Useful links

        * http://matthiaseisen.com/articles/graphviz/
        * https://graphviz.readthedocs.io/
    """
    try:
        import graphviz as gv
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("dotprinting requires graphviz and its python interface; {}", format(e))

    from arxpy.bitvector import core

    assert isinstance(bv, core.Term)

    digraph = gv.Digraph(format='pdf')

    digraph.graph_attr['rankdir'] = 'TD'
    default_node_options = {
        "shape": 'ellipse',
        "color": "black",
    }
    default_edge_options = {}

    def label(node_bv):
        """Return the caption to be displayed."""
        assert isinstance(node_bv, (int, core.Term))

        if isinstance(node_bv, int):
            return str(node_bv)
        elif node_bv.is_Atom:
            string = node_bv.vrepr() if vrepr_label else str(node_bv)
        else:
            string = type(node_bv).__name__

        return string

    def name(node_bv):
        """Return the unique identifier for the node."""
        if vrepr_id:
            if isinstance(node_bv, core.Term):
                return node_bv.vrepr()
            else:
                return str(node_bv)
        else:
            return str(hash(node_bv))

    def traverse(dg, node_bv):
        if isinstance(node_bv, int) or node_bv.is_Atom:
            # noinspection PyTypeChecker
            dg.node(name(node_bv), label(node_bv), **default_node_options)
        else:
            dg.node(name(node_bv), label(node_bv), **default_node_options)

            for node_arg in node_bv.args:
                traverse(dg, node_arg)

                dg.edge(name(node_bv), name(node_arg), **default_edge_options)

    traverse(digraph, bv)

    return digraph.source
