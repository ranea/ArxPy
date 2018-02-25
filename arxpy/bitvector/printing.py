"""The Printing module manages the representation of bit-vector types."""
from sympy.printing import repr as sprepr
from sympy.printing import str as spstr


class BvStrPrinter(spstr.StrPrinter):
    """Printing class that handles the ``str`` method of bit-vector."""

    def need_parentheses(self, bv):
        """Return true if bv need parenthesis when used in infix notation."""
        from arxpy.bitvector import core

        assert isinstance(bv, (core.Term, int))

        if isinstance(bv, int):
            return False
        elif isinstance(bv, core.Term) and len(bv.args) in [0, 1]:
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
        op_name = getattr(bv, "short_name", type(bv).__name__)

        args = []
        for a in bv.args:
            if has_symbol and self.need_parentheses(a):
                args.append("({})".format(self._print(a)))
            else:
                args.append(self._print(a))

        if has_symbol:
            assert sum(bv.arity) in [1, 2]
            if sum(bv.arity) == 1:
                return "{}{}".format(bv.unary_symbol, args[0])
            elif sum(bv.arity) == 2:
                return ('{} {} {}'.format(args[0], bv.infix_symbol, args[1]))
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

        if self.need_parentheses(x):
            x = "({})".format(self._print(x))
        else:
            x = self._print(x)

        return "{}[{}{}{}]".format(x, i, delimiter, j)


class BvReprPrinter(sprepr.ReprPrinter):
    """Printing class that handles the verbose representation of bit-vectors."""

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


def dotprinting(bv, verbose=False, vrepr_identifier=False):
    """Print the given bit-vector term to graphviz format.

    If verbose is True, the verbose representation is used to
    label the nodes.

    If vrepr_identifier is True, the verbose representation is used
    to identify the nodes (instead of the hash value).

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

    .. Useful links

        * http://matthiaseisen.com/articles/graphviz/
        * https://graphviz.readthedocs.io/
    """
    try:
        import graphviz as gv
    except ImportError:
        return None

    from arxpy.bitvector import core

    assert isinstance(bv, core.Term)

    dg = gv.Digraph(format='pdf')

    dg.graph_attr['rankdir'] = 'TD'
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
            string = node_bv.vrepr() if verbose else str(node_bv)
        else:
            string = type(node_bv).__name__

        return string

    def name(node_bv):
        """Return the unique identifier for the node."""
        if vrepr_identifier:
            if isinstance(node_bv, core.Term):
                return node_bv.vrepr()
            else:
                return str(node_bv)
        else:
            return str(hash(node_bv))

    def traverse(dg, node_bv):
        if isinstance(node_bv, int) or node_bv.is_Atom:
            dg.node(name(node_bv), label(node_bv), **default_node_options)
        else:
            dg.node(name(node_bv), label(node_bv), **default_node_options)

            for node_arg in node_bv.args:
                traverse(dg, node_arg)

                dg.edge(name(node_bv), name(node_arg), **default_edge_options)

    traverse(dg, bv)

    return dg.source
