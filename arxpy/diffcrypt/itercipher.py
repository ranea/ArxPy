"""The IterCipher module finds optimal characteristics of iterated ciphers ."""
import contextlib
import sys
import time

from arxpy.bitvector import function

from arxpy.diffcrypt import characteristic
from arxpy.diffcrypt import difference
from arxpy.diffcrypt import differential
from arxpy.diffcrypt import smt


class IterFunction(function.Function):
    """Base class to define iterated bit-vector functions.

    An iterated function is a function based on the repetition
    of a smaller function a number of times (rounds).

    To define a new iterated function, subclass IterFunction, override
    it as a bit-vector function (see Function) and finally override the
    rounds class attribute and the set_rounds class method.

        >>> from arxpy.bitvector.operation import Concat
        >>> from arxpy.diffcrypt.itercipher import IterFunction
        >>> class MyIterFunction(IterFunction):
        ...     input_widths = [8] + [8 for i in range(2)]
        ...     output_widths = [8]
        ...     @classmethod
        ...     def eval(cls, x, *k):
        ...         for k_i in k:
        ...             x ^= k_i
        ...         return x
        ...     rounds = 2
        ...     @classmethod
        ...     def set_rounds(cls, new_rounds):
        ...         cls.rounds = new_rounds
        ...         cls.input_widths = [8] + [8 for i in range(cls.new_rounds)]
        >>> MyIterFunction(0, 0, 1)
        0x01


    As shown in the example, variable-positional arguments (e.g. *k)
    are useful to handle round-dependant number of inputs.
    """

    rounds = None

    @classmethod
    def set_rounds(cls, new_rounds):
        """Change the number of rounds and adjust the input/output widths."""
        return None


class IterBlockCipher(function.CompositeFunction):
    """Base class to define iterated block ciphers.

    An IterBlockCipher is a composite bit-vector function where
    the inner function (key schedule) and the outer function (encryption)
    are iterated bit-vector function (see IterFunction).

    To define a new iterated block cipher, subclass IterBlockCipher,
    override it as a composite bit-vector function (see CompositeFunction)
    and finally override the rounds class attribute and the set_rounds
    class method.

        >>> from arxpy.diffcrypt.itercipher import IterFunction, IterBlockCipher
        >>> class MyKeySchedule(IterFunction):
        ...     input_widths = [8]
        ...     output_widths = [8 for i in range(2)]
        ...     @classmethod
        ...     def eval(cls, k):
        ...         return [k + i for i in range(cls.rounds)]
        ...     rounds = 2
        ...     @classmethod
        ...     def set_rounds(cls, new_rounds):
        ...         cls.rounds = new_rounds
        ...         cls.output_widths = [8 for i in range(cls.new_rounds)]
        >>> class MyEncryption(IterFunction):
        ...     input_widths = [8] + [8 for i in range(2)]
        ...     output_widths = [8]
        ...     @classmethod
        ...     def eval(cls, x, *k):
        ...         for k_i in k:
        ...             x ^= k_i
        ...         return x
        ...     rounds = 2
        ...     @classmethod
        ...     def set_rounds(cls, new_rounds):
        ...         cls.rounds = new_rounds
        ...         cls.input_widths = [8] + [8 for i in range(cls.new_rounds)]
        >>> class MyCipher(IterBlockCipher):
        ...     input_widths = [8, 8]
        ...     output_widths = [8]
        ...     inner_func = MyKeySchedule
        ...     outer_func = MyEncryption
        ...     rounds = 2
        ...     @classmethod
        ...     def set_rounds(cls, new_rounds):
        ...         cls.rounds = new_rounds
        ...         cls.inner_func.set_rounds(new_rounds)
        ...         cls.outer_func.set_rounds(new_rounds)
        >>> MyCipher(0, 0)
        0x01

    """

    rounds = None

    @classmethod
    def set_rounds(cls, new_rounds):
        """Set the cipher rounds and the key schedule/encryption rounds."""
        return None


class OptimalRelatedKeyCh(object):
    """Find related-key optimal characteristics of reduced-round block ciphers.

    Arguments:

    - cipher: an iterated block cipher.
    - diff_type: the type of difference of the characteristic.
    - filename (optional): the name of the file where the results will be
      written. If it is not given, the standard output is used instead.
    - start (optional): the initial number of rounds to consider.
      By default is 1.
    - end (optional): the maximum number of rounds to consider.
      By default is the rounds of the cipher minus 1.

    This function finds optimal characteristics in an incremental way,
    that is, first it finds the optimal characteristic considering
    *start* rounds, then it considers *start + 1* rounds,
    *start + 2* rounds, ..., until *end* rounds.

    Note that in each iteration, two optimal characteristics are searched;
    the characteristic minimizing the encryption weight and the
    characteristic minimizimg the sum of the key schedule weight
    and encryption weight.

    The results are written in markdown format.

    In the key schedule characteristic, the input differences are
    denoted as "mk" and the intermediate ones as "l".
    In the encryption characteristic, the input differences are
    denoted as "p" and the intermediate ones as "x".
    """

    def __init__(self, cipher, diff_type, filename=None, start=1, end=None):
        assert issubclass(cipher, IterBlockCipher)
        assert issubclass(diff_type, difference.Difference)
        self.cipher = cipher
        self.diff_type = diff_type
        self.filename = filename
        self.start = start
        if end is None:
            end = cipher.rounds - 1
        self.end = end

        self._search()

    @contextlib.contextmanager
    def _open(self):
        """Return a file or the standard output depending on filename."""
        if self.filename and self.filename != '-':
            fh = open(self.filename, 'a')
        else:
            fh = sys.stdout

        try:
            yield fh
        finally:
            if fh is not sys.stdout:
                fh.close()

    def smart_print(self, msg):
        with self._open() as fh:
            print(msg, file=fh, flush=True)

    def min_single_weight(self, current_weight, single_ch, parent_ch):
        """Obtain the minimal weight of a single characteristic.

        For example, if the single characteristic is the encryption one,
        it seacrhs for the optimal encryption characteristic where
        the key schedule is omitted (round keys differences are
        chosen freely).
        """
        if not any(isinstance(p, differential.Differential) for p in single_ch.values()):
            msg = "The {} characteristic is trivial".format(single_ch.func.__name__)
            self.smart_print(msg)
            return 0

        target_weight = current_weight

        if single_ch == parent_ch.inner_ch:
            max_weight = sum(single_ch.func.input_widths)
        else:
            max_inner_weight = sum(parent_ch.func.inner_func.input_widths)
            max_weight = sum(parent_ch.func.input_widths) - max_inner_weight

        # First iteration
        #  - Check that the minimum weight of the previous iteration
        #    is no longer valid (no satisfiable).

        start = time.time()

        while target_weight > 0:
            smt_problem = smt.SmtProblem(single_ch, target_weight,
                                         equality=False, parent_ch=parent_ch)

            if not smt_problem.solve():
                break
            else:
                target_weight //= 2

        msg = (
            "\nNo {} characteristic exists with weight "
            "lower than {} ({:.2f}s spent)."
        ).format(single_ch.func.__name__, target_weight, time.time() - start)
        self.smart_print(msg)

        while target_weight < max_weight:
            smt_problem = smt.SmtProblem(single_ch, target_weight, parent_ch=parent_ch)

            problem_size = smt_problem.formula_size, smt_problem.pysmt_formula_size

            msg = "{} characteristic with weight equals to {} ".format(
                single_ch.func.__name__, target_weight
            )

            start = time.time()

            if not smt_problem.solve():
                msg = "\nThere doesn't exist {} ({:.2f}s spent, problem size {}).".format(
                    msg, time.time() - start, problem_size
                )
                self.smart_print(msg)

                target_weight += single_ch.get_inverse_weight_function()(1)
                continue
            else:
                msg = "\nThere exist {} ({:.2f}s spent, problem size {}).".format(
                    msg, time.time() - start, problem_size
                )
                self.smart_print(msg)

            start = time.time()

            assig = smt_problem.solve(get_assignment=True)

            msg = "\n* Assignment found in {:.2f}s. \n".format(time.time() - start)
            if assig["weight"] < 32:
                empirical_weight = single_ch.empirical_weight(
                    list(assig["differences"].values()), False, assig["weight"])
                msg += "* Empirical weight {:.2f}. \n".format(empirical_weight)
            msg += "\n```\n{}\n```".format(assig)

            self.smart_print(msg)

            break

        return target_weight

    def is_satisfiable(self, target_weight, ch, inner_equality=False):
        """Solve the SMT problem related to the related-key characteristic."""
        smt_problem = smt.CompositeSmtProblem(ch, target_weight,
                                              inner_equality=inner_equality)

        problem_size = smt_problem.formula_size, smt_problem.pysmt_formula_size

        comparison = "equals to" if inner_equality else "less than"
        msg = (
            "characteristic with key schedule weight {} {}"
            " and encryption weight equal to {}").format(comparison, *target_weight)

        start = time.time()

        if not smt_problem.solve():
            msg = "\nThere doesn't exist {} ({:.2f}s spent, problem size {}).".format(
                msg, time.time() - start, problem_size
            )
            self.smart_print(msg)
            return False
        else:
            msg = "\nThere exist {} ({:.2f}s spent, problem size {}).".format(
                msg, time.time() - start, problem_size
            )
            self.smart_print(msg)

        start = time.time()

        inner_assig, outer_assig = smt_problem.solve(get_assignment=True)

        msg = "\n* Assignment found in {:.2f}s. \n".format(time.time() - start)
        if inner_assig["weight"] < 32:
            inner_ew = ch.inner_ch.empirical_weight(
                list(inner_assig["differences"].values()), False,
                inner_assig["weight"])
            msg += "* Key schedule empirical weight {:.2f}. \n".format(inner_ew)
        if outer_assig["weight"] < 32:
            outer_ew = ch.outer_ch.empirical_weight(
                list(outer_assig["differences"].values()), False,
                outer_assig["weight"])
            msg += "* Encryption empirical weight {:.2f}. \n".format(outer_ew)
        msg += "\n```\n{}\n{}\n```".format(inner_assig, outer_assig)

        self.smart_print(msg)

        return True

    def _search(self):
        diff_name = getattr(self.diff_type, "short_name", self.diff_type.__name__)
        msg = "# Optimal {} related-key characteristics of {}"
        self.smart_print(msg.format(diff_name, self.cipher.__name__))

        next_outer_weight = None
        min_inner_weight = 0

        for rounds in range(self.start, self.end + 1):
            self.cipher.set_rounds(rounds)

            msg = "\n## {}-rounds characteristics".format(rounds)
            self.smart_print(msg)

            input_diff = self.cipher._symbolic_input("p", "mk")
            input_diff = [difference.DiffVar.from_Variable(d) for d in input_diff]

            ch = characteristic.CompositeCh(self.cipher, self.diff_type,
                                            input_diff, "l", "x")

            self.smart_print("\n### Optimal characteristic w.r.t encryption weight")

            if next_outer_weight is None:
                self.smart_print("\n#### Lower bound encryption weight")
                outer_weight = self.min_single_weight(0, ch.outer_ch, ch)
            else:
                outer_weight = next_outer_weight

            self.smart_print("\n#### Minimum encryption weight")

            max_inner_weight = sum(ch.func.inner_func.input_widths)
            max_outer_weight = sum(ch.func.input_widths) - max_inner_weight
            inner_inv = ch.inner_ch.get_inverse_weight_function()
            outer_inv = ch.outer_ch.get_inverse_weight_function()

            while outer_weight < max_outer_weight:
                target_weight = [max_inner_weight - inner_inv(1), outer_weight]

                if self.is_satisfiable(target_weight, ch):
                    break
                else:
                    outer_weight += outer_inv(1)
            else:
                msg = (
                    "\nThere doesn't exist {} related-key characteristic of "
                    "{}-round {} with encryption weight less than {}."
                ).format(diff_name, rounds, self.cipher.__name__, max_outer_weight)

                self.smart_print(msg)

                return

            self.smart_print("\n#### Lower bound key schedule weight")

            min_inner_weight = self.min_single_weight(min_inner_weight, ch.inner_ch, ch)
            inner_weight = min_inner_weight

            is_trivial = not any(isinstance(p, differential.Differential)
                                 for p in ch.inner_ch.values())

            self.smart_print("\n#### Minimum key schedule weight")

            while inner_weight < max_inner_weight and not is_trivial:
                target_weight = [inner_weight, outer_weight]

                if self.is_satisfiable(target_weight, ch, inner_equality=True):
                    break
                else:
                    inner_weight += inner_inv(1)

            msg = (
                "\n**The optimal {} related-key characteristic of "
                "{}-round {} (w.r.t the encryption weight) has "
                " {} key schedule weight and {} encryption weight.**"
            ).format(diff_name, rounds, self.cipher.__name__, inner_weight, outer_weight)
            self.smart_print(msg)

            next_outer_weight = outer_weight

            msg = (
                "\n### Optimal characteristic w.r.t the sum of the "
                "key schedule weight and the encryption weight"
            )
            self.smart_print(msg)

            offset = 1
            original_sum = inner_weight + outer_weight
            new_inner_weight = inner_weight - (offset + 1)
            new_outer_weight = outer_weight + offset

            while new_inner_weight >= 0 and new_outer_weight < max_outer_weight:
                target_weight = [new_inner_weight, new_outer_weight]

                if self.is_satisfiable(target_weight, ch):
                    temp_max_inner_weight = new_inner_weight
                    new_inner_weight = min_inner_weight

                    while new_inner_weight < temp_max_inner_weight:
                        target_weight = [new_inner_weight, new_outer_weight]

                        if self.is_satisfiable(target_weight, ch, inner_equality=True):
                            inner_weight = new_inner_weight
                            outer_weight = new_outer_weight
                            offset = 0
                            break
                        else:
                            new_inner_weight += 1

                offset += 1
                new_inner_weight = inner_weight - (offset + 1)
                new_outer_weight = outer_weight + offset

            if inner_weight + outer_weight < original_sum:
                msg = (
                    "\n**The optimal {} related-key characteristic of "
                    "{}-round {} (w.r.t the sum of the key schedule weight "
                    " and the encryption weight) has {} key schedule weight"
                    " and {} encryption weight.**"
                ).format(diff_name, rounds, self.cipher.__name__,
                         inner_weight, outer_weight)
                self.smart_print(msg)
            else:
                msg = (
                    "\n**The optimal characteristic w.r.t the encryption weight "
                    "is also the optimal characteristic w.r.t. the sum of the "
                    "key schedule weight and the encryption weight.**"
                )
                self.smart_print(msg)
