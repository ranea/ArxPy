=========
ArxPy 0.2
=========

ArxPy is a python3 library to find XOR differential characteristics and
rotational-XOR characteristics in ARX primitives (e.g., block ciphers)
using SMT solvers.

ArxPy is based on the following papers:

- Towards Finding Optimal Differential Characteristics for ARX: Application to Salsa20
- Observations on the SIMON Block Cipher Family
- Rotational-XOR Cryptanalysis of Reduced-round SPECK
- A Bit-Vector Differential Model for the Modular Addition by a Constant

ArxPy provides a complete documentation in `https://ranea.github.io/ArxPy/ <https://ranea.github.io/ArxPy/>`_.
and an extensive suite of tests.


Usage
=====

First, the ARX cipher needs to be implemented following ArxPy interface.
There are several ARX ciphers already implemented (see the folder ``arxpy/primitives``).
To implemente a new cipher, the easiest way is to take a similar cipher
already implemented as a template and modify the python code directly.

Searching the optimal XOR differential characteristic can be done easily with
the function ``round_based_search_SkCh()``. For example, searching for the best
2-round and 3-round differential characteristics of ``Speck_32_64`` can be done as follows

.. code:: python

    >>> from arxpy.differential.difference import XorDiff
    >>> from arxpy.smt.search import SkChSearchMode, DerMode, round_based_search_SkCh
    >>> from arxpy.primitives import speck
    >>> cipher = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
    >>> start_rounds, end_rounds = 4, 5
    >>> round_based_search_SkCh(cipher, XorDiff, 0, "btor", start_rounds, end_rounds,
    ...                         DerMode.Default, SkChSearchMode.Optimal, True, 0, None)
    Num rounds: 4
    Best characteristic found:
    {'ch_weight': 5,
     'der_weights': [[w0w1w2, 3], [w3, 2]],
     'emp_weight': Counter({5: 163, 4: 74, 6: 17, 7: 2}),
     'exact_weight': 5,
     'input_diff': [[dp0, 0x2800], [dp1, 0x0010]],
     'nonlinear_diffs': [[dx1, 0x0040], [dx6, 0x8000], [dx11, 0x8100], [dx16, 0x8000]],
     'output_diff': [[dx17, 0x8000], [dx19, 0x840a]]}

    Num rounds: 5
    Best characteristic found:
    {'ch_weight': 9,
     'der_weights': [[w0w1w2, 6], [w3w4, 3]],
     'emp_weight': Counter({9: 154, 8: 81, 10: 19, 11: 2}),
     'exact_weight': 9,
     'input_diff': [[dp0, 0x0211], [dp1, 0x0a04]],
     'nonlinear_diffs': [[dx1, 0x2800], [dx6, 0x0040], [dx11, 0x8000], [dx16, 0x8100], [dx21, 0x8000]],
     'output_diff': [[dx22, 0x8000], [dx24, 0x840a]]}


Apart from the cipher and the number of rounds to search, the function
``round_based_search_SkCh()`` has several parameters to adapt the search as needed.
This function stores each characteristic found as a ``ChFound`` object.
The explanation of these paremeters and the attributes of ``ChFound`` objects
can be found in the documentation.
Every function and class in ArxPy is documented with a description of
the object and with doctests!

If the argument ``check`` is ``True`` in ``round_based_search_SkCh()``, the
weight of each characteristic found is checked by computing the empirical
weight (by sampling many plaintexts and keys and computing the
differential probability manually). This is useful since for
ARX ciphers the theoretical weight and the actual weight might differ
in some cases.

Searching the optimal related-key XOR differential characteristic can be done
in a similar way with the function ``round_based_search_RkCh()``. For example,
searching for the best 4-round and 5-round related-key characteristics of
``XTEA`` can be done as follows (this can take some minutes)

.. code:: python

    >>> from arxpy.differential.difference import XorDiff
    >>> from arxpy.smt.search import RkChSearchMode, DerMode, round_based_search_RkCh
    >>> from arxpy.primitives.xtea import XteaCipher
    >>> cipher = XteaCipher
    >>> round_based_search_RkCh(cipher, XorDiff, 0, 0, "btor", 4, 5, DerMode.Default,
    ...                         DerMode.Default, True, RkChSearchMode.OptimalMinSum, True, 0, None)
    Num rounds: 4
    Best related-key characteristic found:
    {'enc_ch_found': {'ch_weight': 0,
                      'der_weights': [[w0w1w2, 0], [w3w4w5, 0], [w6w7, 0]],
                      'emp_weight': Counter({0: 256}),
                      'exact_weight': 0,
                      'input_diff': [[dp0, 0x80000000], [dp1, 0x00000000]],
                      'nonlinear_diffs': [[dx3, 0x00000000], [dx5, 0x00000000], [dx9, 0x00000000], [dx11, 0x00000000], [dx15, 0x00000000], [dx17, 0x00000000], [dx21, 0x00000000], [dx23, 0x80000000]],
                      'output_diff': [[dx17, 0x00000000], [dx23, 0x80000000]]},
     'key_ch_found': {'ch_weight': 0,
                      'der_weights': [[wk0, 0.0], [wk1, 0.0], [wk2, 0.0]],
                      'emp_weight': 0.0,
                      'exact_weight': 0,
                      'input_diff': [[dmk0, 0x80000000], [dmk1, 0x00000000], [dmk2, 0x80000000], [dmk3, 0x00000000]],
                      'nonlinear_diffs': [[dk0, 0x00000000], [dk1, 0x00000000], [dk2, 0x80000000]],
                      'output_diff': [[dmk0, 0x80000000], [dk0, 0x00000000], [dk1, 0x00000000], [dk2, 0x80000000]]}}

    Num rounds: 5
    Best related-key characteristic found:
    {'enc_ch_found': {'ch_weight': 0,
                      'der_weights': [[w0w1w2, 0], [w3w4w5, 0], [w6w7w8, 0], [w9, 0]],
                      'emp_weight': Counter({0: 256}),
                      'exact_weight': 0,
                      'input_diff': [[dp0, 0x80000000], [dp1, 0x00000000]],
                      'nonlinear_diffs': [[dx3, 0x00000000], [dx5, 0x00000000], [dx9, 0x00000000], [dx11, 0x00000000], [dx15, 0x00000000], [dx17, 0x00000000], [dx21, 0x00000000], [dx23, 0x00000000], [dx27, 0x00000000], [dx29, 0x00000000]],
                      'output_diff': [[dx23, 0x00000000], [dx29, 0x00000000]]},
     'key_ch_found': {'ch_weight': 0,
                      'der_weights': [[wk0, 0.0], [wk1, 0.0], [wk2, 0.0]],
                      'emp_weight': 0.0,
                      'exact_weight': 0,
                      'input_diff': [[dmk0, 0x80000000], [dmk1, 0x00000000], [dmk2, 0x00000000], [dmk3, 0x00000000]],
                      'nonlinear_diffs': [[dk0, 0x00000000], [dk1, 0x00000000], [dk2, 0x00000000]],
                      'output_diff': [[dmk0, 0x80000000], [dk0, 0x00000000], [dk1, 0x00000000], [dk2, 0x00000000], [dk2, 0x00000000]]}}

A related-key characteristic is split into two characteristics, one
describing the propagation of differences through the key schedule
(``key_ch_found``) and another one describing the propagation of
differences through the encryption function (``enc_ch_found``).

To search for rotational-XOR characteristics, the same functions
``round_based_search_SkCh()`` and ``round_based_search_RkCh()`` can be used
but using ``RXDiff`` as the ``difference_type`` argument. For example,
searching for 4-round and 5-round rotational-XOR differential characteristics
of ``Cham_64_128`` can be done in the following way

.. code:: python

    >>> from arxpy.differential.difference import RXDiff
    >>> from arxpy.smt.search import RkChSearchMode, DerMode, round_based_search_RkCh
    >>> from arxpy.primitives import cham
    >>> cipher = cham.get_Cham_instance(cham.ChamInstance.cham_64_128)
    >>> round_based_search_RkCh(cipher, RXDiff, 0, 0, "btor", 4, 5, DerMode.Default,
    ...                         DerMode.Default, True, RkChSearchMode.FirstMinSumValid, True, 0, None)
    Num rounds: 4
    Best related-key characteristic found:
    {'enc_ch_found': {'ch_weight': 5,
                      'der_weights': [[w0, 1.375], [w1, 1.375], [w2, 1.375], [w3, 1.375]],
                      'emp_weight': Counter({6: 256}),
                      'exact_weight': 5.659973889568066,
                      'input_diff': [[dp0, 0x8000], [dp1, 0x0002], [dp2, 0x8006], [dp3, 0x0004]],
                      'nonlinear_diffs': [[dx2, 0x0001], [dx7, 0x8000], [dx12, 0x8000], [dx17, 0x8000]],
                      'output_diff': [[dx3, 0x0100], [dx8, 0x0001], [dx13, 0x0080], [dx18, 0x0001]]},
     'key_ch_found': {'ch_weight': 0,
                      'der_weights': [],
                      'emp_weight': 0.0,
                      'exact_weight': 0,
                      'input_diff': [[dmk0, 0x6123], [dmk1, 0x0281], [dmk2, 0xc246], [dmk3, 0x6020], [dmk4, 0x8000], [dmk5, 0x8000], [dmk6, 0x8000], [dmk7, 0x8000]],
                      'nonlinear_diffs': [],
                      'output_diff': [[dk3, 0x8004], [dk7, 0x8681], [dk11, 0x0009], [dk15, 0x8000]]}}

    Num rounds: 5
    Best related-key characteristic found:
    {'enc_ch_found': {'ch_weight': 8,
                      'der_weights': [[w0, 3.375], [w1, 1.375], [w2, 1.375], [w3, 1.375], [w4, 1.375]],
                      'emp_weight': Counter({8: 256}),
                      'exact_weight': 9.074967361960082,
                      'input_diff': [[dp0, 0x0000], [dp1, 0x0002], [dp2, 0x8007], [dp3, 0x0005]],
                      'nonlinear_diffs': [[dx2, 0x0c00], [dx7, 0x0001], [dx12, 0x0000], [dx17, 0x0000], [dx22, 0x0000]],
                      'output_diff': [[dx8, 0x0002], [dx13, 0x0000], [dx18, 0x0000], [dx23, 0x0000]]},
     'key_ch_found': {'ch_weight': 0,
                      'der_weights': [],
                      'emp_weight': 0.0,
                      'exact_weight': 0,
                      'input_diff': [[dmk0, 0x0202], [dmk1, 0x2261], [dmk2, 0x22e7], [dmk3, 0x0503], [dmk4, 0x0103], [dmk5, 0x8000], [dmk6, 0x8000], [dmk7, 0x8000]],
                      'nonlinear_diffs': [],
                      'output_diff': [[dk3, 0x0404], [dk7, 0x0781], [dk11, 0x800b], [dk15, 0x0c00], [dk19, 0x0004]]}}


While there is no tutorial to learn how to use ArxPy yet,
the doctests and doctstrings from ``arxpy/smt/search.py`` provides
plenty of information and examples of searching characteristics.


Installation
============

ArxPy requires python3 (>= 3.7) and the following python libraries:

- cython
- sympy
- bidict
- cffi
- pySMT

These libraries can be easily installed with pip::

    pip install cython sympy bidict cffi pysmt

ArxPy also requires an SMT solver supporting the bit-vector theory,
installed through `pySMT <https://pysmt.readthedocs.io/en/latest/getting_started.html#getting-started>`_.
We recommend boolector. ::

    pysmt-install --btor

Optionally, hypothesis and yices can be installed to run the tests,
and sphinx and sphinx-rtd-theme to build the documentation.



