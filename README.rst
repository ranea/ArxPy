=========
ArxPy 0.3
=========

ArxPy is a python3 library to find XOR differential characteristics,
rotational-XOR characteristics, and impossible differentials
in ARX primitives (e.g., block ciphers) using SMT solvers.

ArxPy implements, improves and unifies the SMT-based methods of the following papers:

- Towards Finding Optimal Differential Characteristics for ARX: Application to Salsa20
- Observations on the SIMON Block Cipher Family
- An Easy-to-Use Tool for Rotational-XOR Cryptanalysis of ARX Block Ciphers
- Rotational-XOR Cryptanalysis of Reduced-round SPECK
- A Bit-Vector Differential Model for the Modular Addition by a Constant
- A Bit-Vector Differential Model for the Modular Addition by a Constant and its Applications to Differential and Impossible-Differential Cryptanalysis

ArxPy provides a complete documentation in `https://ranea.github.io/ArxPy/ <https://ranea.github.io/ArxPy/>`_
and an extensive suite of tests.


Usage
=====

First, the ARX cipher needs to be implemented following ArxPy interface.
There are several ARX ciphers implemented already (see the folder `arxpy/primitives`).
To implement a new cipher, the easiest way is to take a similar cipher
already implemented as a template and modify the python code directly.

Searching the optimal XOR differential characteristic can be done easily with
the function `round_based_search_SkCh()`. For example, searching for the best
2-round and 3-round differential characteristics of Speck32_64 can be done as follows

.. code:: python

    >>> from arxpy.differential.difference import XorDiff
    >>> from arxpy.smt.search_differential import SkChSearchMode, DerMode, round_based_search_SkCh
    >>> from arxpy.primitives import speck
    >>> cipher = speck.get_Speck_instance(speck.SpeckInstance.speck_32_64)
    >>> start_rounds, end_rounds = 4, 5
    >>> round_based_search_SkCh(cipher, XorDiff, 0, "btor", start_rounds, end_rounds,
    ...                         DerMode.Default, SkChSearchMode.Optimal, True, 0, None)
    Num rounds: 4
    Best characteristic found:
    (weight 5) 2800 0010 -> 8000 840a

    Num rounds: 5
    Best characteristic found:
    (weight 9) 0211 0a04 -> 8000 840a


Apart from the cipher and the number of rounds to search, the function
`round_based_search_SkCh()` has several parameters to adapt the search as needed.
For example, by changing `verbose_lvl`to `1` it is shown the intermediate
differences of the found characteristic and the weights of the non-linear operations.
This function stores each characteristic found as a `ChFound` object.
The explanation of these paremeters and the attributes of `ChFound` objects
can be found in the documentation.
Every function and class in ArxPy is documented with a description of
the object and with doctests!

If the argument `check` is `True` in `round_based_search_SkCh()`, the
weight of each characteristic found is checked by computing the empirical
weight (by sampling many plaintexts and keys and computing the
differential probability manually). This is useful since for
ARX ciphers the theoretical weight and the actual weight might differ
in some cases.

Searching the optimal related-key XOR differential characteristic can be done
in a similar way with the function `round_based_search_RkCh()`. For example,
searching for the best 4-round and 5-round related-key characteristics of
XTEA can be done as follows (this can take some minutes)

.. code:: python

    >>> from arxpy.differential.difference import XorDiff
    >>> from arxpy.smt.search_differential import RkChSearchMode, DerMode, round_based_search_RkCh
    >>> from arxpy.primitives.xtea import XteaCipher
    >>> cipher = XteaCipher
    >>> round_based_search_RkCh(cipher, XorDiff, 0, 0, "btor", 4, 5, DerMode.Default,
    ...                         DerMode.Default, True, RkChSearchMode.OptimalMinSum, True, 0, None)
    Num rounds: 4
    Best related-key characteristic found:
    E: (weight 0) 80000000 00000000 -> 00000000 80000000 | K: (weight 0) 80000000 00000000 80000000 00000000 -> 80000000 00000000 00000000 80000000

    Num rounds: 5
    Best related-key characteristic found:
    E: (weight 0) 80000000 00000000 -> 00000000 00000000 | K: (weight 0) 80000000 00000000 00000000 00000000 -> 80000000 00000000 00000000 00000000 00000000


A related-key characteristic is split into two characteristics, one
describing the propagation of differences through the key schedule
('key_ch_found') and another one describing the propagation of
differences through the encryption function ('enc_ch_found').

To search for rotational-XOR characteristics, the same functions
`round_based_search_SkCh()` and `round_based_search_RkCh()` can be used
but using  `RXDiff` as the `difference_type` argument. For example,
searching for 4-round and 5-round rotational-XOR differential characteristics
of CHAM_64_128 can be done in the following way

.. code:: python

    >>> from arxpy.differential.difference import RXDiff
    >>> from arxpy.smt.search_differential import RkChSearchMode, DerMode, round_based_search_RkCh
    >>> from arxpy.primitives import cham
    >>> cipher = cham.get_Cham_instance(cham.ChamInstance.cham_64_128)
    >>> round_based_search_RkCh(cipher, RXDiff, 0, 0, "btor", 4, 5, DerMode.Default,
    ...                         DerMode.Default, True, RkChSearchMode.FirstMinSumValid, True, 0, None)
    Num rounds: 4
    Best related-key characteristic found:
    E: (weight 5) 8000 0002 8006 0004 -> 0100 0001 0080 0001 | K: (weight 0) 6123 0281 c246 6020 8000 8000 8000 8000 -> 8004 8681 0009 8000

    Num rounds: 5
    Best related-key characteristic found:
    E: (weight 8) 0000 0002 8007 0005 -> 0002 0000 0000 0000 | K: (weight 0) 0202 2261 22e7 0503 0103 8000 8000 8000 -> 0404 0781 800b 0c00 0004


While there is no tutorial to learn how to use ArxPy yet,
the doctests and doctstrings from `arxpy/smt/search.py` provides
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
