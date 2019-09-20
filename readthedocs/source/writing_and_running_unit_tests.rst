.. _unittests:

******************************
Writing and Running Unit Tests
******************************

CCL uses the ``pytest`` package to run a suite of unit tests. This package can
be installed via ``pip`` or ``conda`` with ``[pip|conda] install pytest``.

Running the Unit Tests
======================

To run the unit tests, execute

.. code-block:: bash

   pytest -vv pyccl

from the top-level repository directory. Any errors will be reported at the end.

Other useful ``pytest`` options include

- ``-k <name of test>``: adding this option with the name of the test will force
  ``pytest`` to only run that test
- ``-s``: this option makes ``pytest`` print all output to STDOUT instead of its
  usual behavior of suppressing output
- ``-x``: this option forces ``pytest`` to stop after the first failed test
- ``--pdb``: this option will launch the ``Python`` debugger for failed tests,
  usually you want to combine it with ``-s -x -k <name of test>`` in order to
  debug a single failed test


Writing Unit Tests
==================

Please follow the following guidelines when writing unit tests.

#. All unit tests should be written as modules in the ``pytest/tests`` submodule.

#. Each unit test is a function that does some operation with CCL and then
   uses a ``Python`` ``assert`` statement. A unit test is marked by ``pytest``
   as failed when the ``assert`` statement fails.

#. The unit tests should be fast, ideally less than a second or so. This requirement
   means avoiding running ``CLASS`` or ``CAMB`` when it is not needed.

#. The unit tests file should have a name that matches the python module it is
   testing (e.g., ``pyccl/tests/test_bcm.py`` for ``pyccl/bcm.py``).

#. The unit test function itself should have a name that starts with ``test_``
   and is also descriptive.

An example of a unit test would be a file at ``pyccl/tests/test_core.py`` with
the test

.. code-block:: python

   import numpy as np
   import pyccl

   def test_cosmology_sigma8_consistent():
       cosmo = pyccl.Cosmology(
           Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.95,
           transfer_function='bbks')
       assert np.allclose(pyccl.sigma8(cosmo), 0.8)


Building and Running CCL's C Unit Tests
=======================================

CCL has a few vestigial unit tests written in ``C``. Once you have ``CMake``
installed (see :ref:`getting-cmake`), you can build them with the following
commands from the top-level CCL directory

.. code-block:: bash

   $ mkdir -p build
   $ cd build
   $ cmake ..
   $ make check_ccl

Then you can run the tests via

.. code-block:: bash

   $ ./check_ccl
