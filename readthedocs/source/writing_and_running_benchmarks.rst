.. _benchmarks:

******************************
Writing and Running Benchmarks
******************************

Nearly every new feature in CCL is benchmarked against an independent implementation.
If you are adding a new feature, make sure someone can provide you with an independent
output for cross-checks. Independent codes should be publicly available and a script
to run them should be provided such that the user can reproduce the benchmarks.


Running the Benchmarks
----------------------

After CCL is installed, the benchmarks can be run by executing

.. code-block:: bash

   $ pytest -vv benchmarks

from the top-level directory of the repository. The :ref:`unittests` section
contains useful hints on using ``pytest`` to debug individual tests/benchmarks.


Writing a Benchmark
-------------------

Benchmarks in CCL follow the ``pytest`` conventions for the CCL unit tests,
but are located in the ``benchmarks/`` directory. Any data needed for the
benchmark should be added to the ``benchmarks/data`` directory. Code to
produce the benchmark data can be put in ``benchmarks/data/code``. If it cannot
be included in the repository, then a link on the `wiki <https://github.com/LSSTDESC/CCL/wiki/Benchmarks>`_
is acceptable.

The benchmark itself should read the data in the benchmarks data directory and
then compare to the same computation done in CCL. Here is an example benchmark
testing the CCL BBKS transfer function (located in the file ``benchmarks/test_bbks.py``)

.. code-block:: python

   import numpy as np
   import pyccl as ccl
   import pytest

   BBKS_TOLERANCE = 1.0e-5


   @pytest.mark.parametrize(
       'model,w0,wa',
       [(1, -1.0, 0.0),
        (2, -0.9, 0.0),
        (3, -0.9, 0.1)])
   def test_bbks(model, w0, wa):
       cosmo = ccl.Cosmology(
           Omega_c=0.25,
           Omega_b=0.05,
           h=0.7,
           sigma8=0.8,
           n_s=0.96,
           Neff=0,
           m_nu=0.0,
           w0=w0,
           wa=wa,
           T_CMB=2.7,
           m_nu_type='normal',
           Omega_g=0,
           Omega_k=0,
           transfer_function='bbks',
           matter_power_spectrum='linear')

       data = np.loadtxt('./benchmarks/data/model%d_pk.txt' % model)

       k = data[:, 0] * cosmo['h']
       for i in range(6):
           a = 1.0 / (1.0 + i)
           pk = data[:, i+1] / (cosmo['h']**3)
           pk_ccl = ccl.linear_matter_power(cosmo, k, a)
           err = np.abs(pk_ccl/pk - 1)
           assert np.allclose(err, 0, rtol=0, atol=BBKS_TOLERANCE)

Note that the benchmarks are executed from the top-level of the repository so
that the file paths for data are ``benchmarks/data/<data file>``.
