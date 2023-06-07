********************
Development Workflow
********************

Contributions to CCL are made possible with pull requests (PRs). This section
navigates you through the pull request review process and the pre-merge
checklist.


Writing Code
============

Python-level
------------
After completing the :ref:`devinstall`, you can change the ``Python`` library
in your local copy of CCL. Changes are visible on restart of the interpreter.
Refer to the :ref:`code_guidelines` for best practices for writing clean,
maintainable code.


C-level
-------
Unless writing your code in ``C`` provides a significant speed advantage, new
code should be written in ``Python``.

Source code lives in ``src/`` and header files live in ``include/``. For the
build, the source file should be specified in the ``CCL_SRC`` variable of
``CMakeLists.txt``.

To make the new code visible in ``Python``, include it in an interface file
(``pyccl/*.i``) as needed. New interface files should be declared in
``pyccl.ccl.i``. ``SWIG`` will do the rest, and the function will be available
in ``pyccl.ccllib``. Refer to the :ref:`pycint` for more details.

Occasionally, the build will not work correcly. This hints at multiple parallel
installations. To overcome this, refer to :ref:`devinstall`.


Continuous Integration (CI)
===========================
All the things outlined in this paragraph are tested automatically in every
commit of PRs, as well as in the merge commit to the default branch. We
recommend running these tests locally first in order to ensure that the CI will
pass, but you may also follow the online links at the bottom of your PR to
review the output of these tests.

.. image:: _static/ci.png


Unit Tests
----------
CCL uses `pytest <https://docs.pytest.org/en/7.3.x/>`_ as the testing suite.
It can be installed with ``[pip|conda] install pytest``.

**Writing Unit Tests**

Please follow the following guidelines when writing unit tests.

#. All unit tests should be written as modules in the ``pytest/tests``
   submodule.

#. Each unit test is a function that does some operation with CCL and then
   uses a ``Python`` ``assert`` statement. A unit test is marked by ``pytest``
   as failed when the ``assert`` statement fails.

#. The unit tests should be fast, ideally less than a second or so. This
   requirement means avoiding running Boltzmann codes when it is not needed.

#. The unit tests file should have a name that matches the python module it is
   testing (e.g., ``pyccl/tests/test_bcm.py`` for ``pyccl/bcm.py``).

#. The unit test function itself should have a name that starts with ``test_``
   and is also descriptive.

Here is an example of a unit test, ensuring consistency of :math:`\sigma_8`:

.. code-block:: python
    :caption: pyccl/tests/test_core.py

    import numpy as np
    import pyccl as ccl

    def test_cosmology_sigma8_consistent():
        cosmo = ccl.Cosmology(
            Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.95,
            transfer_function='bbks')
        assert np.allclose(ccl.sigma8(cosmo), 0.8)


**Runing Unit Tests**

To run the unit tests, execute

.. code-block:: bash

   $ pytest -vv pyccl

from the top-level repository directory. Errors will be printed at the end.
Add ``--cov=pyccl --cov-report=lcov:lcov.info`` to the above command to export
code coverage info.

Other useful ``pytest`` options include

- ``-k <name of test>``: adding this option with the name of the test will
  force ``pytest`` to only run that test
- ``-s``: this option makes ``pytest`` print all output to STDOUT instead of
  its usual behavior of suppressing output
- ``-x``: this option forces ``pytest`` to stop after the first failed test
- ``--pdb``: this option will launch the ``Python`` debugger for failed tests,
  usually you want to combine it with ``-s -x -k <name of test>`` in order to
  debug a single failed test

Make sure that any modifications to the code do not introduce warnings in
testing unless absolutely necessary.


Benchmarks
----------
Nearly every new feature in CCL is benchmarked against an independent
implementation. If you are adding a new feature, make sure someone can provide
you with an independent output for cross-checks. Independent codes should be
publicly available and a script to run them should be provided such that the
user can reproduce the benchmarks.

**Writing Benchmarks**

Benchmarks in CCL follow the ``pytest`` conventions for the CCL unit tests,
but are located in the ``benchmarks/`` directory. Any data needed for the
benchmark should be added to the ``benchmarks/data`` directory. Code to
produce the benchmark data can be put in ``benchmarks/data/code``. If it cannot
be included in the repository, then a link on the `wiki
<https://github.com/LSSTDESC/CCL/wiki/Benchmarks>`_
is acceptable.

The benchmark itself should read the data in the benchmarks data directory and
compare to the same computation done in CCL. Here is an example benchmark
testing the CCL BBKS transfer function.

.. code-block:: python
    :caption: benchmarks/test_bbks.py

    import numpy as np
    import pytest

    import pyccl as ccl

    BBKS_TOLERANCE = 1.0e-5


    @pytest.mark.parametrize(
        'model,w0,wa',
        [(1, -1.0, 0.0),
         (2, -0.9, 0.0),
         (3, -0.9, 0.1)])
    def test_bbks(model, w0, wa):
        cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8,
                              n_s=0.96, Neff=0, w0=w0, wa=wa, T_CMB=2.7,
                              transfer_function='bbks')

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

**Running Benchmarks**

Benchmarks can be run by executing

.. code-block:: bash

   $ pytest -vv benchmarks

from the top-level directory of the repository. As with unit testing,
``pytest`` may be called with a number of options to assist debugging.
Add ``--cov=pyccl --cov-report=lcov:lcov.info --cov-append`` to the above
command to export code coverage info.

Make sure that any modifications to the code do not introduce warnings in
testing unless absolutely necessary.


Linting
-------
CCL uses `flake8 <https://flake8.pycqa.org/en/latest/>`_ as a lint checker.
You can run this tool locally by executing

.. code-block:: bash

   $ flake8 .

from the top-level directory. Any problems will be printed to ``STDOUT``. No
output indicates that ``flake8`` has succeeded.


Coverage
--------
CCL checks code coverage with `Coveralls <https://coveralls.io>`_. If you have
exported coverage info with testing, you may check the coverage locally with
`coverage.py <https://coverage.readthedocs.io/en/7.2.5/>`_:

.. code-block:: bash

    $ coverage html

This will create a directory ``htmlcov/``. You may navigate it with file
``index.html`` to verify that any code modifications are covered in testing.


Docs Build
----------
To build the documentation, follow these steps:

.. code-block:: bash

   $ cd readthedocs
   $ make clean
   $ make SPHINXOPTS="-W --keep-going" html

``-W`` converts warnings to errors, forcing the build to fail, and
``--keep-going`` ensures that the build does not halt on error. Once the build
is finished, you may inspect the docs in ``readthedocs/_build/index.html`` to
make sure the formatting is correct.


.. _cclnote:

Building the CCL Note
=====================
The CCL note is a latex'ed documented located in ``doc/0000-ccl_note``. It is
used to document the scientific content of the CCL library. Note that
documentation of the actual APIs and code should reside in the docstrings and
other code comments.

To compile the CCL note, type ``make`` in the ``doc/0000-ccl_note`` directory.

If you need to modify the note, the files to modify are:

- ``authors.csv``: To document your contribution.
- ``main.tex``: To detail the changes to the library.
- ``main.bib``: To add new references.


New Releases
============

Semantic Versioning
-------------------
CCL follows the conventions of `semantic versioning <https://semver.org/>`_.
Releases are tagged with 3 numbers: ``v{MAJOR}.{MINOR}.{PATCH}``. Changes to
the codebase which break the API should increment the major version number.

All API changes should be discussed with the CCL team through one of the
available channels (e.g. Slack) before being merged, and ideally before the
development has started.

Deployment
----------
One of the repo admins is able to deploy a new release. The procedure is as
follows:

#. Make sure any API changes are documented in ``CHANGELOG.md``
#. Commit to the default branch
#. Create a new release from the GitHub interface here:
   https://github.com/LSSTDESC/CCL/releases/new
#. Manually create a source distribution from the root CCL folder:

   .. code-block:: bash

      $ python setup.py sdist

   This command will create a ``.tar.gz`` file in the ``dist`` folder.
   CAUTION: Only build and upload the source distribution, not a binary wheel!
#. Upload source distribution to PyPi using ``twine``
   (can be installed using ``pip`` or ``conda``):

   .. code-block:: bash

      $ twine upload  dist/pyccl-x.x.x.tar.gz

   Make sure your ``twine`` and ``setuptools`` packages are up to date, or the
   markdown formatting of the ``README.md`` will not be correctly processed on
   the CCL PyPi page.
#. The ``conda-forge`` automated release bots will detect the new PyPi release
   and automatically make a PR on the CCL feedstock. Once this PR is merged,
   the new CCL release will be available on ``conda`` after a few hours.
#. Rebuild and redeploy the documentation. Note that you may need to adjust the
   major version number in ``readthedocs/conf.py`` if the major version number
   has been incremented.
