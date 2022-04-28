.. _devinstall:

**********************
Developer Installation
**********************

To develop new code for CCL, you need to install it using ``pip``'s development
install. This installation method creates a symbolic link between your ``Python``
``site-packages`` directory and the copy of CCL you are working on locally. Thus
when you change CCL ``Python`` code, you do not need to reinstall CCL in order
for the changes to be visible system-wide. Note, if you change the CCL ``C``
code, you will need to force CCL to recompile the code (and copy the resulting
``.so`` into the ``Python`` package) by rerunning the command below.

To install CCL using a ``pip`` developer installation, you can execute

.. code-block:: bash

   $ pip install --no-deps -e .

from the top-level directory in the repository. You will need ``CMake`` in
order to install CCL in this way. See :ref:`getting-cmake` for help installing
``CMake`` if you do not already have it. In order to run the tests,
you will need ``CAMB``, ``CLASS``, and ``FAST-PT`` installed. See the instructions for
:ref:`boltzmann-codes` and :ref:`getting-fast-pt` for details.

To compile the ``C`` code with debugging symbols, add the ``--debug`` option
when calling ``setup.py``:

.. code-block:: bash

   $ python setup.py --debug develop

Or with ``pip``:

.. code-block:: bash

   $ pip install --no-deps -e . --global-option=--debug

To remove old build products that might conflict when re-compiling CCL, run

.. code-block:: bash

   $ python setup.py clean


C-layer Dependencies and Requirements
=====================================

CCL has several C dependencies. The ``CMake`` build will download and
compile these automatically if they are not present on your system. However,
if you do have them installed locally in a spot accessible to ``CMake``, the
local versions will be used.

These dependencies are

* GNU Scientific Library `GSL <https://www.gnu.org/software/gsl/>`_, version 2.1 or above
* `FFTW3 <http://www.fftw.org/>`_ version 3.1 or above
* `CLASS <http://class-code.net/>`_ version 2.6.3 or above
* `SWIG <http://www.swig.org/>`_


Uninstalling ``CCL`` in Developer Mode
======================================

To uninstall ``CCL`` in developer mode, simply type

.. code-block:: bash

   $ pip uninstall pyccl


Boostrapping a CCL Development Environment with ``conda``
=========================================================

One of the easier ways to get started with CCL development is to use
``conda-forge`` to provide the third-party requirements above and the necessary
compilers. The following commands will get you started with a ``conda-forge``-based
development environment. Note that before you start, make sure to follow the
`conda-forge instructions <https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge>`_
for use and that your ``PYTHONPATH`` variable is not set.

Then do the following

.. code-block:: bash

   $ conda create -n ccl-dev compilers cmake swig pyccl pytest flake8
   $ conda activate ccl-dev
   $ conda uninstall pyccl --force
   $ git clone https://github.com/LSSTDESC/CCL.git
   $ cd CCL
   $ pip install --no-deps -e .

This set of commands leaves a copy of the compiled ``C`` extension in the checked out
copy of the code, e.g.,

.. code-block:: bash

   $ ls pyccl/*.so
   pyccl/_ccllib.so

If you make changes to the ``C`` library or checkout a new branch, simply rerun
``pip install --no-deps -e .`` to rebuild the library.
