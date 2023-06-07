.. _devinstall:

**********************
Developer Installation
**********************

Thank you for contributing to improving the codebase!

The easiest way to install CCL if you are a developer is in its own, isolated
``conda`` environment.

.. code-block:: bash

   $ conda create -n ccl-dev compilers cmake swig pyccl pytest flake8
   $ conda activate ccl-dev

Make sure to follow the `conda-forge instructions
<https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages
-from-conda-forge>`_ and that your ``PYTHONPATH`` is clear before you start.

Then, download a fresh copy of the repo, and build it using ``pip`` in
developer mode. These are also the steps when you do not want to set up a
``conda`` environment.

.. code-block:: bash

   $ git clone https://github.com/LSSTDESC/CCL.git
   $ cd CCL
   $ pip install -e .

This creates a symlink to the Python code and any changes do not require
reinstall. Note that changes in the ``C`` code do require recopiling the
library.

For the developer installation you will need all packages listed in
``.github/environment.yml`` as well as those listed in
``readthedocs/requirements.txt``.

To compile the ``C`` code with debugging symbols, add the ``--debug`` option:

.. code-block:: bash

   $ pip install --no-deps -e . --global-option=--debug

To remove old build products that might conflict when re-compiling CCL, run

.. code-block:: bash

   $ python setup.py clean


Python Debug Mode
=================
The Python wrapper suppresses all but the final error message that spawns from
the ``C`` code, which can impede debugging. To that end, you may enable debug
mode with :func:`~pyccl.pyutils.debug_mode`.
