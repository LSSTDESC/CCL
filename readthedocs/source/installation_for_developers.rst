****************************
Installation for developers
****************************

Development workflow
====================

Installing CCL on the system is not a good idea when doing development, you can compile and run all the libraries and examples directly from your local copy. The only subtlety when not actually installing the library is that one needs to define the environment variable :code:`CCL_PARAM_FILE` pointing to :code:`include/ccl_params.ini`:

.. code-block:: bash

   $ export CCL_PARAM_FILE=/path/to/your/ccl/folder/include/ccl_params.ini

Failure to define this environment variable will result in violent segmentation faults!

Working on the C library
========================

Here are a few common steps when working on the C library:

Cloning a local copy and CCL and compiling it:

.. code-block:: bash

   $ git clone https://github.com/LSSTDESC/CCL
   $ mkdir -p CCL/build && cd CCL/build
   $ cmake ..
   $ make

Updating local copy from master, recompiling what needs recompiling, and running the test suite:

.. code-block:: bash

   $ git pull      # From root folder
   $ make -Cbuild  # The -C option allows you to run make from a different directory
   $ build/check_ccl

Compiling (or recompiling) an example in the :code:`CCL/examples` folder:

.. code-block:: bash

   $ cd examples  # From root folder
   $ make -C../build ccl_sample_pkemu
   $ ./ccl_sample_pkemu

Reconfiguring from scratch (in case something goes terribly wrong):

.. code-block:: bash

   $ cd build
   $ rm -rf *
   $ cmake ..
   $ make

Working on the Python library
=============================

Here are a few common steps when working on the Python module:

Building the python module:

.. code-block:: bash

   $ python setup.py build

After that, you can start your interpreter from the root CCL folder and import pyccl.

Running the tests after a modification of the C library:

.. code-block:: bash

   $ python setup.py build
   $ python setup.py test
