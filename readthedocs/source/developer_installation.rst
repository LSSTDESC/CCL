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

   $ pip install -e .

from the top-level directory in the repository. You will need ``CMake`` in
order to install CCL in this way. See :ref:`getting-cmake` for help installing
``CMake`` if you do not already have it. In order to run the tests,
you will need both ``CAMB`` and ``CLASS`` installed. See the instructions for
:ref:`boltzmann-codes` for details.


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
