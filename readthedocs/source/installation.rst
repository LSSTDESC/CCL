************
Installation
************

CCL can be installed from ``pip``, ``conda``, or directly from source.
It is configured to install most of its requirements automatically. However, if
you want to use CCL with Boltzmann codes like ``CLASS`` or ``CAMB``, you will
need to make sure the ``Python`` wrappers for these packages are installed
as well. See the instructions for :ref:`boltzmann-codes` below.

CCL works on Linux or Mac OS. Windows installation is not supported.

Once installed, you can take CCL for a spin by following the :ref:`quickstart`
instructions.


With conda
==========

Installing CCL with ``conda`` will also install a copy of ``pycamb``, the
``Python`` wrapper for ``CAMB`` in your environment.

.. code-block:: bash

   $ conda install -c conda-forge pyccl


With pip
========

The PyPi installation will actually build a new copy of CCL from source as
the code is installed. In order to do this, you will need ``CMake`` installed
on your system. See the instructions below for :ref:`getting-cmake`.

Once you have ``CMake``, simply run:

.. code-block:: bash

   $ pip install pyccl


.. _boltzmann-codes:

Getting a Boltzmann Code
========================

In order to use CCL with a Boltzmann code, you will need the ``Python`` wrappers
for either ``CLASS`` or ``CAMB``.

CLASS
-----

In order to use CCL with ``CLASS``, you need to install the ``CLASS`` ``Python``
wrapper ``classy``. See their instructions
`here <https://github.com/lesgourg/class_public/wiki/Python-wrapper>`_.
Note that you may need to edit the ``Makefile`` in the ``CLASS`` repo to work
with your system. Please report any installation issues to the ``CLASS`` developers
on their `issues tracker <https://github.com/lesgourg/class_public/issues>`_.

CAMB
----

If you are working in a ``conda`` environment, then ``CAMB`` is available via

.. code-block:: bash

   $ conda install -c conda-forge camb

An installation with ``pip`` should work as well. See the `CAMB <https://github.com/cmbant/CAMB>`_
repo for more details. Note that if you installed CCL with ``conda``, ``camb``
should already be in your environment.


.. _getting-cmake:

Getting CMake
=============

The ``pip`` installation of CCL requires that ``CMake`` is installed on your
system. ``CMake`` is available from package managers like ``apt-get`` and
``homebrew``. You need version 3.2 or higher.

Ubuntu
------

.. code-block:: bash

   $ sudo apt-get install cmake

OS X
----

On MacOS X you can either install with a DMG from
`this page <https://cmake.org/download/>`_ or with a package manager such as
`homebrew <https://brew.sh/>`_, `MacPorts <https://www.macports.org/>`_, or
`Fink <(http://www.finkproject.org/>`_.

For instance with ``homebrew``, you can run

.. code-block:: bash

   $ brew install cmake


Known Installation Issues
=========================

#. For some Mac OSX versions, the standard ``C`` headers are not in the usual spot, resulting in an
   error of ``fatal error: 'stdio.h' file not found``. This can be resolved with the command:

   .. code:: bash

      $ sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /

   which will install all the required headers into ``/usr/include``.


.. _uninstalling:

Uninstalling ``CCL``
====================

``CCL`` can be uninstalled using the uninstallation functionality of the
package manager (i.e., ``conda`` or ``pip``) you used to install it. When in doubt,
first try with ``conda`` and then try with ``pip``. In either case, the command is

.. code-block:: bash

   $ [conda|pip] uninstall pyccl
