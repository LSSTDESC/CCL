************
Installation
************

CCL can be installed from ``pip``, ``conda``, or directly from source.
It is configured to install most of its requirements automatically. However, if
you want to use CCL with Boltzmann codes like ``CLASS`` or ``CAMB``, carry out
perturbation theory calculations with ``FAST-PT`` or ``velocileptors``, or make
use some of the emulators implemented, you will need to make sure these packages
are installed as well. A list of all current alternative dependencies can be
found `here <https://github.com/LSSTDESC/CCL/blob/master/.github/environment.yml>`__.
See the instructions for :ref:`boltzmann-codes`, :ref:`getting-pt`, and
:ref:`getting-emus` below.

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


Google Colab
============

To install ``pyccl`` on https://colab.research.google.com then one way is the following

.. code-block:: bash

   !pip install -q condacolab
   import condacolab
   condacolab.install()
   !mamba install pyccl


.. _boltzmann-codes:

Getting a Boltzmann Code
========================

In order to use CCL with a Boltzmann code, you will need the ``Python`` wrappers
for either ``CLASS`` or ``CAMB``.

CLASS
-----

In order to use CCL with ``CLASS``, you need to install the ``CLASS`` ``Python``
wrapper ``classy``. See their instructions
`here <https://github.com/lesgourg/class_public/wiki/Python-wrapper>`__.
Note that you may need to edit the ``Makefile`` in the ``CLASS`` repo to work
with your system. Please report any installation issues to the ``CLASS`` developers
on their `issues tracker <https://github.com/lesgourg/class_public/issues>`__.

CAMB
----

If you are working in a ``conda`` environment, then ``CAMB`` is available via

.. code-block:: bash

   $ conda install -c conda-forge camb

An installation with ``pip`` should work as well. See the `CAMB <https://github.com/cmbant/CAMB>`__
repo for more details. Note that if you installed CCL with ``conda``, ``camb``
should already be in your environment.

ISiTGR
------

In order to use CCL with ``ISiTGR``, you need to install the ``ISiTGR`` ``Python``
wrapper ``isitgr`` with:

   $ pip install isitgr [--user]

See further instructions `here <https://github.com/mishakb/ISiTGR>`__.

If you are working in a ``conda`` environment, then ``ISiTGR`` is also available via

.. code-block:: bash

   $ conda install -c conda-forge isitgr

Note that if you installed CCL with ``conda``, ``isitgr``
should already be in your environment.


.. _getting-pt:

Getting PT packages
===================

Getting FAST-PT
---------------

To use ``FAST-PT`` with CCL, you can install it with:

.. code-block:: bash

   $ pip install fast-pt

Note the hyphen in the package name! You can also get it directly from the
`FAST-PT <https://github.com/JoeMcEwen/FAST-PT>`__ repo.

Getting velocileptors
---------------------

To use ``velocileptors`` with CCL, you can install it with:

.. code-block:: bash

   $ python3 -m pip install -v git+https://github.com/sfschen/velocileptors

See full instructions in the ``velocileptors``
`github repo <https://github.com/sfschen/velocileptors>`__.


.. _getting-emus:

Getting emulators
=================

The following emulators with external dependencies are currently supported
in CCL.

BACCO emu
---------

`Source code <https://bitbucket.org/rangulo/baccoemu>`__. Installation:

.. code-block:: bash

   $ python3 -m pip install baccoemu

MiraTitan mass function emulator
--------------------------------

`Source code <https://github.com/SebastianBocquet/MiraTitanHMFemulator>`__. Installation:

.. code-block:: bash

   $ python3 -m pip install MiraTitanHMFemulator


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
`this page <https://cmake.org/download/>`__ or with a package manager such as
`homebrew <https://brew.sh/>`__, `MacPorts <https://www.macports.org/>`__, or
`Fink <(http://www.finkproject.org/>`__.

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

   On Mac OSX 10.15 or greater, this patch is no longer included.
   Instead, you can manually add the location of required headers to your CPATH by running the following, or adding to your ``.bash_profile``:

   .. code:: bash

      $ export CPATH="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include"

#. Newer versions of Xcode do not automatically have the required command line tools installed. This can be done from the command line:

    .. code:: bash

      $ sudo xcode-select --install
      $ sudo xcodebuild -license


.. _uninstalling:

Uninstalling ``CCL``
====================

``CCL`` can be uninstalled using the uninstallation functionality of the
package manager (i.e., ``conda`` or ``pip``) you used to install it. When in doubt,
first try with ``conda`` and then try with ``pip``. In either case, the command is

.. code-block:: bash

   $ [conda|pip] uninstall pyccl
