.. _installation:

************
Installation
************

CCL is a Python package. It works on Linux and Mac OS. Windows installation is
not supported. You can install it with

.. code-block:: bash

    $ conda install -c conda-forge pyccl

or

.. code-block:: bash

    $ pip install pyccl

or on `Google Colab <https://colab.research.google.com/>`_

.. code-block:: sh

   !pip install -q condacolab
   import condacolab
   condacolab.install()
   !mamba install pyccl

The setup requires `FFTW <https://www.fftw.org/>`_,
`GSL <https://www.gnu.org/software/gsl/>`_, `SWIG <https://www.swig.org/>`_,
and `Numpy <https://numpy.org/>`_. It is configured to install its dependencies
automatically, but as this is not always possible, you might need to install
the requirements separately if the build fails.

To use CCL, `Scipy <https://scipy.org/>`_ is required. If you want to use any
of the Boltzmann codes, or do perturbation theory calculations you will also
need `CAMB <https://camb.readthedocs.io/en/latest/>`_,
`CLASS <https://lesgourg.github.io/class_public/class.html>`_,
`ISiTGR <https://github.com/mishakb/ISiTGR>`_, and
`FAST-PT <https://github.com/JoeMcEwen/FAST-PT>`_. Note that the Python
wrappers for these packages are also needed.

Once installed, you can take CCL for a spin by following the :ref:`quickstart`
instructions.


.. _getting-cmake:

Getting CMake
=============

The ``pip`` installation of CCL requires that `CMake <https://cmake.org/>`_ is
installed. It is available from package managers like ``apt-get`` and
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


Known Issues
============

#. For some Mac OSX versions, the standard ``C`` headers are not in the usual spot, resulting in an
   error of ``fatal error: 'stdio.h' file not found``. This can be resolved with the command:

   .. code-block:: bash

      $ sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /

   which will install all the required headers into ``/usr/include``.

   On Mac OSX 10.15 or greater, this patch is no longer included.
   Instead, you can manually add the location of required headers to your CPATH by running the following, or adding to your ``.bash_profile``:

   .. code:: bash

      $ export CPATH="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/include"

#. Newer versions of Xcode do not automatically have the required command line tools installed. This can be done from the command line:

   .. code-block:: bash

      $ sudo xcode-select --install
      $ sudo xcodebuild -license


.. _uninstalling:

Uninstalling
============

``CCL`` can be uninstalled using the uninstallation functionality of the
package manager (i.e., ``conda`` or ``pip``) you used to install it. When in doubt,
first try with ``conda`` and then try with ``pip``. In either case, the command is

.. code-block:: bash

   $ [conda|pip] uninstall pyccl
