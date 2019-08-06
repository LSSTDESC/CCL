************************
Installation
************************

With pip
========

CCL is available as a Python package through PyPi. To install, simply run:

.. code-block:: bash

   $ pip install pyccl

This should work as long as :code:`CMake` is installed on your system. Once installed, take it for a spin in by following some example notebooks `here <https://github.com/LSSTDESC/CCL/blob/master/examples>`_.

If you want the C library available, you have to follow the manual installation.

Dependencies and Requirements
=============================

You can also install CCL from the source, however in order to compile CCL you need a few libraries:

* GNU Scientific Library `GSL <https://www.gnu.org/software/gsl/>`_, version 2.1 or above
* `FFTW3 <http://www.fftw.org/>`_ version 3.1 or above
* `CLASS <http://class-code.net/>`_ version 2.6.3 or above
* FFTlog (`here <http://casa.colorado.edu/~ajsh/FFTLog/>`_ and `here <https://github.com/slosar/FFTLog>`_) is provided within CCL, with minor modifications.

Additionally, to build the code you will need

* `CMake <https://cmake.org/>`_ version 3.2 or above.
* `SWIG <http://www.swig.org/>`_ is needed if you wish to modify CCL and have it availabe in Python.

:code:`CMake` is the only requirement that needs to be installed manually if you are using :code:`pip`.

On Ubuntu:

.. code-block:: bash

   $ sudo apt-get install cmake

On MacOS X you can either install with a DMG from `this page <https://cmake.org/download/>`_ or with a package manager such as brew, MacPorts, or Fink. For instance with brew:

.. code-block:: bash

   $ brew install cmake

You will avoid potential issues if you install :code:`GSL` and :code:`FFTW` on your system before building CCL, but is only necessary if you want to properly install the C library. Otherwise :code:`CMake` will automatically download and build the missing requirements in order to compile CCL.

To install all the dependencies at once, and avoid having :code:`CMake` recompiling them, for instance on Ubuntu:

.. code-block:: bash

   $ sudo apt-get install cmake swig libgsl-dev libfftw3-dev


Compiling and Installing the CCL C library
==========================================
To download hte latest version of CCL:

.. code-block:: bash

   $ git clone https://github.com/LSSTDESC/CCL.git
   $ cd CCL

or download and extract the latest stable release from `here <https://github.com/LSSTDESC/CCL/releases>`_. Then, from the base CCL directory run:

.. code-block:: bash

   $ mkdir build && cd build
   $ cmake ..

This will run the configuration script, try to detect the required dependencies on your machine and generate a Makefile. Once CMake has been configured, to build and install the library simply run for the :code:`build` directory:

.. code:: bash

   $ make
   $ make install

Often admin privileges will be needed to install the library. If you have those just type:

.. code:: bash

   $ sudo make install


**Note**: This is the default install procedure, but depending on your system you might want to customize the intall process. Here are a few common configuration options:

In case you have several C compilers, you can direct which one for :code:`CMake` to use by setting the environment variable :code:`CC` **before** running :code:`CMake`:

.. code:: bash

   $ export CC=gcc

By default, :code:`CMake` will try to install CCL in :code:`/usr/local`. If you would like to instead install elsewhere (such as if you don't have admin privileges), you can specify it **before** running :code:`CMake` by doing:

.. code:: bash

   $ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..

This will instruct :code:`CMake` to install CCL in the following folders: :code:`/path/to/install/include`,:code:`/path/to/install/share`, and :code:`/path/to/install/lib`.

Depending on where you install CCL you might need to add the installation path to your :code:`PATH` and :code:`LD_LIBRARY_PATH` environment variables. In the default case, this is accomplished with:

.. code:: bash

   $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
   $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/bin

To make sure that everything is working properly, you can run all unit tests after installation by running from the root CCL directory:

.. code:: bash

   $ check_ccl

Assuming that the tests pass, you have successfully installed CCL!

If you ever need to uninstall CCL, run the following from the :code:`build` directory:

.. code:: bash

   $ make uninstall

You may need to prepend a :code:`sudo` if you installed CCL in a protected folder.

Once the CLASS library is installed, `CCL` can be easily installed using an
*autotools*-generated configuration file. To install `CCL`, from the base directory
(the one where this file is located) run:

Often admin privileges will be needed to install the library. If you have those just type:

.. code:: bash

   sudo make install

If you don't have admin privileges, you can still install the library in a user-defined directory by running

.. code:: bash

   ./configure --prefix=/path/to/install
   make
   make install

where ``/path/to/install`` is the absolute path to the directory where you want the library to be installed. If non-existing, this will create two directories, ``/path/to/install/include`` and ``/path/to/install/lib``, and the library and header files will be installed there. Note that, in order to use `CCL` with your own scripts you'll have to add ``/path/to/install/lib`` to your ``LD_LIBRARY_PATH``. `CCL` has been successfully installed on several different Linux and Mac OS X systems.

To make sure that everything is working properly, you can run all unit tests after installation by running

.. code:: bash

   make check

Assuming that the tests pass, you can then move on to installing the Python wrapper (optional).

After pulling a new version of `CCL` from the `GitHub repository <https://github.com/LSSTDESC/CCL>`_, you can recompile the library by running:

.. code:: bash

   make clean; make uninstall
   make
   make install

Install the pyccl Python module
===============================
CCL also comes with a Python wrapper, called pyccl, which can be built and installed regardless of whether you install the C library. For convenience, we provide a PyPi hosted package which can be installed simply by running:

.. code:: bash

   $ pip install pyccl # append --user for single user install

This only assumes that :code:`CMake` is available on your system, you don't need to download the source yourself.

You can also build and install pyccl from the CCL source, again without necessarily installing the C library. Download the latest version of CCL:

.. code:: bash

   $ git clone https://github.com/LSSTDESC/CCL.git
   $ cd CCL

And from the root CCL folder, simply run:

.. code:: bash

   $ python setup.py install # append --user for single user install

The pyccl module will be installed into a sensible location in your :code:`PYTHONPATH`, and so should be picked up automatically by your Python interpreter. You can then simply import the module using import pyccl.

You can quickly check whether pyccl has been installed correctly by running :code:`python -c "import pyccl"` and checking that no errors are returned.

For a more in-depth test to make sure everything is working, run from the root CCL directory:

.. code:: bash

   python setup.py test

This will run the embedded unit tests (may take a few minutes).

Whatever the install method, if you have :code:`pip` installed, you can always uninstall the pyton wrapper by running:

.. code:: bash

   pip uninstall pyccl

For quick introduction to `CCL` in Python look at notebooks in ``**_tests/_**``.

Compiling against an external version of CLASS
==============================================

The default installation procedure for CCL implies automatically downloading and installing a tagged version of CLASS. Optionally, you can also link CCL against a different version of CLASS. This is useful if you want to use a modified version of CLASS, or a different or more up-to-date version of the standard CLASS.

To compile CCL with an external version of CLASS, just run the following :code:`CMake` command at the first configuration step of the install (from the build directory, make sure it is empty to get a clean configuration):

.. code:: bash

   $ cmake -DEXTERNAL_CLASS_PATH=/path/to/class ..

the rest of the build process should be the same.

Known Installation Issues
=========================

#. If upon running the C tests you get an error from CLASS saying it cannot find
   the file ``sBBN_2017.dat``, it means that the CLASS parameter files are not properly
   installed on your system. Make sure you have indeed installed the C library by running:

   .. code:: bash

      $ make install

   from the ``CCL/build`` directory.

#. If you are having issues with GSL versions linking, please try the following during the configuration step:

   .. code:: bash

      ./configure CFLAGS="-I/usr/local/include" LDFLAGS="-L/usr/local/lib"

#. If you move or delete the source directory after installing CCL, some functions
   may fail. The source directory contains files needed by CLASS (which is contained
   within CCL) at run-time.

#. If you are planning to compile your own file that calls CCL, then you should
   add the following to your ``.bashrc``:

   .. code:: bash

      $ export LD_LIBRARY_PATH=/path/to/where/ccl/is/installed/lib:$LD_LIBRARY_PATH

#. In some Mac systems, the ``angpow`` installation script might fail on the first
   run with the message

   .. code:: bash

      $ CMake Error: The source directory "CCL/angpow" does not appear to contain CMakeLists.txt.
      $ Specify --help for usage, or press the help button on the CMake GUI.

   If this happens, try running it a second time.

#. In some Mac systems, depending on the XCode version of the C/C++ compiler,
   you can get a compilation error. For a proposed solution,
   see `here <https://github.com/LSSTDESC/CCL/issues/379#issuecomment-391456284>`_.

#. If using a ``sudo make install`` for the C library and attempting to install the
   developer Python library, it may be necessary to include a ``sudo``
   for your chosen Python installation.

#. If you are building CCL using a conda environment, be mindful of the known
   issues documented `here <https://github.com/LSSTDESC/CCL/issues/548>`_.

#. For some Mac OSX versions, the standard C headers are not in the usual spot, resulting in an
   error of ``fatal error: 'stdio.h' file not found`` while attempting to install
   CLASS. This can be resolved with the command:

   .. code:: bash

      $ sudo installer -pkg /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg -target /

   which will install all the required headers into ``/usr/include``.
