************************
Installation
************************

In order to compile CCL you need a few libraries:

* GNU Scientific Library `GSL <https://www.gnu.org/software/gsl/>`_. Note that `CCL` uses version 2.1 or higher of GSL (which is not yet standard in all systems).
* The `SWIG <http://www.swig.org/>`_ Python wrapper generator is not needed to run `CCL`, but must be installed if you intend to modify `CCL` in any way.
* `FFTW3 <http://www.fftw.org/>`_ is required for computation of correlation functions.
* FFTlog (`here <http://casa.colorado.edu/~ajsh/FFTLog/>`_ and `here <https://github.com/slosar/FFTLog>`_) is provided within `CCL`, with minor modifications.
* The C library associated to the CLASS code. The installation of this library is described below.

Installing CLASS
================
CCL uses CLASS as one of the possible ways of computing the matter power spectrum. In order to communicate with CLASS, CCL must be linked to its library. Before installing CCL proper you must therefore install this library first. Since this process is not necessarily straightforward, we provide a python script ``class_install.py`` that automatically downloads and install the latest tagged stable version of CLASS. You should run this script (``python class_install.py``) before carrying out the next steps. By default, the script assumes that your main C compiler is ``gcc``. If that's not the case, pass the name of your C compiler to the script via the command-line argument ``--c_comp`` (i.e. ``python class_install.py --c_comp=[name of compiler]``). Type ``python class_install.py -h`` for further details.

This procedure has one final caveat: if you already have a working installation of CCL, ``class_install.py`` may fail the first time you run it. This can be fixed by either simply running ``class_install.py`` a second time, or by starting from scratch (i.e. downloading or cloning CCL).

.. note::
   
   If you want to use your own version of CLASS, you should follow the steps described in the section "Compiling against an external version of CLASS" below.

C-only installation
================================
Once the CLASS library is installed, `CCL` can be easily installed using an *autotools*-generated configuration file. To install `CCL`, from the base directory (the one where this file is located) run:

.. code:: bash

   ./configure
   make
   make install

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

After pulling a new version of `CCL` from the `git <https://github.com/LSSTDESC/CCL>`_ repository, you can recompile the library by running:

.. code:: bash
	  
   make clean; make uninstall
   make
   make install


Known installation issues
=========================
1. If you are having issues with GSL versions linking, please try the following during the configuration step:

.. code:: bash

   ./configure CFLAGS="-I/usr/local/include" LDFLAGS="-L/usr/local/lib"

2. If you are having issues with FFTW linking, please make sure you have the latest version correctly installed. See more on `FFTW homepage <http://www.fftw.org/download.html>`_.
3. If you move or delete the source directory after installing `CCL`, some functions may fail. The source directory contains files needed by CLASS (which is contained within `CCL`) at run-time.
4. If you are planning to compile your own file that calls `CCL`, then you should add the following to your .bashrc:

.. code:: bash
	  
   export LD_LIBRARY_PATH=/path/to/where/ccl/is/installed/lib:$LD_LIBRARY_PATH

5. We know of one case with Mac OS where ``libtools`` had the "lock" function set to "yes" and this caused the installation to stall. However, this is very rare. If this happens, after the ``configure`` step, edit ``libtool`` to set the "lock" to "no".

C++ compatibility
====================
`CCL` library can be called from C++ code without any  additional requirements or modifications. To make sure that there are no problems you can run

.. code:: bash
	  
   make check-cpp
   ./tests/ccl_sample_run

Python installation
====================
The Python wrapper is called ``pyccl``. Generally, you can build and install the ``pyccl`` wrapper directly, without having to first compile the C version of `CCL`. The Python wrapper's build tools currently assume that your C compiler is `gcc`, and that you have a working Python 2.x or 3.x installation with ``numpy`` and ``distutils``. You will also need ``swig`` if you wish to change the `CCL` code itself, rather than just installing it as-is.

The Python wrapper installs the C libraries automatically and requires that GSL2.x and FFTW are already installed. Note that the C libraries will be installed in the same prefix as the Python files.

* To build and install the wrapper for the current user only, run

.. code:: bash
	  
   python setup.py install --user

* To build install the wrapper for all users, run

.. code:: bash

   sudo python setup.py install

* To build the wrapper in-place in the source directory (for testing), run

.. code:: bash
	  
   python setup.py build_ext --inplace

If you choose either of the first two options, the ``pyccl`` module will be installed into a sensible location in your ``PYTHONPATH``, and so should be picked up automatically by your Python interpreter. You can then simply import the module using ``import pyccl``. If you use the last option, however, you must either start your interpreter from the root `CCL` directory, or manually add the root `CCL` directory to your ``PYTHONPATH``.

On some systems, building or installing the Python wrapper fails with a message similar to

.. code:: bash

   fatal error: 'gsl/gsl_interp2d.h' file not found.

This happens when the build tools fail to find the directory containing the GSL header files, e.g. when they have been installed in a non-standard directory. To work around this problem, use the ``--include-dirs`` option when running the ``setup.py build_ext`` step above, i.e. if the GSL header files are in the directory ``/path/to/include/``, you would run

.. code:: bash
	  
   python setup.py build_ext --library-dirs=/path/to/install/lib/ --rpath=/path/to/install/lib/ --include-dirs=/path/to/include/

and then run one of the ``setup.py install`` commands listed above. (Note: As an alternative to the ``--include-dirs1`` option, you can use ``-I/path/to/include`` instead.)

You can quickly check whether ``pyccl`` has been installed correctly by running ``python -c "import pyccl"`` and checking that no errors are returned. For a more in-depth test to make sure everything is working, run

.. code:: bash
	  
   python setup.py test

This will run the embedded unit tests (may take a few minutes). Using this last method to install the Python library allows you to uninstall it simply by running

.. code:: bash

   python setup.py uninstall

For quick introduction to `CCL` in Python look at notebooks in ``**_tests/_**``.

Compiling against an external version of CLASS
==================================================
The default installation procedure for `CCL` implies automatically downloading and installing a tagged version of `CLASS`. Optionally, you can also link `CCL` against an external version of `CLASS`. This is useful if you want to use a modified version of `CLASS`, or a different or more up-to-date version of the standard `CLASS`.

To compile `CCL` with an external version of `CLASS`, you must first prepare the external copy so that it can be linked as a shared library. By default, the `CLASS` build tools create a static library. After compiling `CLASS` in the usual way (by running `make`), look for a static library file called ``***libclass.a***`` that should have been placed in the root source directory. Then, run the following command from that directory (Linux only):

.. code:: bash

   gcc -shared -o libclass.so -Wl,--whole-archive libclass.a -Wl,--no-whole-archive -lgomp

This should create a new shared library, ``***libclass.so***``, in the same directory. (N.B. The ``-lgomp`` flag has to appear at the end of the command; otherwise the linker can fail.) If you are running Mac OS X, use the following command instead:

.. code:: bash

   gcc -fpic -shared -o libclass.dylib -Wl,-all\_load libclass.a -Wl,-noall\_load

Next, change to the root `CCL` directory and run ``make clean`` if you have previously run the compilation process. Then, set the ``CLASSDIR`` environment variable to point to the directory containing ***libclass.so***:

.. code:: bash

   export CLASSDIR=/path/to/external/class

Then, run ``./configure`` and compile and install `CCL` as usual. The `CCL` build tools should take care of linking to the external version of `CLASS`.

Once compilation has finished, run ``make check`` to make sure everything is working correctly. Remember to add the external `CLASS` library directory to your system library path, using either ``export LD_LIBRARY_PATH=/path/to/external/class`` (Linux) or ``export DYLD_FALLBACK_LIBRARY_PATH=/path/to/external/class`` (Mac). The system must be able to find both the `CCL` and `CLASS` libraries; it is not enough to only add `CCL` to the library path.

Docker image installation
=========================
The Dockerfile to generate a Docker image is included in the `CCL` repository as Dockerfile. This can be used to create an image that Docker can spool up as a virtual machine, allowing you to utilize `CCL` on any infrastructure with minimal hassle. The details of Docker and the installation process can be found at `https://www.docker.com/ <https://www.docker.com/>`_. Once Docker is installed, it is a simple process to create an image! In a terminal of your choosing (with Docker running), type the command ``docker build -t ccl .`` in the `CCL` directory.

The resulting Docker image has two primary functionalities. The first is a CMD that will open Jupyter notebook tied to a port on your local machine. This can be used with the following run command: ``docker run -p 8888:8888 ccl``. You can then access the notebook in the browser of your choice at ``localhost:8888``. The second is to access the bash itself, which can be done using ``docker run -it ccl bash``.

This Dockerfile currently contains all installed C libraries and the Python wrapper. It currently uses continuumio/anaconda as the base image and supports ipython and Jupyter notebook. There should be minimal slowdown due to the virtualization.
