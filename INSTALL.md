<!---
STYLE CONVENTION USED   
    bolt italic:
        ***file***"
    code:
       `program` or `library``
       `commands` or `paths`
       `variable`
    bold code:
        **`function`**
        **`type`** or **`structure`**
-->
# Installation

## TLDR

`CCL` is available as a Python package through PyPi. To install, simply run:
```
$ pip install pyccl
```
This should work as long as `CMake` is installed on your system (if it doesn't follow the detailed instructions below).
Once `CCL` is installed, take it for a spin by following some example notebooks [here](examples).

`CCL` comes in two forms, a C library and a Python module. These components can
be installed independently of each other, instructions are provided below.

## Dependencies and requirements

`CCL` requires the following software and libraries:
  * [GSL](https://www.gnu.org/software/gsl/) version 2.1 or above
  * [FFTW3](http://www.fftw.org/) version 3.1 or above
  * [CLASS](http://class-code.net/) version 2.6.3 or above
  * [Angpow](https://gitlab.in2p3.fr/campagne/AngPow)
  * FFTlog([here](http://casa.colorado.edu/~ajsh/FFTLog/) and [here](https://github.com/slosar/FFTLog))is provided within `CCL`, with minor modifications.

In addition, the build system for `CCL` relies on the following software:
  * [CMake](https://cmake.org/) version 3.2 or above
  * [SWIG](http://www.swig.org/)

**`CMake` is the only requirement that needs to be manually installed**:
  * On Ubuntu:
  ```sh
  $ sudo apt-get install cmake
  ```
  * On MacOs X:
    * Install using a `DMG` package from this [download page](https://cmake.org/download/)
    * Install using a package manager ([brew](https://brew.sh/), [MacPorts](https://www.macports.org/), [Fink](http://www.finkproject.org/)). For instance with brew:
    ```sh
    $ brew install cmake
    ```

It is preferable to install `GSL` and `FFTW` on your system before building `CCL`
but only necessary if you want to properly install the C library, otherwise
`CMake` will automatically download and build the missing requirements in order
to compile `CCL`.

To install all the dependencies at once, and avoid having `CMake` recompiling them, for instance on Ubuntu:
  ```sh
  $ sudo apt-get install cmake swig libgsl-dev libfftw3-dev
  ```

## Compile and install the CCL C library

To download the latest version of `CCL`:
```sh
$ git clone https://github.com/LSSTDESC/CCL.git
$ cd CCL
```
or download and extract the latest stable release from [here](https://github.com/LSSTDESC/CCL/releases). Then, from the base `CCL` directory run:
```sh
$ mkdir build && cd build
$ cmake ..
```
This will run the configuration script, try to detect the required dependencies
on your machine and generate a Makefile. Once CMake has been configured, to build
and install the library simply run for the `build` directory:
```sh
$ make
$ make install
```
Often admin privileges will be needed to install the library. If you have those just type:
```sh
$ sudo make install
```

**Note:** This is the default install procedure, but depending on your system
you might want to customize the intall process. Here are a few common configuration
options:
  - *C compiler*: In case you have several C compilers on your machine, you will probably
need to specify which one to use to `CMake` by setting the environment `CC` like
so, **before** running `CMake`:
```sh
$ export CC=gcc
```
  - *Install directory*: By default, `CMake` will try to install `CCL` in `/usr/local`, if you would like
to instead install CCL in a user-defined directory (for instance if you don't have
 admin privileges), you can specify it to `CMake` by running instead the following command:
```sh
$ cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..
```
This will instruct CMake to install `CCL` in the following folders: `/path/to/install/include`,`/path/to/install/share` ,`/path/to/install/lib`.

Depending on where you install `CCL` your might need to add the installation path
to your to your `PATH` and `LD_LIBRARY_PATH` environment variables. In the default
case, it will look like:
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export PATH=$PATH:/usr/local/bin
```

To make sure that everything is working properly, you can run all unit tests after installation by running from the root `CCL` directory:
```sh
$ check_ccl
```
Assuming that the tests pass, you have successfully installed `CCL`!


If you ever need to uninstall `CCL`, run the following from the `build` directory:
```sh
$ make uninstall
```
You may need to prepend a `sudo` if you installed `CCL` in a protected folder.

<!---
### C++ compatibility
`CCL` library can be called from C++ code without any additional requirements or modifications.
To make sure that there are no problems you can run:
````sh
make check-cpp
./examples/ccl_sample_run
````
TODO: add this to cmake
--->
## Install the pyccl Python module

`CCL` also comes with a Python wrapper, called `pyccl`, which can be built and
installed regardless of whether you install the C library. For convenience, we
provide a PyPi hosted package which can be installed simply by running:
```sh
$ pip install pyccl # append --user for single user install
```
This only assumes that `CMake` is available on your system, you don't need to
download the source yourself.

You can also build and install `pyccl` from the `CCL` source, again **without necessarily
installing the C library**. Download the latest version of `CCL`:
```sh
$ git clone https://github.com/LSSTDESC/CCL.git
$ cd CCL
```

And from the root `CCL` folder, simply run:
````sh
$ python setup.py install # append --user for single user install
````

The `pyccl` module will be installed into a sensible location in your `PYTHONPATH`,
and so should be picked up automatically by your Python interpreter. You can then simply
import the module using `import pyccl`.

You can quickly check whether `pyccl` has been installed correctly by running
`python -c "import pyccl"` and checking that no errors are returned.

For a more in-depth test to make sure everything is working, run from the root
`CCL` directory:
````sh
$ python setup.py test
````
This will run the embedded unit tests (may take a few minutes).

Whatever the install method, you can always uninstall the pyton wrapper by running:
````sh
$ pip uninstall pyccl
````

For quick introduction to `CCL` in Python look at notebooks in **_tests/_**.

## Known installation issues
1. In case you have several C compilers on your system, `CMake` may not default
to the one you want to use. You can specify which C compiler will be used to compile
`CCL` by setting the `CC` environment variable **before** calling any `cmake` or `python setup.py` commands:
```sh
export CC=gcc
```
2. If upon running the C tests you get an error from CLASS saying it cannot find the file `sBBN_2017.dat`, your system is not finding the local copy of CLASS. To solve this, do
```sh
export CLASS_PARAM_DIR=your_ccl_path/CCL/build/extern/share/class/
```
or add this to your `.bashrc`.

## Development workflow

**Installing `CCL` on the system is not a good idea when doing development**, you
can compile and run all the libraries and examples directly from your local copy.

The only subtlety when not actually installing the library is that one needs to
define the environment variable `CCL_PARAM_FILE` pointing to `include/ccl_params.ini` :

```sh
export CCL_PARAM_FILE=/path/to/your/ccl/folder/include/ccl_params.ini
```

Failure to define this environment variable will result an exception.

### Working on the C library
Here are a few common steps when working on the C library:

  - Cloning a local copy and CCL and compiling it:
  ```sh
  $ git clone https://github.com/LSSTDESC/CCL
  $ mkdir -p CCL/build && cd CCL/build
  $ cmake ..
  $ make
  ```

  - Updating local copy from master, recompiling what needs recompiling, and
  running the test suite:
  ```sh
  $ git pull      # From root folder
  $ make -Cbuild  # The -C option allows you to run make from a different directory
  $ export CCL_PARAM_FILE=$PWD/include/ccl_params.ini # set the parameter file
  $ build/check_ccl
  ```

  - Compiling (or recompiling) an example in the `CCL/examples` folder:
  ```sh
  $ cd examples  # From root folder
  $ make -C../build ccl_sample_pkemu
  $ ./ccl_sample_pkemu
  ```

  - Reconfiguring from scratch (in case something goes terribly wrong):
  ```sh
  $ cd build
  $ rm -rf *
  $ cmake ..
  $ make
  ```

  - Building CCL in Debug mode. This will disable optimizations and allow you to
  use a debugger:
  ```sh
  $ mkdir -p CCL/debug && cd CCL/debug
  $ cmake -DCMAKE_BUILD_TYPE=Debug ..
  $ make
  ```

### Working on the Python library
Here are a few common steps when working on the Python module:

  - Building the python module:
  ```sh
  $ python setup.py build
  ```
  After that, you can start your interpreter from the root `CCL` folder and import
  pyccl

  - Running the tests after a modification of the C library:
  ```sh
  $ python setup.py build
  $ python setup.py test
  ```


## Compiling against an external version of CLASS

The default installation procedure for `CCL` implies automatically downloading and installing a tagged version of `CLASS`. Optionally, you can also link `CCL` against a different version of `CLASS`. This is useful if you want to use a modified version of `CLASS`, or a different or more up-to-date version of the standard `CLASS`.

To compile `CCL` with an external version of `CLASS`, just run the following `CMake`
command at the first configuration step of the install (from the `build` directory, make sure it is empty to get a clean configuration):
```sh
$ cmake -DEXTERNAL_CLASS_PATH=/path/to/class ..
```
the rest of the build process should be the same.

## Docker image installation

The Dockerfile to generate a Docker image is included in the `CCL` repository as Dockerfile. This can be used to create an image that Docker can spool up as a virtual machine, allowing you to utilize `CCL` on any infrastructure with minimal hassle. The details of Docker and the installation process can be found at [https://www.docker.com/](https://www.docker.com/). Once Docker is installed, it is a simple process to create an image! In a terminal of your choosing (with Docker running), type the command `docker build -t ccl .` in the `CCL` directory.

The resulting Docker image has two primary functionalities. The first is a CMD that will open Jupyter notebook tied to a port on your local machine. This can be used with the following run command: `docker run -p 8888:8888 ccl`. You can then access the notebook in the browser of your choice at `localhost:8888`. The second is to access the bash itself, which can be done using `docker run -it ccl bash`.

This Dockerfile currently contains all installed C libraries and the Python wrapper. It currently uses continuumio/anaconda as the base image and supports ipython and Jupyter notebook. There should be minimal slowdown due to the virtualization.

