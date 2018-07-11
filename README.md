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
# CCL     [![Build Status](https://travis-ci.org/LSSTDESC/CCL.svg?branch=master)](https://travis-ci.org/LSSTDESC/CCL) [![Coverage Status](https://coveralls.io/repos/github/LSSTDESC/CCL/badge.svg?branch=master)](https://coveralls.io/github/LSSTDESC/CCL?branch=master)
LSST DESC Core Cosmology Library (`CCL`) provides routines to compute basic cosmological observables with validated numerical accuracy.

The library is written in C99 and all functionality is directly callable from C and C++ code.  We also provide python bindings for higher-level functions.

See also our [wiki](https://github.com/LSSTDESC/CCL/wiki).

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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/bin
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

## Development workflow

**Installing `CCL` on the system is not a good idea when doing development**, you
can compile and run all the libraries and examples directly from your local copy.
The only subtlety when not actually installing the library is that one needs to
define the environment variable `CCL_PARAM_FILE` pointing to `include/ccl_params.ini` :
```sh
export CCL_PARAM_FILE=/path/to/your/ccl/folder/include/ccl_params.ini
```
Failure to define this environment variable will result in violent segfaults !

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

# Documentation

`CCL` has basic [doxygen](http://www.stack.nl/~dimitri/doxygen/) documentation for its C routines. This can be found in the directory ***doc/html*** within the `CCL` repository by opening the ***index.html*** file in your browser. The python routines are documented in situ; you can view the documentation for a function by calling `help(function name)` from within `python`.

This document contains basic information about used structures and functions. At the end of document is provided code which implements these basic functions (also in ***examples/ccl_sample_run.c***). More information about `CCL` functions and implementation can be found in ***doc/0000-ccl_note/0000-ccl_note.pdf***.

### Cosmological parameters
Start by defining cosmological parameters defined in structure **`ccl_parameters`**. This structure (exact definition in ***include/ccl_core.h***) contains densities of matter, parameters of dark energy (`w0`, `wa`), Hubble parameters, primordial power spectra, radiation parameters, derived parameters (`sigma_8`, `Omega_1`, `z_star`) and modified growth rate.

Currently, the following families of models are supported:
* Flat ΛCDM
* wCDM and the CPL model (w0 + wa)
* Non-zero curvature (K)
* Arbitrary, user-defined modified growth function
* A single massive neutrino species or multiple equal-mass massive neutrinos (non-compatible with the user-defined modified growth function)

You can initialize this structure through function **`ccl_parameters_create`** which returns object of type **`ccl_parameters`**.
```c
ccl_parameters ccl_parameters_create(
    double Omega_c, double Omega_b, double Omega_k, double N_nu_rel, double N_nu_mass, double mnu,
    double w0, double wa, double h, double norm_pk, double n_s, int nz_mgrowth, double *zarr_mgrowth,
    double *dfarr_mgrowth, int *status);
```
where:
* `Omega_c`: cold dark matter
* `Omega_b`: baryons
* `Omega_k`: curvature
* `N_nu_rel`: Number of relativisitic species
* `N_nu_mass`: N_nu_mass
* `mnu`: deneutrino masssc
* `w0`: Dark energy eqn of state parameter
* `wa`: Dark energy eqn of state parameter, time variation
* `h`: Hubble constant in units of 100 km/s/Mpc
* `norm_pk`: the normalization of the power spectrum, either A_s or sigma_8
* `n_s`: the power-law index of the primordial power spectrum
* `nz_mgrowth`: the number of redshifts where the modified growth is provided
* `zarr_mgrowth`: the array of redshifts where the modified growth is provided
* `dfarr_mgrowth`: the modified growth function vector provided
* `status`: Status flag. 0 if there are no errors, nonzero otherwise.

For some specific cosmologies you can also use functions **`ccl_parameters_create_flat_lcdm`**, **`ccl_parameters_create_flat_wcdm`**, **`ccl_parameters_create_flat_wacdm`**, **`ccl_parameters_create_lcdm`**, which automatically set some parameters. For more information, see file ***include/ccl_core.c***.

The status flag `int status = 0` is passed around in almost every `CCL` function. Normally zero is returned while nonzero if there were some errors during a function call. For specific cases see documentation for **`ccl_error.c`**.

### The `ccl_cosmology` object
For the majority of `CCL`'s functions you need an object of type **`ccl_cosmology`**, which can be initialized by function **`ccl_cosmology_create`**
```c
ccl_cosmology * ccl_cosmology_create(ccl_parameters params, ccl_configuration config);
```

Note that the function returns a pointer. Variable `params` of type **`ccl_parameters`** contains cosmological parameters created in previous step. Structure **`ccl_configuration`** contains information about methods for computing transfer function, matter power spectrum, the impact of baryons on the matter power spectrum and mass function (for available methods see `include/ccl_config.h`). In the default configuration `default_config`, `CCL` will use the following set-up:
```c
const ccl_configuration default_config = {ccl_boltzmann_class, ccl_halofit, ccl_nobaryons, ccl_tinker};

```
After you are done working with this cosmology object, you should free its work space by **`ccl_cosmology_free`**
```c
void ccl_cosmology_free(ccl_cosmology * cosmo);
```

### Distances, Growth factor and Density parameter functions
With defined cosmology we can now compute distances, growth factor (and rate), sigma_8 or density parameters. For comoving radial distance you can call function **`ccl_comoving_radial_distance`**
```c
double ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a, int* status);
```
which returns distance to scale factor `a` in units of Mpc. For luminosity distance call function **`ccl_luminosity_distance`**
```c
double ccl_luminosity_distance(ccl_cosmology * cosmo, double a, int * status);
```
which also returns distance in units of Mpc. For growth factor (normalized to 1 at `z` = 0) at sale factor `a` call **`ccl_growth_factor`**
```c
double ccl_growth_factor(ccl_cosmology * cosmo, double a, int * status);
```
For evaluating density parameters (e.g. matter, dark energy or radiation) call function **`ccl_omega_x`**
```c
double ccl_omega_x(ccl_cosmology * cosmo, double a, ccl_omega_x_label label, int* status);
```
where **`ccl_omega_x_label`** `label` defines species type: `'matter' (0)`, `'dark_energy'(1)`, `'radiation'(2)`, and `'curvature'(3)`.

For more routines to compute distances, growth rates and density parameters (e.g. at multiple times at once) see file ***include/ccl_background.h***

###  Matter power spectra and sigma_8
For given cosmology we can compute linear and non-linear matter power spectra using functions **`ccl_linear_matter_power`** and **`ccl_nonlin_matter_power`**
```c
double ccl_linear_matter_power(ccl_cosmology * cosmo, double k, double a,int * status);
double ccl_nonlin_matter_power(ccl_cosmology * cosmo, double k, double a,int * status);
```

It is possible to incorporate the impact of baryonic processes on the total matter power spectrum via the **`baryons_power_spectrum`** flag is set to **`ccl_bcm`**. Please see the CCL note for details on the implementation.
Sigma_8 can be calculated by function **`ccl_sigma8`**, or more generally by function **`ccl_sigmaR`**, which computes the variance of the density field smoothed by spherical top-hat window function on a comoving distance `R` (in Mpc).

```c
double ccl_sigmaR(ccl_cosmology *cosmo, double R, int * status);
double ccl_sigma8(ccl_cosmology *cosmo, int * status);
```
These and other functions for different matter power spectra can be found in file ***include/ccl_power.h***.

### Angular power spectra
`CCL` can compute angular power spectra for three tracer types: galaxy number counts, galaxy weak lensing and CMB lensing. Tracer parameters are defined in structure **`CCL_ClTracer`**. In general, you can create this object with function **`ccl_cl_tracer`**
````c
CCL_ClTracer *ccl_cl_tracer(ccl_cosmology *cosmo,int tracer_type,
				int has_rsd,int has_magnification,int has_intrinsic_alignment,
				int nz_n,double *z_n,double *n,
				int nz_b,double *z_b,double *b,
				int nz_s,double *z_s,double *s,
				int nz_ba,double *z_ba,double *ba,
				int nz_rf,double *z_rf,double *rf,
				double z_source, int * status);
````
Exact definition of these parameters are described in file ***include/ccl_cls.h***. Usually you can use simplified versions of this function, namely **`ccl_cl_tracer_number_counts`, `ccl_cl_tracer_number_counts_simple`, `ccl_cl_tracer_lensing`, `ccl_cl_tracer_lensing_simple`** or **`ccl_cl_tracer_cmblens`**. Two most simplified versions (one for number counts and one for shear) take parameters:
````c
CCL_ClTracer *ccl_cl_tracer_number_counts_simple(ccl_cosmology *cosmo,
						     int nz_n,double *z_n,double *n,
                                                     int nz_b,double *z_b,double *b, int * status);
CCL_ClTracer *ccl_cl_tracer_lensing_simple(ccl_cosmology *cosmo,
					       int nz_n,double *z_n,double *n, int * status);
````
where `nz_n` is dimension (number of bins) of arrays `z_n` and `n`. `z_n` and `n` are arrays for the number count of objects per redshift interval (arbitrary normalization - renormalized inside). `nz_b`, `z_b` and `b` are the same for the clustering bias.

Before computing the angular power spectrum, users must define a workspace structure that contains the relevant parameters for the computation:
````c
CCL_ClWorkspace *ccl_cl_workspace_default(int lmax,int l_limber,int non_limber_method,
					  double l_logstep,int l_linstep,
					  double dchi,double dlk,double zmin,int *status)
````
where `lmax` sets the maximum multipole, `l_limber` the limit multipole from which the Limber approximation is used (`l_limber=-1` means that the Liber approximation is never used). The `non_limber_method` variable can be set to `CCL_NONLIMBER_METHOD_NATIVE` or `CCL_NONLIMBER_METHOD_ANGPOW` to choose the method to compute the non-Limber part of the angular power spectrum (either the native `CCL` code or the [`Angpow` library](https://github.com/LSSTDESC/CCL/blob/non_limber_speedup/README.md#installing-angpow)). Then `l_linstep` sets the maximum multipole until which the angular power spectrum is computed at each multipole, and `l_logstep` the logarithmic stepping to use above `l_linstep` (then the power spectrum is interpolated at each multipole). `dchi` sets the interval in comoving distance to use for the native non-Limber computation and `dlk`the logarithmic stepping for the Fourier k-integration (`Angpow` is not concerned by these two parameters). A simplified workspace is provided for computations that use only the Limber approximation at each multipole:
````c
CCL_ClWorkspace *ccl_cl_workspace_default_limber(int lmax,double l_logstep,int l_linstep,
						 double dlk,int *status)
````

With initialized tracers and workspace you can compute limber power spectrum with **`ccl_angular_cls`**
````c
double ccl_angular_cls(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,CCL_ClTracer *clt2,
						int nl_out,int *l_out,double *cl_out,int * status);
````
with `l_out` and `cl_out` arrays of size `nl_out` that contains the multipoles and the angular power spectrum.

After you are done working with tracers, you should free its work space by **`ccl_cl_tracer_free`** and **`ccl_cl_workspace_free`**
````c
void ccl_cl_tracer_free(CCL_ClTracer *clt);
void ccl_cl_workspace_free(CCL_ClWorkspace *w);
````

Note that for the moment `Angpow` can not handle the magnification lensing term for the galaxy number count tracers, and has not been tested for the weak lensing tracer. This limitations will be removed in the near future.

### Halo mass function
The halo mass function *dN/dM* can be obtained by function **`ccl_massfunc`**
````c
double ccl_massfunc(ccl_cosmology * cosmo, double smooth_mass, double a, double odelta, int * status);
````
where `smooth_mass` is mass smoothing scale (in units of *M_sun*) and `odelta` is choice of Delta. For more details (or other functions like **`sigma_M`**) see ***include/ccl_massfunc.h*** and ***src/mass_func.c***.

### LSST Specifications
`CCL` includes LSST specifications for the expected galaxy distributions of the full galaxy clustering sample and the lensing source galaxy sample. Start by defining a flexible photometric redshift model given by function
````c
double (* your_pz_func)(double z_ph, double z_s, void *param, int * status);
````
which returns the likelihood of measuring a particular photometric redshift `z_ph` given a spectroscopic redshift `z_s`, with a pointer to additional arguments `param` and a status flag. Then you call function **`ccl_specs_create_photoz_info`**
````c
user_pz_info* ccl_specs_create_photoz_info(void * user_params,
                                           double(*user_pz_func)(double, double, void*, int*));
````
which creates a strcture **`user_pz_info`** which holds information needed to compute *dN/dz*
````c
typedef struct {
  double (* your_pz_func)(double, double, void *, int*);
  void *  your_pz_params;
} user_pz_info;
````
The expected *dN/dz* for lensing or clustering galaxies with given binning can be obtained by function **`ccl_specs_dNdz_tomog`**
````c
void ccl_specs_dNdz_tomog(double z, int dNdz_type, double bin_zmin, double bin_zmax,
                          user_pz_info * user_info,  double *tomoout, int *status);
````
Result is returned in `tomoout`. Allowed types of `dNdz_type` (currently one for clustering and three for lensing - fiducial, optimistic, and conservative - cases are considered) and other information and functions like bias clustering or sigma_z are specified in file ***include/ccl_lsst_specs.h***

After you are done working with photo_z, you should free its work space by **`ccl_specs_free_photoz_info`**
````c
void ccl_specs_free_photoz_info(user_pz_info *my_photoz_info);
````

## Example code
This code can also be found in ***examples/ccl_sample_run.c*** You can run the following example code. For this you will need to compile with the following command:
````sh
gcc -Wall -Wpedantic -g -I/path/to/install/include -std=gnu99 -fPIC examples/ccl_sample_run.c \
-o examples/ccl_sample_run -L/path/to/install/lib -L/usr/local/lib -lgsl -lgslcblas -lm -lccl
````
where `/path/to/install/` is the path to the location where the library has been installed.

```c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"
#include "ccl_lsst_specs.h"

#define OC 0.25
#define OB 0.05
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define NORMPS 0.80
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 512
#define PS 0.1
#define NREL 3.046
#define NMAS 0
#define MNU 0.0



// The user defines a structure of parameters
// to the user-defined function for the photo-z probability
struct user_func_params
{
  double (* sigma_z) (double);
};

// The user defines a function of the form double function ( z_ph, z_s, void * user_pz_params)
// where user_pz_params is a pointer to the parameters of the user-defined function.
// This returns the probabilty of obtaining a given photo-z given a particular spec_z.
double user_pz_probability(double z_ph, double z_s, void * user_par, int * status)
{
  double sigma_z = ((struct user_func_params *) user_par)->sigma_z(z_s);
  return exp(- (z_ph-z_s)*(z_ph-z_s) / (2.*sigma_z*sigma_z)) / (pow(2.*M_PI,0.5)*sigma_z);
}

int main(int argc,char **argv)
{
  //status flag
  int status =0;

  // Initialize cosmological parameters
  ccl_configuration config=default_config;
  config.transfer_function_method=ccl_boltzmann_class;
  ccl_parameters params = ccl_parameters_create(OC, OB, OK, NREL, NMAS, MNU, W0, WA, HH,
  		 	  			NORMPS, NS,-1,-1,-1,-1,NULL,NULL, &status);
  //printf("in sample run w0=%1.12f, wa=%1.12f\n", W0, WA);

  // Initialize cosmology object given cosmo params
  ccl_cosmology *cosmo=ccl_cosmology_create(params,config);

  // Compute radial distances (see include/ccl_background.h for more routines)
  printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc\n",
	 ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status));
  printf("Luminosity distance to z = %.3lf is chi = %.3lf Mpc\n",
	 ZD,ccl_luminosity_distance(cosmo,1./(1+ZD), &status));
  printf("Distance modulus to z = %.3lf is mu = %.3lf Mpc\n",
	 ZD,ccl_distance_modulus(cosmo,1./(1+ZD), &status));


  //Consistency check
  printf("Scale factor is a=%.3lf \n",1./(1+ZD));
  printf("Consistency: Scale factor at chi=%.3lf Mpc is a=%.3lf\n",
	 ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status),
	 ccl_scale_factor_of_chi(cosmo,ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status), &status));

  // Compute growth factor and growth rate (see include/ccl_background.h for more routines)
  printf("Growth factor and growth rate at z = %.3lf are D = %.3lf and f = %.3lf\n",
	 ZD, ccl_growth_factor(cosmo,1./(1+ZD), &status),ccl_growth_rate(cosmo,1./(1+ZD), &status));

  // Compute Omega_m, Omega_L and Omega_r at different times
  printf("z\tOmega_m\tOmega_L\tOmega_r\n");
  double Om, OL, Or;
  for (int z=10000;z!=0;z/=3){
    Om = ccl_omega_x(cosmo, 1./(z+1), ccl_omega_m_label, &status);
    OL = ccl_omega_x(cosmo, 1./(z+1), ccl_omega_l_label, &status);
    Or = ccl_omega_x(cosmo, 1./(z+1), ccl_omega_g_label, &status);
    printf("%i\t%.3f\t%.3f\t%.3f\n", z, Om, OL, Or);
  }
  Om = ccl_omega_x(cosmo, 1., ccl_omega_m_label, &status);
  OL = ccl_omega_x(cosmo, 1., ccl_omega_l_label, &status);
  Or = ccl_omega_x(cosmo, 1., ccl_omega_g_label, &status);
  printf("%i\t%.3f\t%.3f\t%.3f\n", 0, Om, OL, Or);

  // Compute sigma_8
  printf("Initializing power spectrum...\n");
  printf("sigma_8 = %.3lf\n\n", ccl_sigma8(cosmo, &status));

  //Create tracers for angular power spectra
  double z_arr_gc[NZ],z_arr_sh[NZ],nz_arr_gc[NZ],nz_arr_sh[NZ],bz_arr[NZ];
  for(int i=0;i<NZ;i++) {
    z_arr_gc[i]=Z0_GC-5*SZ_GC+10*SZ_GC*(i+0.5)/NZ;
    nz_arr_gc[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2));
    bz_arr[i]=1+z_arr_gc[i];
    z_arr_sh[i]=Z0_SH-5*SZ_SH+10*SZ_SH*(i+0.5)/NZ;
    nz_arr_sh[i]=exp(-0.5*pow((z_arr_sh[i]-Z0_SH)/SZ_SH,2));
  }

  //CMB lensing tracer
  CCL_ClTracer *ct_cl=ccl_cl_tracer_cmblens_new(cosmo,1100.,&status);

  //Galaxy clustering tracer
  CCL_ClTracer *ct_gc=ccl_cl_tracer_number_counts_simple_new(cosmo,NZ,
                                z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr, &status);

  //Cosmic shear tracer
  CCL_ClTracer *ct_wl=ccl_cl_tracer_lensing_simple_new(cosmo,NZ,z_arr_sh,nz_arr_sh, &status);
  printf("ell C_ell(c,c) C_ell(c,g) C_ell(c,s) C_ell(g,g) C_ell(g,s) C_ell(s,s) \n");
  for(int l=2;l<=NL;l*=2) {
    double cl_cc=ccl_angular_cl(cosmo,l,ct_cl,ct_cl, &status); //CMBLensing-CMBLensing
    double cl_cg=ccl_angular_cl(cosmo,l,ct_cl,ct_gc, &status); //CMBLensing-Clustering
    double cl_cs=ccl_angular_cl(cosmo,l,ct_wl,ct_cl, &status); //CMBLensing-Galaxy lensing
    double cl_gg=ccl_angular_cl(cosmo,l,ct_gc,ct_gc, &status); //Galaxy-galaxy
    double cl_gs=ccl_angular_cl(cosmo,l,ct_gc,ct_wl, &status); //Galaxy-lensing
    double cl_ss=ccl_angular_cl(cosmo,l,ct_wl,ct_wl, &status); //Lensing-lensing
    printf("%d %.3lE %.3lE %.3lE %.3lE %.3lE %.3lE\n",l,cl_cc,cl_cg,cl_cs,cl_gg,cl_gs,cl_ss);
  }
  printf("\n");

  //Free up tracers
  ccl_cl_tracer_free(ct_gc);
  ccl_cl_tracer_free(ct_cl);
  ccl_cl_tracer_free(ct_wl);

  //Halo mass function
  printf("M\tdN/dlog10M(z = 0, 0.5, 1))\n");
  for(int logM=9;logM<=15;logM+=1) {
    printf("%.1e\t",pow(10,logM));
    for(double z=0; z<=1; z+=0.5) {
      printf("%e\t", ccl_massfunc(cosmo, pow(10,logM),1.0/(1.0+z), 200., &status));
    }
    printf("\n");
  }
  printf("\n");

  //Halo bias
  printf("Halo bias: z, M, b1(M,z)\n");
  for(int logM=9;logM<=15;logM+=1) {
    for(double z=0; z<=1; z+=0.5) {
      printf("%.1e %.1e %.2e\n",1.0/(1.0+z),pow(10,logM),ccl_halo_bias(cosmo,pow(10,logM),1.0/(1.0+z), 200., &status));
    }
  }
  printf("\n");

  // LSST Specification
  // The user declares and sets an instance of parameters to their photo_z function:
  struct user_func_params my_params_example;
  my_params_example.sigma_z = ccl_specs_sigmaz_sources;

  // Declare a variable of the type of user_pz_info to hold the struct to be created.
  user_pz_info * pz_info_example;

  // Create the struct to hold the user information about photo_z's.
  pz_info_example = ccl_specs_create_photoz_info(&my_params_example, &user_pz_probability);

  // Alternatively, we could have used the built-in Gaussian photo-z pdf,
  // which assumes sigma_z = sigma_z0 * (1 + z) (not used in what follows).
  double sigma_z0 = 0.05;
  user_pz_info *pz_info_gaussian;
  pz_info_gaussian = ccl_specs_create_gaussian_photoz_info(sigma_z0);

  double z_test;
  double dNdz_tomo;
  int z;
  FILE * output;

  //Try splitting dNdz (lensing) into 5 redshift bins
  double tmp1,tmp2,tmp3,tmp4,tmp5;
  printf("Trying splitting dNdz (lensing) into 5 redshift bins. "
         "Output written into file tests/specs_example_tomo_lens.out\n");
  output = fopen("./tests/specs_example_tomo_lens.out", "w");

  if(!output) {
    fprintf(stderr, "Could not write to 'tests' subdirectory"
                    " - please run this program from the main CCL directory\n");
    exit(1);
  }
  status = 0;
  for (z=0; z<100; z=z+1) {
    z_test = 0.035*z;
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,6., pz_info_example,&dNdz_tomo,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,0.6, pz_info_example,&tmp1,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.6,1.2, pz_info_example,&tmp2,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.2,1.8, pz_info_example,&tmp3,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.8,2.4, pz_info_example,&tmp4,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 2.4,3.0, pz_info_example,&tmp5,&status);
    fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
  }

  fclose(output);

  //Try splitting dNdz (clustering) into 5 redshift bins
  printf("Trying splitting dNdz (clustering) into 5 redshift bins. "
         "Output written into file tests/specs_example_tomo_clu.out\n");
  output = fopen("./tests/specs_example_tomo_clu.out", "w");     
  for (z=0; z<100; z=z+1) {
    z_test = 0.035*z;
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.,6., pz_info_example,&dNdz_tomo,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.,0.6, pz_info_example,&tmp1,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.6,1.2, pz_info_example,&tmp2,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,1.2,1.8, pz_info_example,&tmp3,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,1.8,2.4, pz_info_example,&tmp4,&status);
    ccl_specs_dNdz_tomog(z_test, DNDZ_NC,2.4,3.0, pz_info_example,&tmp5,&status);
    fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
  }
  printf("ccl_sample_run completed, status = %d\n",status);
  fclose(output);

  //Free up photo-z info
  ccl_specs_free_photoz_info(pz_info_example);

  //Always clean up!!
  ccl_cosmology_free(cosmo);

  return 0;
}
````

## Python wrapper
A Python wrapper for `CCL` is provided through a module called `pyccl`. The whole `CCL` interface can be accessed through regular Python functions and classes, with all of the computation happening in the background through the C code. The functions all support `numpy` arrays as inputs and outputs, with any loops being performed in the C code for speed.

The Python module has essentially the same functions as the C library, just presented in a more standard Python-like way. You can inspect the available functions and their arguments by using the built-in Python **`help()`** function, as with any Python module.

Below is a simple example Python script that creates a new **Cosmology** object, and then uses it to calculate the angular power spectra for a simple lensing cross-correlation. It should take a few seconds on a typical laptop.

````python
import pyccl as ccl
import numpy as np

# Create new Cosmology object with a given set of parameters. This keeps track
# of previously-computed cosmological functions
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)

# Define a simple binned galaxy number density curve as a function of redshift
z_n = np.linspace(0., 1., 200)
n = np.ones(z_n.shape)

# Create objects to represent tracers of the weak lensing signal with this
# number density (with has_intrinsic_alignment=False)
lens1 = ccl.ClTracerLensing(cosmo, False, n=(z_n, n))
lens2 = ccl.ClTracerLensing(cosmo, False, n=(z_n, n))

# Calculate the angular cross-spectrum of the two tracers as a function of ell
ell = np.arange(2, 10)
cls = ccl.angular_cl(cosmo, lens1, lens2, ell)
print cls
````


# License, Credits, Feedback etc
The `CCL` is still under development and should be considered research in progress. You are welcome to re-use the code, which is open source and available under terms consistent with [BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) licensing. If you make use of any of the ideas or software in this package in your own research, please cite them as "(LSST DESC, in preparation)" and provide a link to this repository: https://github.com/LSSTDESC/CCL. For free use of the `CLASS` library, the `CLASS` developers require that the `CLASS` paper be cited: CLASS II: Approximation schemes, D. Blas, J. Lesgourgues, T. Tram, arXiv:1104.2933, JCAP 1107 (2011) 034. The `CLASS` repository can be found in http://class-code.net. If you have comments, questions, or feedback, please [write us an issue](https://github.com/LSSTDESC/CCL/issues). Finally, CCL uses code from the [FFTLog](http://casa.colorado.edu/~ajsh/FFTLog/) package.  We have obtained permission from the FFTLog author to include modified versions of his source code.
