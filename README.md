# CCL     [![Build Status](https://travis-ci.org/LSSTDESC/CCL.svg?branch=master)](https://travis-ci.org/LSSTDESC/CCL)
LSST DESC Core Cosmology Library: cosmology routines with validated numerical accuracy.

The library is written in C99 and all functionality is directly callable from C and C++ code.  We also provide python bindings for higher-level functions.

See also our wiki: https://github.com/LSSTDESC/CCL/wiki

# Installation
In order to compile CCL you need GSL. You can get GSL here: https://www.gnu.org/software/gsl/. Note that CCL uses version 2+ of GSL (which is not yet standard in all systems). It also needs the FFTW libraries that can be found here: http://www.fftw.org/

# C-only installation
To install CCL, from the base directory (the one where this file is located) run:
```sh
./configure
make
make install
```
Often admin privileges will be needed to install the library. If you have those just type:
```sh
sudo make install
```
If you don't have admin privileges, you can still install the library by running
```sh
./configure --prefix=/path/to/install
make
make install
```
where `/path/to/install` is the absolute path to the directory where you want the library to be installed. If non-existing, this will create two directories, `/path/to/install/include` and `/path/to/install/lib`, and the library and header files will be installed there. Note that, in order to use `CCL` with your own scripts you'll have to add `/path/to/install/lib` to your LD_LIBRARY_PATH.

All unit tests can be run after installation by running
```sh
make check
```
## Known installation issues
1. If you move or delete the source directory after installing CCL, some functions may fail. The source directory contains files needed by *CLASS* (which is contained within CCL) at run-time.

## Python installation
The Python wrapper is called *pyccl*. Before you can build it, you must have compiled and installed the C version of CCL, as *pyccl* will be dynamically linked to it. The Python wrapper's build tools currently assume that your C compiler is *gcc*, and that you have a working Python 2.x installation with *numpy* and *distutils* with *swig*.

The Python wrapper installs the C libraries automatically and requires that GSL2.x and FFTW are already installed. The C libraries will be installed in `/PATH/TO/PREFIX/lib` and `/PATH/TO/PREFIX/include`.

* To build and install the wrapper for the current user only, run
````sh
python setup.py install --user
````
* To build install the wrapper for all users, run
````sh
sudo python setup.py install
````
* To build the wrapper in-place in the source directory (for testing), run
````sh
python setup.py build_ext --inplace
````
If you choose either of the first two options, the *pyccl* module will be installed into a sensible location in your *PYTHONPATH*, and so should be picked up automatically by your Python interpreter. You can then simply import the module using `import pyccl`. If you use the last option, however, you must either start your interpreter from the root CCL directory, or manually add the root CCL directory to your *PYTHONPATH*.

On some systems, building or installing the Python wrapper fails with a message similar to
````sh
fatal error: 'gsl/gsl_interp2d.h' file not found.
````
This happens when the build tools fail to find the directory containing the GSL header files, e.g. when they have been installed in a non-standard directory. To work around this problem, use the `--include-dirs` option when running the `setup.py build_ext` step above, i.e. if the GSL header files are in the directory `/path/to/include/`, you would run
````sh
python setup.py build_ext --library-dirs=/path/to/install/lib/ --rpath=/path/to/install/lib/ --include-dirs=/path/to/include/
````
and then run one of the `setup.py install` commands listed above. (Note: As an alternative to the `--include-dirs` option, you can use `-I/path/to/include` instead.)

You can quickly check whether *pyccl* has been installed correctly by running `python -c "import pyccl"` and checking that no errors are returned. For a more in-depth test to make sure everything is working, change to the `tests/` sub-directory and run `python run_tests.py`. These tests will take a few minutes.

## Compiling against an external version of CLASS

*CCL* has a built-in version of *CLASS* that is used to calculate power spectra and other cosmological functions. This is compiled by default. Optionally, you can also link *CCL* against an external version of *CLASS*. This is useful if you want to use a modified version of *CLASS*, or a different or more up-to-date version of the standard *CLASS*.

To compile *CCL* with an external version of *CLASS*, you must first prepare the external copy so that it can be linked as a shared library. By default, the *CLASS* build tools create a static library. After compiling *CLASS* in the usual way (by running `make`), look for a static library file called `libclass.a` that should have been placed in the root source directory. Then, run the following command from that directory (Linux only):
````sh
gcc -shared -o libclass.so -Wl,--whole-archive libclass.a \
                           -Wl,--no-whole-archive -lgomp
````
This should create a new shared library, `libclass.so`, in the same directory. (N.B. The `-lgomp` flag has to appear at the end of the command; otherwise the linker can fail.) If you are running Mac OS X, use the following command instead:
````sh
gcc -fpic -shared -o libclass.dylib -Wl,-all\_load libclass.a -Wl,-noall\_load
````

Next, change to the root *CCL* directory and run `make clean` if you have previously run the compilation process. Then, set the `CLASSDIR` environment variable to point to the directory containing `libclass.so`:
````sh
export CLASSDIR=/path/to/external/class
````
Then, run `./configure` and compile and install *CCL* as usual. The *CCL* build tools should take care of linking to the external version of *CLASS*.

Once compilation has finished, run `make check` to make sure everything is working correctly. Remember to add the external *CLASS* library directory to your system library path, using either `export LD_LIBRARY_PATH=/path/to/external/class` (Linux) or `export DYLD_FALLBACK_LIBRARY_PATH=/path/to/external/class` (Mac). The system must be able to find both the *CCL* and *CLASS* libraries; it is not enough to only add *CCL* to the library path.


## Docker image installation

The Dockerfile to generate a Docker image is included in the CCL repository as Dockerfile. This can be used to create an image that Docker can spool up as a virtual machine, allowing you to utilize CCL on any infrastructure with minimal hassle. The details of Docker and the installation process can be found at [https://www.docker.com/](https://www.docker.com/). Once Docker is installed, it is a simple process to create an image! In a terminal of your choosing (with Docker running), type the command `docker build -t ccl .` in the CCL directory.

The resulting Docker image has two primary functionalities. The first is a CMD that will open Jupyter notebook tied to a port on your local machine. This can be used with the following run command: `docker run -p 8888:8888 ccl`. You can then access the notebook in the browser of your choice at `localhost:8888`. The second is to access the bash itself, which can be done using `docker run -it ccl bash`.

This Dockerfile currently contains all installed C libraries and the Python wrapper. It currently uses continuumio/anaconda as the base image and supports ipython and Jupyter notebook. There should be minimal slowdown due to the virtualization.


# Documentation

This document contains basic information about used structures and functions. At the end of document is provided code which implements these basic functions (also in *tests/ccl_sample_run.c*). More information about CCL functions and implementation can be found in *doc/0000-ccl_note/0000-ccl_note.pdf*.

### Cosmological parameters
Start by defining cosmological parameters defined in structure **`ccl_parameters`**. This structure (exact definition in `include/ccl_core.h`) contains densities of matter, parameters of dark energy (`w0`, `wa`), Hubble parameters, primordial power spectra, radiation parameters, derived parameters (`sigma_8`, `Omega_1`, `z_star`) and modified growth rate.

You can initialize this structure through function **`ccl_parameters_create`** which returns object of type **`ccl_parameters`**.
```c
ccl_parameters ccl_parameters_create(
	double Omega_c, double Omega_b, double Omega_k, double Omega_n, double w0, double wa, double h,
	double A_s, double n_s, int nz_mgrowth, double *zarr_mgrowth, double *dfarr_mgrowth
);
```
where:
* `Omega_c`: cold dark matter
* `Omega_b`: baryons
* `Omega_m`: matter
* `Omega_n`: neutrinos
* `Omega_k`: curvature
* little `omega_x` means "Omega_x h^2"
* `w0`: Dark energy eqn of state parameter
* `wa`: Dark energy eqn of state parameter, time variation
* `H0`: Hubble's constant in km/s/Mpc.
* `h`: Hubble's constant divided by (100 km/s/Mpc).
* `A_s`: amplitude of the primordial PS
* `n_s`: index of the primordial PS

For some specific cosmologies you can also use functions **`ccl_parameters_create_flat_lcdm`**, **`ccl_parameters_create_flat_wcdm`**, **`ccl_parameters_create_flat_wacdm`**, **`ccl_parameters_create_lcdm`**, which automatically set some parameters. For more information, see file `include/ccl_core.c`.

### The `ccl_cosmology` object
For the majority of CCL's functions you need an object of type **`ccl_cosmology`**, which can be initalize by function **`ccl_cosmology_create`**
```c
ccl_cosmology * ccl_cosmology_create(ccl_parameters params, ccl_configuration config);
```
Note that the function returns a pointer. Variable **`params`** of type **`ccl_parameters`** contains cosmological parameters created in previous step. Structure **`ccl_configuration`** contains information about methods for computing transfer function, matter power spectrum and mass function (for available methods see `include/ccl_config.h`). For now, you should use default configuration **`default_config`**
```c
const ccl_configuration default_config = {ccl_boltzmann_class, ccl_halofit, ccl_tinker};
```
After you are done working with this cosmology object, you should free its work space by **`ccl_cosmology_free`**
```c
void ccl_cosmology_free(ccl_cosmology * cosmo);
```

### Distances and Growth factor
With defined cosmology we can now compute distances, growth factor (and rate) or sigma_8. For comoving radial distance you can call function **`ccl_comoving_radial_distance`**
```c
double ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a);
```
which returns distance to scale factor **`a`** in units of Mpc. For luminosity distance call function **`ccl_luminosity_distance`**
```c
double ccl_luminosity_distance(ccl_cosmology * cosmo, double a);
```
which also returns distance in units of Mpc. For growth factor (normalized to 1 at **`z`** = 0) at sale factor **`a`** call **`ccl_growth_factor`**
```c
double ccl_growth_factor(ccl_cosmology * cosmo, double a);
```
For more routines to compute distances and growth rates (e.g. at multiple times at once) see file `include/ccl_background.h`

###  Matter power spectra and sigma_8
For given cosmology we can compute linear and non-linear matter power spectra using functions **`ccl_linear_matter_power`** and **`ccl_nonlin_matter_power`**
```c
double ccl_linear_matter_power(ccl_cosmology * cosmo, double k, double a);
double ccl_nonlin_matter_power(ccl_cosmology * cosmo, double k, double a);
```
Sigma_8 can be calculated by function **`ccl_sigma8`**, or more generally by function **`ccl_sigmaR`**, which computes the variance of the density field smoothed by spherical top-hat window function on a comoving distance **`R`** (in Mpc).
```c
double ccl_sigmaR(ccl_cosmology *cosmo, double R);
double ccl_sigma8(ccl_cosmology *cosmo);

````
These and other functions for different matter power spectra can be found in file *include/ccl_power.h*.

### Angular power spectra
CCL can compute angular power spectra for two tracer types: galaxy number counts and galaxy weak lensing. Tracer parameters are defined in structure **CCL_ClTracer**. In general, you can create this object with function **ccl_cl_tracer_new**
````c
CCL_ClTracer *ccl_cl_tracer_new(ccl_cosmology *cosmo,int tracer_type,
				int has_rsd,int has_magnification,int has_intrinsic_alignment,
				int nz_n,double *z_n,double *n,
				int nz_b,double *z_b,double *b,
				int nz_s,double *z_s,double *s,
				int nz_ba,double *z_ba,double *ba,
				int nz_rf,double *z_rf,double *rf);
````
Exact definition of these parameters are described in file *include/ccl_cls.h*. Usually you can use simplified versions of this function, namely **ccl_cl_tracer_number_counts_new, ccl_cl_tracer_number_counts_simple_new, ccl_cl_tracer_lensing_new** or **ccl_cl_tracer_lensing_simple_new**. Two most simplified versions (one for number counts and one for shear) take parameters:
````c
CCL_ClTracer *ccl_cl_tracer_number_counts_simple_new(ccl_cosmology *cosmo, int nz_n,double *z_n,
                                                     double *n, int nz_b,double *z_b,double *b);
CCL_ClTracer *ccl_cl_tracer_lensing_simple_new(ccl_cosmology *cosmo, int nz_n,double *z_n,double *n);

````
where **nz_n** is dimension of arrays **z_n** and **n**. **z_n** and **n** are arrays for the number count of objects per redshift interval (arbitrary normalization - renormalized inside). **nz_b, z_b** and **b** are the same for the clustering bias.
With initialized tracers you can compute limber power spectrum with **ccl_angular_cl**
````c
double ccl_angular_cl(ccl_cosmology *cosmo,int l,CCL_ClTracer *clt1,CCL_ClTracer *clt2);
````
After you are done working with tracers, you should free its work space by **ccl_cl_tracer_free**
````c
void ccl_cl_tracer_free(CCL_ClTracer *clt);
````

### Halo mass function
The halo mass function *dN/dM* can be obtained by function **ccl_massfunc**
````c
double ccl_massfunc(ccl_cosmology * cosmo, double halo_mass, double redshift)
````
where **halo_mass** is mass smoothing scale (in units of *M_sun/h*. For more details (or other functions like *sigma_M*) see *include/ccl_massfunc.h* and *src/mass_func.c*.

### LSST Specifications
CCL includes LSST specifications for the expected galaxy distributions of the full galaxy clustering sample and the lensing source galaxy sample. Start by defining a flexible photometric redshift model given by function
````c
double (* your_pz_func)(double photo_z, double spec_z, void *param);
````
which returns the probability of measuring a particular photometric redshift, given a spectroscopic redshift and other relevant parameters. Then you call function **ccl_specs_create_photoz_info**
````c
user_pz_info* ccl_specs_create_photoz_info(void * user_params, 
                                           double(*user_pz_func)(double, double,void*));
````
which creates a strcture **user_pz_info** which holds information needed to compute *dN/dz*
````c
typedef struct {
	//first double corresponds to photo-z, second to spec-z
        double (* your_pz_func)(double, double, void *); 
        void *  your_pz_params;
} user_pz_info;
````
The expected *dN/dz* for lensing or clustering galaxies with given binnig can be obtained by function **ccl_specs_dNdz_tomog**
````c
int ccl_specs_dNdz_tomog(double z, int dNdz_type, double bin_zmin, double bin_zmax, 
                         user_pz_info * user_info,  double *tomoout);
````
Result is returned in **tomoout**. This function returns zero if called with an allowable type of dNdz, non-zero otherwise. Allowed types of dNdz (currently one for clustering and three for lensing - fiducial, optimistic, and conservative - cases are considered) and other information and functions like bias clustering or sigma_z are specified in file *include/ccl_lsst_specs.h* 

After you are done working with photo_z, you should free its work space by **ccl_specs_free_photoz_info**
````c
void ccl_specs_free_photoz_info(user_pz_info *my_photoz_info);
````

## Example code
This code can also be found in *tests/ccl_sample_run.c* You can run the following example code. For this you will need to compile with the following command:
````sh
gcc -Wall -Wpedantic -g -I/path/to/install/include -std=gnu99 -fPIC tests/ccl_sample_run.c \
-o tests/ccl_sample_run -L/path/to/install/lib -L/usr/local/lib -lgsl -lgslcblas -lm -lccl
````
where */path/to/install/* is the path to the location where the library has been installed.

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
#define S8 0.80
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 512

// The user defines a structure of parameters to the user-defined function for the photo-z probability 
struct user_func_params
{
	double (* sigma_z) (double);
};

// The user defines a function of the form double function ( z_ph, z_spec, void * user_pz_params) where user_pz_params is a pointer to the parameters of the user-defined function. This returns the probability of obtaining a given photo-z given a particular spec_z.
double user_pz_probability(double z_ph, double z_s, void * user_par)
{
        struct user_func_params * p = (struct user_func_params *) user_par;
        return exp(- (z_ph-z_s)*(z_ph-z_s) / (2.*(p->sigma_z(z_s))*(p->sigma_z(z_s)))) / (pow(2.*M_PI,0.5)*(p->sigma_z(z_s))*(p->sigma_z(z_s)));
}

int main(int argc,char **argv){

	// Initialize cosmological parameters
	ccl_configuration config=default_config;
	config.transfer_function_method=ccl_bbks;
	ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,NAN,NS,-1,NULL,NULL);
	params.sigma_8=S8;

	// Initialize cosmology object given cosmo params
	ccl_cosmology *cosmo=ccl_cosmology_create(params,config);

	// Compute radial distances (see include/ccl_background.h for more routines)
	printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc\n",
		ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD)));
	printf("Luminosity distance to z = %.3lf is chi = %.3lf Mpc\n",
		ZD,ccl_luminosity_distance(cosmo,1./(1+ZD)));

        //Consistency check
        printf("Scale factor is a=%.3lf \n",1./(1+ZD));
        printf("Consistency: Scale factor at chi=%.3lf Mpc is a=%.3lf\n",
               ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status),
               ccl_scale_factor_of_chi(cosmo,ccl_comoving_radial_distance(cosmo,1./(1+ZD), &status), &status));
	 
	// Compute growth factor and growth rate (see include/ccl_background.h for more routines)
	printf("Growth factor and growth rate at z = %.3lf are D = %.3lf and f = %.3lf\n",
		ZD, ccl_growth_factor(cosmo,1./(1+ZD)),ccl_growth_rate(cosmo,1./(1+ZD)));

	// Compute sigma_8
	printf("Initializing power spectrum...\n");
	printf("sigma_8 = %.3lf\n\n", ccl_sigma8(cosmo));

	//Create tracers for angular power spectra
	double z_arr_gc[NZ],z_arr_sh[NZ],nz_arr_gc[NZ],nz_arr_sh[NZ],bz_arr[NZ];
	for(int i=0;i<NZ;i++)
	{
		z_arr_gc[i]=Z0_GC-5*SZ_GC+10*SZ_GC*(i+0.5)/NZ;
		nz_arr_gc[i]=exp(-0.5*pow((z_arr_gc[i]-Z0_GC)/SZ_GC,2));
		bz_arr[i]=1+z_arr_gc[i];
		z_arr_sh[i]=Z0_SH-5*SZ_SH+10*SZ_SH*(i+0.5)/NZ;
		nz_arr_sh[i]=exp(-0.5*pow((z_arr_sh[i]-Z0_SH)/SZ_SH,2));
	}

	//Galaxy clustering tracer
	CCL_ClTracer *ct_gc=ccl_cl_tracer_number_counts_simple_new(cosmo,NZ,z_arr_gc,nz_arr_gc,NZ,z_arr_gc,bz_arr);

	//Cosmic shear tracer
	CCL_ClTracer *ct_wl=ccl_cl_tracer_lensing_simple_new(cosmo,NZ,z_arr_sh,nz_arr_sh);
	printf("ell C_ell(g,g) C_ell(g,s) C_ell(s,s) | r(g,s)\n");
	for(int l=2;l<=NL;l*=2)
	{
		double cl_gg=ccl_angular_cl(cosmo,l,ct_gc,ct_gc); //Galaxy-galaxy
		double cl_gs=ccl_angular_cl(cosmo,l,ct_gc,ct_wl); //Galaxy-lensing
		double cl_ss=ccl_angular_cl(cosmo,l,ct_wl,ct_wl); //Lensing-lensing
		printf("%d %.3lE %.3lE %.3lE | %.3lE\n",l,cl_gg,cl_gs,cl_ss,cl_gs/sqrt(cl_gg*cl_ss));
	}
	printf("\n");

	//Free up tracers
	ccl_cl_tracer_free(ct_gc);
	ccl_cl_tracer_free(ct_wl);
	
	//Halo mass function
	printf("M\tdN/dM(z = 0, 0.5, 1))\n");
	for(int logM=9;logM<=15;logM+=1)
	{
		printf("%.1e\t",pow(10,logM));
		for(double z=0; z<=1; z+=0.5)
		{
			printf("%e\t", ccl_massfunc(cosmo, pow(10,logM),z));
		}
		printf("\n");
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
	
	double z_test;
	double dNdz_tomo;
	int z,status;
	FILE * output;
	
	//Try splitting dNdz (lensing) into 5 redshift bins
	double tmp1,tmp2,tmp3,tmp4,tmp5;
	printf("Trying splitting dNdz (lensing) into 5 redshift bins. Output written into file tests/specs_example_tomo_lens.out\n");
	output = fopen("./tests/specs_example_tomo_lens.out", "w");     
	for (z=0; z<100; z=z+1){
		z_test = 0.035*z;
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,6., pz_info_example,&dNdz_tomo); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.,0.6, pz_info_example,&tmp1); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 0.6,1.2, pz_info_example,&tmp2);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.2,1.8, pz_info_example,&tmp3);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 1.8,2.4, pz_info_example,&tmp4); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_WL_FID, 2.4,3.0, pz_info_example,&tmp5);
		fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
	}

	fclose(output);

	//Try splitting dNdz (clustering) into 5 redshift bins
	printf("Trying splitting dNdz (clustering) into 5 redshift bins. Output written into file tests/specs_example_tomo_clu.out\n");
	output = fopen("./tests/specs_example_tomo_clu.out", "w");     
	for (z=0; z<100; z=z+1){
		z_test = 0.035*z;
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.,6., pz_info_example,&dNdz_tomo); 
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.,0.6, pz_info_example,&tmp1);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,0.6,1.2, pz_info_example,&tmp2);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,1.2,1.8, pz_info_example,&tmp3);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,1.8,2.4, pz_info_example,&tmp4);
		status = ccl_specs_dNdz_tomog(z_test, DNDZ_NC,2.4,3.0, pz_info_example,&tmp5);
		fprintf(output, "%f %f %f %f %f %f %f\n", z_test,tmp1,tmp2,tmp3,tmp4,tmp5,dNdz_tomo);
	}

	fclose(output);
	
	//Free up photo-z info
	ccl_specs_free_photoz_info(pz_info_example);

	//Always clean up!!
	ccl_cosmology_free(cosmo);

	return 0;
}
````

## Python wrapper
A Python wrapper for CCL is provided through a module called *pyccl*. The whole CCL interface can be accessed through regular Python functions and classes, with all of the computation happening in the background through the C code. The functions all support *numpy* arrays as inputs and outputs, with any loops being performed in the C code for speed.

The Python module has essentially the same functions as the C library, just presented in a more standard Python-like way. You can inspect the available functions and their arguments by using the built-in Python **help()** function, as with any Python module.

Below is a simple example Python script that creates a new **Cosmology** object, and then uses it to calculate the angular power spectra for a simple lensing cross-correlation. It should take a few seconds on a typical laptop.

````python
import pyccl as ccl
import numpy as np

# Create new Parameters object, containing cosmo parameter values
p = ccl.Parameters(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=1e-10, n_s=0.96)

# Create new Cosmology object with these parameters. This keeps track of
# previously-computed cosmological functions
cosmo = ccl.Cosmology(p)

# Define a simple binned galaxy number density curve as a function of redshift
z_n = np.linspace(0., 1., 200)
n = np.ones(z_n.shape)

# Create objects to represent tracers of the weak lensing signal with this
# number density (with has_intrinsic_alignment=False)
lens1 = ccl.ClTracerLensing(cosmo, False, z_n, n)
lens2 = ccl.ClTracerLensing(cosmo, False, z_n, n)

# Calculate the angular cross-spectrum of the two tracers as a function of ell
ell = np.arange(2, 10)
cls = ccl.angular_cl(cosmo, lens1, lens2, ell)
print cls
````


# License, Credits, Feedback etc
The CCL is still under development and should be considered research in progress. You are welcome to re-use the code, which is open source and available under the modified BSD license. If you make use of any of the ideas or software in this package in your own research, please cite them as "(LSST DESC, in preparation)" and provide a link to this repository: https://github.com/LSSTDESC/CCL If you have comments, questions, or feedback, please [write us an issue](https://github.com/LSSTDESC/CCL/issues).


