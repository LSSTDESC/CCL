# CCL
DESC Core Cosmology Library: cosmology routines with validated numerical accuracy.

The library is written in C99 and all functionality is directly callable from C and C++ code.  We also provide python bindings for higher-level functions.

See also our wiki https://github.com/DarkEnergyScienceCollaboration/CCL/wiki
# Installation
In order to compile CCL you need GSL. You can get GSL here: https://www.gnu.org/software/gsl/. Note that CCL uses version 2+ of GSL (which is not yet standard in all systems).

To install CCL, from the base directory (the one where this file is located) run:
````sh
./configure
make
make install
````
Often admin privileges will be needed to install the library. If you have those just type:
````sh
sudo make install
````
If you don't have admin privileges, you can still install the library by running
````sh
./configure --prefix=/path/to/install
make
make install
````
where /path/to/install is the absolute path to the directory where you want the library to be installed. If non-existing, this will create two directories, /path/to/install/include and /path/to/install/lib, and the library and header files will be installed there. Note that, in order to use CCL with your own scripts you'll have to add /path/to/install/lib to your LD_LIBRARY_PATH.

All unit tests can be run after installation by running
````sh
make check
````

## Known installation issues
1. You need to link to GSL-2 in your local version of the Makefile.
2. Sometimes, "make check" can fail. In that case, go to "*tests/ccl_test.c*" and comment out "**define CTEST_SEGFAULT**"

 
# Documentation
This document contains basic information about used structures and functions. At the end of document is provided code which implements these basic functions (also in *tests/min_code.c*).
### Cosmological parameters
Start by defining cosmological parameters defined in structure **ccl_parameters**. This structure (exact definition in *include/ccl_core.h*) contains densities of matter, parameters of dark energy (*w0, wa*), Hubble parameters, primordial poer spectra, radiation parameters, derived parameters (*sigma_8, Omega_1, z_star*) and modified growth rate.

You can initialize this structure through function **ccl_parameters_create** which returns object of type **ccl_parameters**.
````c
ccl_parameters ccl_parameters_create(
	double Omega_c, double Omega_b, double Omega_k, double Omega_n, double w0, double wa, double h,
	double A_s, double n_s, int nz_mgrowth, double *zarr_mgrowth, double *dfarr_mgrowth
);
````
where:
* Omega_c: cold dark matter
* Omega_b: baryons
* Omega_m: matter
* Omega_n: neutrinos
* Omega_k: curvature
* little omega_x means Omega_x*h^2
* w0: Dark energy eq of state parameter
* wa: Dark energy eq of state parameter, time variation
* H0: Hubble's constant in km/s/Mpc.
* h: Hubble's constant divided by (100 km/s/Mpc).
* A_s: amplitude of the primordial PS
* n_s: index of the primordial PS

For some specific cosmologies you can also use functions **ccl_parameters_create_flat_lcdm, ccl_parameters_create_flat_wcdm, ccl_parameters_create_flat_wacdm, ccl_parameters_create_lcdm**, which automatically set some parameters. For more information, see file *include/ccl_core.c*
### Cosmology object
For majority of CCL`s functions you need an object of type **ccl_cosmology**, which can be initalize by function **ccl_cosmology_create**
````c
ccl_cosmology * ccl_cosmology_create(ccl_parameters params, ccl_configuration config);
````
Note that the function returns a pointer. Variable **params** of type **ccl_parameters** contains cosmological parameters created in previous step. Structure **ccl_configuration** contains information about methods for computing transfer function, matter power spectrum and mass function (for available methods see *include/ccl_config.h*). For now, you should use default configuration **default_config**
````c
const ccl_configuration default_config = {ccl_boltzmann_class, ccl_halofit, ccl_tinker};
````
After you are done working with this cosmology object, you should free its work space by **ccl_cosmology_free**
````c
void ccl_cosmology_free(ccl_cosmology * cosmo);
````
### Distances and Growth factor
With defined cosmology we can now compute distances, growth factor (and rate) or sigma_8. For comoving radial distance you can call function **ccl_comoving_radial_distance**
````c
double ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a);
````
which returns distance to scale factor **a** in units of Mpc. For luminosity distance call function **ccl_luminosity_distance**
````c
double ccl_luminosity_distance(ccl_cosmology * cosmo, double a);
````
which also returns distance in units of Mpc. For growth factor (normalized to 1 at **z** = 0) at sale factor **a** call **ccl_growth_factor**
````c
double ccl_growth_factor(ccl_cosmology * cosmo, double a);
````
For more routines to compute distances and growth rates (e.g. at multiple times at once) see file *include/ccl_background.h*
###  Matter power spectra and sigma_8
For given cosmology we can compute linear and non-linear matter power spectra using functions **ccl_linear_matter_power** and **ccl_nonlin_matter_power**
````c
double ccl_linear_matter_power(ccl_cosmology * cosmo, double a, double k);
double ccl_nonlin_matter_power(ccl_cosmology * cosmo, double a, double k);
````
Sigma_8 can be calculated by function **ccl_sigma8**, or more generally by function **ccl_sigmaR**, which computes the variance of the density field smoothed by spherical top-hat window function on a comoving distance **R** (in Mpc).
````c
double ccl_sigmaR(ccl_cosmology *cosmo, double R);
double ccl_sigma8(ccl_cosmology *cosmo);
````
These and other functions for different matter power spectra can be found in file *include/ccl_power.h*.
<<<<<<< b57cc2940e42eed4b11c528eafdbabca66c1460b
### Example code
=======

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
CCL_ClTracer *ccl_cl_tracer_number_counts_simple_new(ccl_cosmology *cosmo, int nz_n,double *z_n,double *n, int nz_b,double *z_b,double *b);
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
user_pz_info* ccl_specs_create_photoz_info(void * user_params, double(*user_pz_func)(double, double,void*));
````
which creates a strcture **user_pz_info** which holds information needed to compute *dN/dz*
````c
typedef struct {
        double (* your_pz_func)(double, double, void *); //first double corresponds to photo-z, second to spec-z
        void *  your_pz_params;
} user_pz_info;
````
The expected *dN/dz* for lensing or clustering galaxies with given binnig can be obtained by function **ccl_specs_dNdz_tomog**
````c
int ccl_specs_dNdz_tomog(double z, int dNdz_type, double bin_zmin, double bin_zmax, user_pz_info * user_info,  double *tomoout);
````
Result is returned in **tomoout**. This function returns zero if called with an allowable type of dNdz, non-zero otherwise. Allowed types of dNdz (currently one for clustering and three for lensing - fiducial, optimistic, and conservative - cases are considered) and other information and functions like bias clustering or sigma_z are specified in file *include/ccl_lsst_specs.h* 

After you are done working with photo_z, you should free its work space by **ccl_specs_create_photoz_info**
````c
user_pz_info* ccl_specs_create_photoz_info(void * user_params, double(*user_pz_func)(double, double,void*));
````

## Example code
>>>>>>> halo mass func
This code can also be found in *tests/min_code.h* You can run the following example code. For this you will need to compile with:
````sh
gcc -Wall -Wpedantic -g -O0 -I./include -std=c99 -fPIC tests/min_code.c -o tests/min_code -L./lib -L/usr/local/lib -lgsl -lgslcblas -lm -Lclass -lclass -lccl
````

````c
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "ccl.h"

#define OC 0.25
#define OB 0.05
#define OK 0.00
#define ON 0.00
#define HH 0.70
#define W0 -1.0
#define WA 0.00
#define NS 0.96
#define AS 2.1E-9
#define ZD 0.5
#define NZ 128
#define Z0_GC 0.50
#define SZ_GC 0.05
#define Z0_SH 0.65
#define SZ_SH 0.05
#define NL 512

int main(int argc,char **argv){
	// Initialize cosmological parameters
	ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,AS,NS,-1,NULL,NULL);

	// Initialize cosmology object given cosmo params
	ccl_cosmology *cosmo=ccl_cosmology_create(params,default_config);

	// Compute radial distances
	printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc\n",
		ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD)));
	printf("Luminosity distance to z = %.3lf is chi = %.3lf Mpc\n",
		ZD,ccl_luminosity_distance(cosmo,1./(1+ZD)));

	// Compute growth factor and growth rate
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

	//Free up tracers
	ccl_cl_tracer_free(ct_gc);
	ccl_cl_tracer_free(ct_wl);

	//Always clean up!!
	ccl_cosmology_free(cosmo);

	return 0;
}
````

# License
CCL is now under development.
