# CCL
DESC Core Cosmology Library: cosmology routines with validated numerical accuracy.

The library is written in C99 and all functionality is directly callable from C and C++ code.  We also provide python bindings for higher-level functions.
# Installation

# Documentation
This document contains basic information about used structures and functions. At the end of document is provided code which implements these basic functions (also in *tests/min_code.c*). You can also try **make example**.
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
const ccl_configuration default_config = {ccl_fitting_function, ccl_halofit, ccl_tinker};
````
### Distances and Growth factor
With defined cosmology we can now compute distances, growth factor (and rate) or sigma_8. For comoving radial distance you can call function **ccl_comoving_radial_distance**
````c
double ccl_comoving_radial_distance(ccl_cosmology * cosmo, double a);
````
which returns distance to scale factor **a** in units of Mpc/h. For luminosity distance call function **ccl_luminosity_distance**
````c
double ccl_luminosity_distance(ccl_cosmology * cosmo, double a);
````
which also returns distance in units of Mpc/h. For growth factor (normalized to 1 at **z** = 0) at sale factor **a** call **ccl_growth_factor**
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
Sigma_8 can be calculated by function **ccl_sigma8**, or more generally by function **ccl_sigmaR**
````c
double ccl_sigmaR(ccl_cosmology *cosmo, double R);
double ccl_sigma8(ccl_cosmology *cosmo);
````
These and other functions for different matter power spectra can be found in file *include/ccl_power.h*.
### Example code
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

int main(int argc,char **argv){
    // Initialize cosmological parameters
    ccl_parameters params=ccl_parameters_create(OC,OB,OK,ON,W0,WA,HH,AS,NS,-1,NULL,NULL);
    
    // Initialize cosmology object given cosmo params
    ccl_cosmology *cosmo=ccl_cosmology_create(params,default_config);
    
    // Compute radial distances
    printf("Comoving distance to z = %.3lf is chi = %.3lf Mpc/h\n",
		ZD,ccl_comoving_radial_distance(cosmo,1./(1+ZD)));
	printf("Luminosity distance to z = %.3lf is chi = %.3lf Mpc/h\n",
		ZD,ccl_luminosity_distance(cosmo,1./(1+ZD)));
		
	// Compute growth factor and growth rate
	printf("Growth factor and growth rate at z = %.3lf are D = %.3lf and f = %.3lf\n",
		ZD, ccl_growth_factor(cosmo,1./(1+ZD)),ccl_growth_rate(cosmo,1./(1+ZD)));  
		
    // Compute sigma_8
	printf("* sigma_8 = %.3lf\n", ccl_sigma8(cosmo));
}
````

# License
