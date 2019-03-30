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
    double cl_cc=ccl_angular_cl(cosmo,l,ct_cl,ct_cl, &status); //CMBLensingTracer-CMBLensingTracer
    double cl_cg=ccl_angular_cl(cosmo,l,ct_cl,ct_gc, &status); //CMBLensingTracer-Clustering
    double cl_cs=ccl_angular_cl(cosmo,l,ct_wl,ct_cl, &status); //CMBLensingTracer-Galaxy lensing
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
lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))
lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z_n, n))

# Calculate the angular cross-spectrum of the two tracers as a function of ell
ell = np.arange(2, 10)
cls = ccl.angular_cl(cosmo, lens1, lens2, ell)
print cls
````
