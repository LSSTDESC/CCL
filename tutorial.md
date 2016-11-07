# Simple Example
Start by initializing cosmological parameters defined in structure **ccl_parameters**

    typedef struct ccl_parameters;

This structure contains densities of matter, parameters of dark energy (*w0, wa*), Hubble parameters, primordial poer spectra, radiation parameters, derived parameters (*sigma_8, Omega_1, z_star*) and modified growth rate.

You can initialize this structure through function **ccl_parameters_create**

    ccl_parameters ccl_parameters_create(double Omega_c, double Omega_b, double Omega_k, double Omega_n, double w0, double wa, double h, double A_s, double n_s,int nz_mgrowth,double *zarr_mgrowth,double *dfarr_mgrowth);

Omega_c: cold dark matter
Omega_b: baryons
Omega_m: matter
Omega_n: neutrinos
Omega_k: curvature
little omega_x means Omega_x*h^2
w0: Dark energy eq of state parameter
wa: Dark energy eq of state parameter, time variation
H0: Hubble's constant in km/s/Mpc.
h: Hubble's constant divided by (100 km/s/Mpc).
A_s: amplitude of the primordial PS
n_s: index of the primordial PS

For some specific cosmologies you can also use functions **ccl_parameters_create_flat_lcdm, ccl_parameters_create_flat_wcdm, ccl_parameters_create_flat_wacdm, ccl_parameters_create_lcdm**. For more information, see file *include/ccl_core.h*

