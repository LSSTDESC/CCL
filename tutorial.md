# Documentation
This documentat contains basic information about used structures and functions. At the end of document is provided code which implements these basic function. See also test **ccl_sample_run.c**
### Cosmological parameters
Start by defining cosmological parameters defined in structure **ccl_parameters**. This structure (exact definition in *include/ccl_core.h*) contains densities of matter, parameters of dark energy (*w0, wa*), Hubble parameters, primordial poer spectra, radiation parameters, derived parameters (*sigma_8, Omega_1, z_star*) and modified growth rate.

You can initialize this structure through function **ccl_parameters_create** which returns object of type **ccl_parameters**.

    ccl_parameters ccl_parameters_create(
    double Omega_c, double Omega_b, double Omega_k, double Omega_n, double w0, double wa, double h, double A_s, double n_s,int nz_mgrowth,double *zarr_mgrowth,double *dfarr_mgrowth
    );
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
For CCL`s functions you need an object of type **ccl_cosmology**, which can be initalize by function **ccl_cosmology_create**

    ccl_cosmology * ccl_cosmology_create(ccl_parameters params, ccl_configuration config);

Note that the function returns pointer. Variable **params** of type **ccl_parameters** contains cosmological parameters created in previous step. Object **config** of type **ccl_configuration**. Structure **ccl_configuration** contains information about methods for computing transfer function, matter power spectrum and mass function (for available methods see *include/ccl_config.h*). For now, you should use default configuration **default_config**

    const ccl_configuration default_config = {ccl_fitting_function, ccl_halofit, ccl_tinker};

