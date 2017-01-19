
import ccllib
from ccllib import *
import numpy as np


def growth_factorx(cosmo, a):
    if isinstance(a, float):
        # Use single-value function
        return ccllib.growth_factor(cosmo, a)
    elif isinstance(a, np.ndarray):
        # Use vectorised function
        return ccllib.growth_factor_vec(cosmo, a, a.size)
    else:
        # Use vectorised function
        return ccllib.growth_factor_vec(cosmo, a, len(a))

#A_SPLINE_DELTA
#A_SPLINE_MAX
#A_SPLINE_MIN
#A_SPLINE_NA
#CCL_ClTracer
#CCL_ClTracer_swigregister
#CCL_ERROR_CLASS
#CCL_ERROR_INCONSISTENT
#CCL_ERROR_INTEG
#CCL_ERROR_LINSPACE
#CCL_ERROR_MEMORY
#CCL_ERROR_ROOT
#CCL_ERROR_SPLINE
#CCL_ERROR_SPLINE_EV
#CLIGHT_HMPC
#CL_TRACER_NC
#CL_TRACER_WL
#DNDZ_NC
#DNDZ_WL_CONS
#DNDZ_WL_FID
#DNDZ_WL_OPT
#EPSREL_DIST
#EPSREL_DNDZ
#EPSREL_GROWTH
#EPS_SCALEFAC_GROWTH
#GNEWT
#K_MAX
#K_MAX_INT
#K_MIN
#K_MIN_INT
#K_PIVOT
#LOGM_SPLINE_DELTA
#LOGM_SPLINE_MAX
#LOGM_SPLINE_MIN
#LOGM_SPLINE_NM
#MPC_TO_METER
#N_A
#N_K
#PC_TO_METER
#RHO_CRITICAL
#SOLAR_MASS
#SplPar
#SplPar_swigregister
#Z_MAX_SOURCES
#Z_MIN_SOURCES
#__builtins__
#__doc__
#__file__
#__name__
#__package__
#_ccllib
#_newclass
#_object
#_swig_getattr
#_swig_getattr_nondynamic
#_swig_property
#_swig_repr
#_swig_setattr
#_swig_setattr_nondynamic

#angular_cl
#angulo
#bbks
#boltzmann
#boltzmann_camb
#boltzmann_class
#check_status
#cl_tracer_free
#cl_tracer_lensing_new
#cl_tracer_lensing_simple_new
#cl_tracer_new
#cl_tracer_number_counts_new
#cl_tracer_number_counts_simple_new
#comoving_radial_distance
#comoving_radial_distances
#configuration
#configuration_swigregister
#cosmology
#cosmology_compute_distances
#cosmology_compute_growth
#cosmology_compute_power
#cosmology_compute_sigma
#cosmology_create
#cosmology_free
#cosmology_swigregister
#cvar
#data
#data_swigregister
#default_config
#eisenstein_hu
#emulator
#fitting_function
#growth_factor
#growth_factor_unnorm
#growth_factors
#growth_factors_unnorm
#growth_rate
#growth_rates
#h_over_h0
#h_over_h0s
#halo_model
#halofit
#linear
#linear_matter_power
#linear_matter_powers
#linear_spacing
#log_spacing
#luminosity_distance
#luminosity_distances
#massfunc
#massfunc_m2r
#none
#nonlin_matter_power
#omega_m_z
#parameters
#parameters_create
#parameters_create_flat_lcdm
#parameters_create_flat_wacdm
#parameters_create_flat_wcdm
#parameters_create_lcdm
#parameters_swigregister
#scale_factor_of_chi
#scale_factor_of_chis
#sigma8
#sigmaM
#sigmaR
#specs_bias_clustering
#specs_create_photoz_info
#specs_dNdz_tomog
#specs_free_photoz_info
#specs_sigmaz_clustering
#specs_sigmaz_sources
#tinker
#user_pz_info
#user_pz_info_swigregister
#watson

