# Sets the environment variable for default config path if it does not exist yet
from os import environ, path
if environ.get("CCL_PARAM_FILE") is None:
    environ["CCL_PARAM_FILE"] = path.dirname(path.abspath(__file__)) + '/ccl_params.ini'

from pyccl import ccllib as lib

from pyccl import core, constants, background, power, massfunction

# Core data structures
from pyccl.core import Parameters, Cosmology

# Background cosmology functions and growth functions
from pyccl.background import growth_factor, growth_factor_unnorm, \
    growth_rate, comoving_radial_distance, comoving_angular_distance, \
    h_over_h0, luminosity_distance, distance_modulus, scale_factor_of_chi, \
    omega_x, rho_x

# Power spectrum calculations and sigma8
from pyccl.power import linear_matter_power, nonlin_matter_power, sigmaR, \
    sigma8

# Halo mass function
from pyccl.massfunction import massfunc, massfunc_m2r, sigmaM, halo_bias

# Cl's and tracers
from pyccl.cls import angular_cl, ClTracer, ClTracerNumberCounts, \
    ClTracerLensing, ClTracerCMBLensing

from pyccl.lsst_specs import bias_clustering, sigmaz_clustering, \
    sigmaz_sources, dNdz_tomog, PhotoZFunction, PhotoZGaussian

# Useful constants and unit conversions
from pyccl.constants import CLIGHT_HMPC, MPC_TO_METER, PC_TO_METER, \
                      GNEWT, RHO_CRITICAL, SOLAR_MASS

from pyccl.correlation import correlation, correlation_3d

from pyccl.neutrinos import Omeganuh2, nu_masses

# Expose function to toggle debug mode
from pyccl.pyutils import debug_mode
