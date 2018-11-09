"""The pyccl package contains all of the submodules that are implemented in
individual files in CCL.
"""
# flake8: noqa
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

# Sets the environment variable for default config path if it does not
# exist yet
from os import environ, path
if environ.get("CCL_PARAM_FILE") is None:
    environ["CCL_PARAM_FILE"] = (
        path.dirname(path.abspath(__file__)) + '/ccl_params.ini')
if environ.get("CLASS_PARAM_DIR") is None:
    environ["CLASS_PARAM_DIR"] = path.dirname(path.abspath(__file__))

from . import ccllib as lib
from . import core, constants, background, power, massfunction, halomodel

# Core data structures
from .core import Cosmology

# Background cosmology functions and growth functions
from .background import growth_factor, growth_factor_unnorm, \
    growth_rate, comoving_radial_distance, comoving_angular_distance, \
    h_over_h0, luminosity_distance, distance_modulus, scale_factor_of_chi, \
    omega_x, rho_x

# Power spectrum calculations and sigma8
from .power import linear_matter_power, nonlin_matter_power, sigmaR, \
    sigmaV, sigma8

# Halo mass function
from .massfunction import massfunc, massfunc_m2r, sigmaM, halo_bias

# Cl's and tracers
from .cls import angular_cl, NumberCountsTracer, WeakLensingTracer, CMBLensingTracer

from .redshifts import  dNdz_tomog, PhotoZFunction, PhotoZGaussian, dNdzFunction, dNdzSmail

# Useful constants and unit conversions
from .constants import CLIGHT_HMPC, MPC_TO_METER, PC_TO_METER, \
                      GNEWT, RHO_CRITICAL, SOLAR_MASS

from .correlation import correlation, correlation_3d

# Properties of haloes
from .halomodel import halomodel_matter_power, halo_concentration

# Specific to massive neutrinos
from .neutrinos import Omeganuh2, nu_masses

# Expose function to toggle debug mode
from .pyutils import debug_mode

from .errors import CCLError
