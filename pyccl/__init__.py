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
if environ.get("CLASS_PARAM_DIR") is None:
    environ["CLASS_PARAM_DIR"] = path.dirname(path.abspath(__file__))

from . import ccllib as lib
from . import core, constants, background, power, massfunction, halomodel, p2d

# Core data structures
from .core import Cosmology

# Background cosmology functions and growth functions
from .background import growth_factor, growth_factor_unnorm, \
    growth_rate, comoving_radial_distance, comoving_angular_distance, \
    h_over_h0, luminosity_distance, distance_modulus, scale_factor_of_chi, \
    omega_x, rho_x

# Generalized power spectra
from .p2d import Pk2D

# Power spectrum calculations and sigma8
from .power import linear_matter_power, nonlin_matter_power, sigmaR, \
    sigmaV, sigma8

# Halo mass function
from .massfunction import massfunc, massfunc_m2r, sigmaM, halo_bias

# Cl's and tracers
from .cls import angular_cl, NumberCountsTracer, WeakLensingTracer, CMBLensingTracer

from .redshifts import  dNdz_tomog, PhotoZFunction, PhotoZGaussian, dNdzFunction, dNdzSmail

# Useful constants and unit conversions
physical_constants = lib.cvar.constants

from .correlation import (
    correlation, correlation_3d, correlation_multipole, correlation_3dRsd,
    correlation_3dRsd_avgmu, correlation_pi_sigma)

# Properties of haloes
from .halomodel import halomodel_matter_power, halo_concentration

# Specific to massive neutrinos
from .neutrinos import Omeganuh2, nu_masses

# Expose function to toggle debug mode
from .pyutils import debug_mode

from .errors import CCLError
