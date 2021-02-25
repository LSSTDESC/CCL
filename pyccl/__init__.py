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
from . import core, constants, background, power, halomodel, pk2d, tk3d, haloprofile, halos, massfunction, nl_pt

# Core data structures
from .core import Cosmology, CosmologyVanillaLCDM, CosmologyCalculator

# Background cosmology functions and growth functions
from .background import growth_factor, growth_factor_unnorm, \
    growth_rate, comoving_radial_distance, angular_diameter_distance, comoving_angular_distance, \
    h_over_h0, luminosity_distance, distance_modulus, scale_factor_of_chi, \
    omega_x, rho_x

# Generalized power spectra
from .pk2d import Pk2D, parse_pk2d

# Generalized connected trispectra
from .tk3d import Tk3D

# Power spectrum calculations, sigma8 and kNL
from .power import linear_power, nonlin_power, linear_matter_power, nonlin_matter_power, \
    sigmaR, sigmaV, sigma8, sigmaM, kNL

# BCM stuff
from .bcm import bcm_model_fka, bcm_correct_pk2d

# Old halo mass function
from .massfunction import massfunc, halo_bias, massfunc_m2r

# Cl's and tracers
from .tracers import Tracer, NumberCountsTracer, WeakLensingTracer, CMBLensingTracer, \
    tSZTracer, get_density_kernel, get_kappa_kernel, get_lensing_kernel
from .cls import angular_cl
from .covariances import angular_cl_cov_cNG

# Useful constants and unit conversions
physical_constants = lib.cvar.constants

from .correlations import (
    correlation, correlation_3d, correlation_multipole, correlation_3dRsd,
    correlation_3dRsd_avgmu, correlation_pi_sigma)

# Properties of haloes
from .halomodel import (
    halomodel_matter_power, halo_concentration,
    onehalo_matter_power, twohalo_matter_power)

# Halo density profiles
from .haloprofile import nfw_profile_3d, einasto_profile_3d, hernquist_profile_3d, nfw_profile_2d

# Specific to massive neutrinos
from .neutrinos import Omeganuh2, nu_masses

# Expose function to toggle debug mode
from .pyutils import debug_mode, resample_array

from .errors import CCLError, CCLWarning
