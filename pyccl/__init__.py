from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass
del get_distribution, DistributionNotFound

# Set the environment variable for default config path
from os import environ, path
if environ.get("CLASS_PARAM_DIR") is None:
    environ["CLASS_PARAM_DIR"] = path.dirname(path.abspath(__file__))
del environ, path

# SWIG-generated
from . import ccllib as lib

# Errors
from .errors import (
    CCLError,
    CCLWarning,
    CCLDeprecationWarning,
)

# Core data structures
from .core import (
    Cosmology,
    CosmologyVanillaLCDM,
    CosmologyCalculator,
)

# Background cosmology functions and growth functions
from .background import (
    growth_factor,
    growth_factor_unnorm,
    growth_rate,
    comoving_radial_distance,
    angular_diameter_distance,
    comoving_angular_distance,
    luminosity_distance,
    distance_modulus,
    h_over_h0,
    scale_factor_of_chi,
    omega_x,
    rho_x,
    sigma_critical,
)

# Boltzmann solvers
from .boltzmann import (
    get_camb_pk_lin,
    get_isitgr_pk_lin,
    get_class_pk_lin,
)

# Generalized power spectra
from .pk2d import (
    Pk2D,
    parse_pk2d,
)

# Generalized connected trispectra
from .tk3d import Tk3D

# Power spectrum calculations, sigma8 and kNL
from .power import (
    linear_power,
    nonlin_power,
    linear_matter_power,
    nonlin_matter_power,
    sigmaR,
    sigmaV,
    sigma8,
    sigmaM,
    kNL,
)

# Baryons & Neutrinos
from .baryons import (
    bcm_model_fka,
    bcm_correct_pk2d,
    baryon_correct,
)

from .neutrinos import (
    Omeganuh2,
    Omega_nu_h2,
    nu_masses,
)

# Cells & Tracers
from .cells import angular_cl
from .tracers import (
    Tracer,
    NumberCountsTracer,
    WeakLensingTracer,
    CMBLensingTracer,
    tSZTracer,
    CIBTracer,
    ISWTracer,
    get_density_kernel,
    get_kappa_kernel,
    get_lensing_kernel,
)

# Correlations & Covariances
from .correlations import (
    correlation,
    correlation_3d,
    correlation_multipole,
    correlation_3dRsd,
    correlation_3dRsd_avgmu,
    correlation_pi_sigma,
)

from .covariances import (
    angular_cl_cov_cNG,
    angular_cl_cov_SSC,
    sigma2_B_disc,
    sigma2_B_from_mask,
)

# Hashing, Caching, CCL base, Mutation locks
from .base import (
    CCLObject,
    CCLHalosObject,
    Hashing,
    Caching,
    hash_,
    cache,
    UnlockInstance,
    unlock_instance,
)

# Parameters
from .parameters import (
    CCLParameters,
    gsl_params,
    spline_params,
    physical_constants,
)

# Emulators
from .emulator import (
    EmulatorObject,
    Emulator,
    PowerSpectrumEmulator
)

# Miscellaneous
from .pyutils import debug_mode, resample_array

# Deprecated & Renamed modules
from . import baryons, cells


def __getattr__(name):
    rename = {"bcm": "baryons", "cls": "cells"}
    if name in rename:
        from .errors import CCLDeprecationWarning
        import warnings
        warnings.warn(f"Module {name} has been renamed to {rename[name]}.",
                      CCLDeprecationWarning)
        name = rename[name]
        return eval(name)
    raise AttributeError(f"No module named {name}.")


from .halomodel import (
    halomodel_matter_power,
    halo_concentration,
    onehalo_matter_power,
    twohalo_matter_power,
)

from .massfunction import (
    massfunc,
    halo_bias,
    massfunc_m2r,
)

from .haloprofile import (
    nfw_profile_3d,
    einasto_profile_3d,
    hernquist_profile_3d,
    nfw_profile_2d,
)


__all__ = (
    'lib', 'cache', 'hash_', 'CCLObject', 'CCLHalosObject',
    'UnlockInstance', 'unlock_instance',
    'CCLParameters', 'physical_constants', 'gsl_params', 'spline_params',
    'CCLError', 'CCLWarning', 'CCLDeprecationWarning',
    'Cosmology', 'CosmologyVanillaLCDM', 'CosmologyCalculator',
    'growth_factor', 'growth_factor_unnorm', 'growth_rate',
    'comoving_radial_distance', 'angular_diameter_distance',
    'comoving_angular_distance', 'luminosity_distance', 'distance_modulus',
    'h_over_h0', 'scale_factor_of_chi', 'omega_x', 'rho_x', 'sigma_critical',
    'get_camb_pk_lin', 'get_isitgr_pk_lin', 'get_class_pk_lin',
    'Pk2D', 'parse_pk2d', 'Tk3D',
    'linear_power', 'nonlin_power',
    'linear_matter_power', 'nonlin_matter_power',
    'sigmaR', 'sigmaV', 'sigma8', 'sigmaM', 'kNL',
    'bcm_model_fka', 'bcm_correct_pk2d', 'baryon_correct',
    'Omeganuh2', 'Omega_nu_h2', 'nu_masses',
    'angular_cl',
    'Tracer', 'NumberCountsTracer', 'WeakLensingTracer', 'CMBLensingTracer',
    'tSZTracer', 'CIBTracer', 'ISWTracer',
    'get_density_kernel', 'get_kappa_kernel', 'get_lensing_kernel',
    'correlation', 'correlation_3d', 'correlation_multipole',
    'correlation_3dRsd', 'correlation_3dRsd_avgmu', 'correlation_pi_sigma',
    'angular_cl_cov_cNG', 'angular_cl_cov_SSC',
    'sigma2_B_disc', 'sigma2_B_from_mask',
    'Hashing', 'Caching',
    'EmulatorObject', 'Emulator', 'PowerSpectrumEmulator',
    'debug_mode', 'resample_array',
    'halomodel_matter_power', 'halo_concentration',
    'onehalo_matter_power', 'twohalo_matter_power',
    'massfunc', 'halo_bias', 'massfunc_m2r', 'nfw_profile_3d',
    'einasto_profile_3d', 'hernquist_profile_3d', 'nfw_profile_2d',
    'baryons', 'cells',
)
