# flake8: noqa E402
from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass
del version, PackageNotFoundError

# Set the environment variable for default config path
from os import environ, path
if environ.get("CLASS_PARAM_DIR") is None:
    environ["CLASS_PARAM_DIR"] = path.dirname(path.abspath(__file__))
del environ, path

# Patch for deprecated alias in Numpy >= 1.20.0 (used in ISiTGR & FAST-PT).
# Deprecation cycle starts in Numpy 1.20 and ends in Numpy 1.24.
from packaging.version import parse
import numpy
numpy.int = int if parse(numpy.__version__) >= parse("1.20.0") else numpy.int
del parse, numpy

# SWIG-generated
from . import ccllib as lib

# Hashing, Caching, CCL base, Mutation locks
from .base import *

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

from .pspec import PowerSpectrum

# Generalized power spectra
from .pk2d import (
    Pk2D,
    parse_pk2d,
)

# Generalized connected trispectra
from .tk3d import Tk3D

# Power spectrum calculations, sigma8 and kNL
from .power import *

# Baryons & Neutrinos
from .bcm import (
    bcm_model_fka,
    bcm_correct_pk2d,
)

from .neutrinos import *

# Cells & Tracers
from .cells import angular_cl
from .tracers import (
    Tracer,
    NzTracer,
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

# Miscellaneous
from .pyutils import debug_mode, resample_array

# Deprecated & Renamed modules
import warnings as _warnings
_warnings.warn(
    "The default CMB temperature (T_CMB) will change in CCLv3.0.0, "
    "from 2.725 to 2.7255 (Kelvin).", CCLDeprecationWarning)

def __getattr__(name):
    rename = {"cls": "cells"}
    if name in rename:
        from .errors import CCLDeprecationWarning
        _warnings.warn(f"Module {name} has been renamed to {rename[name]}.",
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

from .baryons import Baryons


__all__ = (
    'lib',
    'CCLError', 'CCLWarning', 'CCLDeprecationWarning',
    'Cosmology', 'CosmologyVanillaLCDM', 'CosmologyCalculator',
    'growth_factor', 'growth_factor_unnorm', 'growth_rate',
    'comoving_radial_distance', 'angular_diameter_distance',
    'comoving_angular_distance', 'luminosity_distance', 'distance_modulus',
    'h_over_h0', 'scale_factor_of_chi', 'omega_x', 'rho_x', 'sigma_critical',
    'Pk2D', 'parse_pk2d', 'Tk3D',
    'bcm_model_fka', 'bcm_correct_pk2d',
    'angular_cl',
    'Tracer', 'NumberCountsTracer', 'WeakLensingTracer', 'CMBLensingTracer',
    'tSZTracer', 'CIBTracer', 'ISWTracer', 'NzTracer',
    'get_density_kernel', 'get_kappa_kernel', 'get_lensing_kernel',
    'correlation', 'correlation_3d', 'correlation_multipole',
    'correlation_3dRsd', 'correlation_3dRsd_avgmu', 'correlation_pi_sigma',
    'angular_cl_cov_cNG', 'angular_cl_cov_SSC',
    'sigma2_B_disc', 'sigma2_B_from_mask',
    'debug_mode', 'resample_array',
    'halomodel_matter_power', 'halo_concentration',
    'onehalo_matter_power', 'twohalo_matter_power',
    'massfunc', 'halo_bias', 'massfunc_m2r', 'nfw_profile_3d',
    'einasto_profile_3d', 'hernquist_profile_3d', 'nfw_profile_2d',
    'Baryons',
)
