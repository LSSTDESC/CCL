# flake8: noqa E402
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

# monkey patch for isitgr and fast-pt if Numpy>=1.24
from packaging.version import parse
import numpy as np
if parse(np.__version__) >= parse('1.24'):
    np.int = int
del parse
del np

# Errors
from .errors import (
    CCLError,
    CCLWarning,
    CCLDeprecationWarning,
)

# Constants and accuracy parameters
from .parameters import (
    CCLParameters,
    gsl_params,
    spline_params,
    physical_constants,
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
from .bcm import (
    bcm_model_fka,
    bcm_correct_pk2d,
)

from .neutrinos import (
    Omeganuh2,
    nu_masses,
)

# Cells & Tracers
from .cls import angular_cl
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


# Miscellaneous
from .pyutils import debug_mode, resample_array


# Deprecated & Renamed modules
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
    'lib',
    'CCLParameters', 'spline_params', 'gsl_params', 'physical_constants',
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
    'bcm_model_fka', 'bcm_correct_pk2d',
    'Omeganuh2', 'nu_masses',
    'angular_cl',
    'Tracer', 'NumberCountsTracer', 'WeakLensingTracer', 'CMBLensingTracer',
    'tSZTracer', 'CIBTracer', 'ISWTracer',
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
)
