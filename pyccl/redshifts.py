"""
The functions in redshifts provide useful routines
for making predictions for LSST-specific observables.
These include routines for predicting the linear bias
of the clustering sample, the dispersion of photometric
redshifts for clustering and lensing samples, and for
predicting the redshift distribution of a given tomographic
photometric redshift bin. We also provide functionality
for the user to incorporate their own photo-z model
and to split the redshift distributions in tomographic
bins based on photo-z cuts.

These routines are based on the LSST Science book
and the Chang et al. (2013) paper. These provide several
options to model the expected redshift distributions
of LSST galaxies that we use for the tomographic photo-z binning.
The options are as follows.

dNdz options
 - 'nc': redshift distribution for number counts, i.e., the clustering sample.
 - 'wl_cons': redshift distribution for galaxies with shapes for lensing. This
              option adopts a conservative cut on shape quality criteria.
 - 'wl_fid': redshift distribution for galaxies with shapes for lensing. This
              option adopts a fiducial cut on shape quality criteria.
 - 'wl_opt': redshift distribution for galaxies with shapes for lensing. This
              option adopts an optimistic cut on shape quality criteria.

"""

import numpy as np
from . import ccllib as lib
from .core import check
from .pyutils import _vectorize_fn, _vectorize_fn_simple

dNdz_types = {
    'nc':           lib.DNDZ_NC,
    'wl_cons':      lib.DNDZ_WL_CONS,
    'wl_fid':       lib.DNDZ_WL_FID,
    'wl_opt':       lib.DNDZ_WL_OPT
}

"""A user-defined photo-z function.
This functions allows the user to create (or
delete) a function that returns the likelihood of measuring
a certain z_ph given a z_spec, allowing for user-defined arguments.
"""


class PhotoZFunction(object):

    def __init__(self, func, args=None):
        """Create a new photo-z function.

        Args:
            func (:obj: callable): Must have the call signature
                                   func(z_ph, z_s, args).
            args (tuple, optional): Extra arguments to be passed as the third
                                    argument of func().
        """
        # Wrap user-defined function up so that only two args are needed
        # at run-time

        def _func(z_ph, z_s):
            return func(z_ph, z_s, args)

        # Create user_pz_info object
        self.pz_func = lib.specs_create_photoz_info_from_py(_func)

    def __del__(self):
        """Destructor for PhotoZFunction object."""
        try:
            lib.specs_free_photoz_info(self.pz_func)
        except Exception:
            pass


class PhotoZGaussian(PhotoZFunction):
    """
    Gaussian photo-z function with sigma(z) = sigma_z0 (1 + z).
    """

    def __init__(self, sigma_z0):
        """Create a new Gaussian photo-z function.

        Args:
            sigma_z0 (float): Width of photo-z uncertainty at z=0, assuming
            that the uncertainty evolves like sigma_z0 * (1 + z).
        """
        # Create user_pz_info object
        self.sigma_z0 = sigma_z0
        self.pz_func = lib.specs_create_gaussian_photoz_info(sigma_z0)

    def __del__(self):
        """Destructor for PhotoZGaussian object."""
        try:
            lib.specs_free_photoz_info_gaussian(self.pz_func)
        except Exception:
            pass


def bias_clustering(cosmo, a):
    """Bias clustering, b(a), at a scale
    factor, a, of the clustering sample.
    This function outputs the expected linear bias
    of galaxies in the LSST clustering sample
    at a given scale factor a. The function is
    taken from the LSST Science Book (arXiv:0912.0201).

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        a (float or array_like): Scale factor(s), normalized to 1 today.

    Returns:
        specs_bias_clustering (float or array_like): Bias at each scale factor.
    """
    return _vectorize_fn(lib.specs_bias_clustering,
                         lib.specs_bias_clustering_vec, cosmo, a)


def sigmaz_clustering(z):
    """Photo-z dispersion, sigma(z), for the clustering sample
    at a given redshift. This function returns the expected
    photometric redshift dispersion at a given redshift for
    galaxies in the LSST clustering sample.

    .. note:: assumes Gaussian uncertainties.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        z (float or array_like): Redshift(s).

    Returns:
        specs_sigmaz_clustering (float or array_like): Dispersion at each scale
            factor.
    """
    return _vectorize_fn_simple(lib.specs_sigmaz_clustering,
                                lib.specs_sigmaz_clustering_vec, z,
                                returns_status=False)


def sigmaz_sources(z):
    """Photo-z dispersion, sigma(z), for the lensing sample.
    This function returns the expected
    photometric redshift dispersion at a given redshift for
    galaxies in the LSST weak lensing gold sample.

    .. note:: assumes Gaussian uncertainties.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
        z (float or array_like): Redshift(s).

    Returns:
        specs_sigmaz_sources (float or array_like): Dispersion at each scale
            factor.
    """
    return _vectorize_fn_simple(lib.specs_sigmaz_sources,
                                lib.specs_sigmaz_sources_vec, z,
                                returns_status=False)


def dNdz_tomog(z, dNdz_type, zmin, zmax, pz_func):
    """Calculates dNdz in a particular tomographic bin, convolved
    with a photo-z model (defined by the user), and normalized.

    Args:
        z (float or array_like): Spectroscopic redshifts to evaluate dNdz at.
        dNdz_type (:obj:`str`): Type of redshift distribution.
        zmin (float): Minimum photo-z of the bin.
        zmax (float): Maximum photo-z of the bin.
        pz_func (callable): User-defined photo-z function.

    Return:
        dNdz (float or array_like): dNdz values evalued at each z.

    """
    # Ensure that an array will be passed to specs_dNdz_tomog_vec
    z = np.atleast_1d(z)

    # Do type-check for pz_func argument
    if not isinstance(pz_func, PhotoZFunction):
        raise TypeError("pz_func must be a PhotoZFunction instance.")

    # Get dNdz type
    if dNdz_type not in dNdz_types.keys():
        raise ValueError("'%s' not a valid dNdz_type." % dNdz_type)

    # Call dNdz tomography function
    status = 0
    dNdz, status = lib.specs_dNdz_tomog_vec(dNdz_types[dNdz_type], zmin, zmax,
                                            pz_func.pz_func, z, z.size, status)
    check(status)
    return dNdz


# Provide aliases for functions to retain consistency with C API
specs_bias_clustering = bias_clustering
specs_sigmaz_clustering = sigmaz_clustering
specs_sigmaz_sources = sigmaz_sources
specs_dNdz_tomog = dNdz_tomog
