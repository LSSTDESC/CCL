"""
The functions in redshifts provide useful routines
for making predictions for LSST-specific observables.
These include routines for predicting the linear bias
of the clustering sample, and for
predicting the redshift distribution of a given tomographic
photometric redshift bin. We also provide functionality
for the user to incorporate their own photo-z and true dNdz model
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
        self.pz_func = lib.create_photoz_info_from_py(_func)

    def __del__(self):
        """Destructor for PhotoZFunction object."""
        try:
            lib.free_photoz_info(self.pz_func)
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
        self.pz_func = lib.create_gaussian_photoz_info(sigma_z0)

    def __del__(self):
        """Destructor for PhotoZGaussian object."""
        try:
            lib.free_photoz_info(self.pz_func)
        except Exception:
            pass


class dNdzFunction(object):

    def __init__(self, func, args=None):
        """Create a new dNdz function.

        Args:
            func (:obj: callable): Must have the call signature
                                   func(z, args).
            args (tuple, optional): Extra arguments to be passed as the third
                                    argument of func().
        """
        # Wrap user-defined function up so that only one arg is needed
        # at run-time
        def _func(z):
            return func(z, args)

        # Create user_pz_info object
        self.dN_func = lib.create_dNdz_info_from_py(_func)

    def __del__(self):
        """Destructor for PhotoZFunction object."""
        try:
            lib.free_dNdz_info(self.dN_func)
        except Exception:
            pass


class dNdzSmail(dNdzFunction):

    def __init__(self, alpha, beta, z0):
        """Create a new dNdz function of the Smail type.
        z**alpha * exp(-(z/z0)**beta)

        Args:
           alpha (float): alpha parameter
           beta (float) : beta parameter
           z0 (float): z0 parameter
        """

        self.alpha = alpha
        self.beta = beta
        self.z0 = z0
        self.dN_func = lib.create_Smail_dNdz_info(alpha, beta, z0)

    def __del__(self):
        """Destructor for PhotoZFunction object."""
        try:
            lib.free_dNdz_info(self.dN_func)
        except Exception:
            pass


def dNdz_tomog(z, zmin, zmax, pz_func, dNdz_func):
    """Calculates dNdz in a particular tomographic bin, convolved
    with a photo-z model (defined by the user), and normalized.

    Args:
        z (float or array_like): Spectroscopic redshifts to evaluate dNdz at.
        zmin (float): Minimum photo-z of the bin.
        zmax (float): Maximum photo-z of the bin.
        pz_func (callable): User-defined photo-z function.
        dNdz_func (callable): User-defined true dNdz function.

    Return:
        dNdz (float or array_like): tomographic dNdz values evalued at each z.

    """
    # Ensure that an array will be passed to dNdz_tomog_vec
    z = np.atleast_1d(z)

    # Do type-check for pz_func argument
    if not isinstance(pz_func, PhotoZFunction):
        raise TypeError("pz_func must be a PhotoZFunction instance.")

    # Do type-check for dNdz_func argument
    if not isinstance(dNdz_func, dNdzFunction):
        raise TypeError("dNdz_func must be a dNdzFunction instance.")

    # Call dNdz tomography function
    status = 0
    dNdz, status = lib.dNdz_tomog_vec(zmin, zmax, pz_func.pz_func,
                                      dNdz_func.dN_func, z, z.size,
                                      status)
    check(status)
    return dNdz
