"""
Module for definition of MG muSigma parameters
"""
import numpy as np
from . import ccllib as lib
from .pyutils import _vectorize_fn6


def mu_MG(cosmo, a, k=None):
    """Redshift-dependent modification to Poisson equation under modified
    gravity.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object
        a (float or array_like): Scale factor(s), normalized to 1 today.
        k (float or array_like): Wavenumber for scale

    Returns:
        float or array_like: Modification to Poisson equation \
            under modified gravity at scale factor a. \
            mu_MG is assumed to be proportional to Omega_Lambda(z), \
            see e.g. Abbott et al. 2018, 1810.02499, Eq. 9.
    """

    if (isinstance(k, (list, np.ndarray))):
        return _vectorize_fn6(lib.mu_MG,
                              lib.mu_MG_vec,
                              cosmo, a, k)
    else:
        if (isinstance(k, float) or isinstance(k, int)):
            k = np.array([k])
            return _vectorize_fn6(lib.mu_MG,
                                  lib.mu_MG_vec,
                                  cosmo, a, k)
        else:
            k = np.array([0])
            return _vectorize_fn6(lib.mu_MG,
                                  lib.mu_MG_vec,
                                  cosmo, a, k)


def Sig_MG(cosmo, a, k=None):
    """Redshift-dependent modification to Poisson equation for massless
    particles under modified gravity.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        a (float or array_like): Scale factor(s), normalized to 1 today.
        k (float or array_like): Wavenumber for scale

    Returns:
        float or array_like: Modification to Poisson equation under \
            modified gravity at scale factor a. \
            Sig_MG is assumed to be proportional to Omega_Lambda(z), \
            see e.g. Abbott et al. 2018, 1810.02499, Eq. 9.
    """
    if (isinstance(k, (list, np.ndarray))):
        return _vectorize_fn6(lib.Sig_MG,
                              lib.Sig_MG_vec,
                              cosmo, a, k)
    else:
        if (isinstance(k, float) or isinstance(k, int)):
            k = np.array([k])
            return _vectorize_fn6(lib.Sig_MG,
                                  lib.Sig_MG_vec,
                                  cosmo, a, k)
        else:
            k = np.array([0])
            return _vectorize_fn6(lib.Sig_MG,
                                  lib.Sig_MG_vec,
                                  cosmo, a, k)
