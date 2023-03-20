from . import ccllib as lib
from .pyutils import check
from .pk2d import Pk2D
from .base import unlock_instance
import numpy as np


def bcm_model_fka(cosmo, k, a):
    """The BCM model correction factor for baryons.

    .. note:: BCM stands for the "baryonic correction model" of Schneider &
              Teyssier (2015; https://arxiv.org/abs/1510.06034). See the
              `DESC Note <https://github.com/LSSTDESC/CCL/blob/master/doc\
/0000-ccl_note/main.pdf>`_
              for details.

    .. note:: The correction factor is applied multiplicatively so that
              :math:`P_{\\rm corrected}(k, a) = P(k, a)\\, f_{\\rm bcm}(k, a)`.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        k (float or array_like): Wavenumber; Mpc^-1.
        a (float): Scale factor.

    Returns:
        float or array_like: Correction factor to apply to the power spectrum.
    """
    k_use = np.atleast_1d(k)

    z = 1./a - 1.
    kh = k_use / cosmo['h']
    b0 = 0.105*cosmo['bcm_log10Mc'] - 1.27
    bfunc = b0 / (1. + (z/2.3)**2.5)
    bfunc4 = (1-bfunc)**4
    kg = 0.7 * bfunc4 * cosmo['bcm_etab']**(-1.6)
    gf = bfunc / (1 + (kh/kg)**3) + 1. - bfunc  # k in h/Mpc
    scomp = 1 + (kh / cosmo['bcm_ks'])**2  # k in h/Mpc
    fka = gf * scomp

    if np.ndim(k) == 0:
        fka = fka[0]
    return fka


@unlock_instance(mutate=True, argv=1)
def bcm_correct_pk2d(cosmo, pk2d):
    """Apply the BCM model correction factor to a given power spectrum.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        pk2d (:class:`~pyccl.pk2d.Pk2D`): power spectrum.
    """

    if not isinstance(pk2d, Pk2D):
        raise TypeError("pk2d must be a Pk2D object")
    a_arr, lk_arr, pk_arr = pk2d.get_spline_arrays()
    k_arr = np.exp(lk_arr)
    fka = np.array([bcm_model_fka(cosmo, k_arr, a) for a in a_arr])
    pk_arr *= fka

    logp = np.all(pk_arr > 0)
    if logp:
        pk_arr = np.log(pk_arr)

    lib.f2d_t_free(pk2d.psp)
    status = 0
    pk2d.psp, status = lib.set_pk2d_new_from_arrays(
        lk_arr, a_arr, pk_arr.flatten(),
        int(pk2d.extrap_order_lok),
        int(pk2d.extrap_order_hik),
        int(logp), status)
    check(status, cosmo)
