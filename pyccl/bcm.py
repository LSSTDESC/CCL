__all__ = ("bcm_model_fka", "bcm_correct_pk2d",)

import numpy as np

from . import BaryonsSchneider15, check, deprecated, lib, unlock


@deprecated(new_api=BaryonsSchneider15)
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
    bcm = BaryonsSchneider15(log10Mc=cosmo['bcm_log10Mc'],
                             eta_b=cosmo['bcm_etab'],
                             k_s=cosmo['bcm_ks'])
    return bcm.boost_factor(cosmo, k, a)


@deprecated(new_api=BaryonsSchneider15)
@unlock(name="pk2d")
def bcm_correct_pk2d(cosmo, pk2d):
    """Apply the BCM model correction factor to a given power spectrum.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
        pk2d (:class:`~pyccl.pk2d.Pk2D`): power spectrum.
    """

    bcm = BaryonsSchneider15(log10Mc=cosmo['bcm_log10Mc'],
                             eta_b=cosmo['bcm_etab'],
                             k_s=cosmo['bcm_ks'])
    a_arr, lk_arr, pk_arr = pk2d.get_spline_arrays()
    k_arr = np.exp(lk_arr)
    fka = bcm.boost_factor(cosmo, k_arr, a_arr)
    pk_arr *= fka
    if pk2d.psp.is_log:
        np.log(pk_arr, out=pk_arr)
    lib.f2d_t_free(pk2d.psp)
    status = 0
    pk2d.psp, status = lib.set_pk2d_new_from_arrays(
        lk_arr, a_arr, pk_arr.flatten(),
        int(pk2d.extrap_order_lok),
        int(pk2d.extrap_order_hik),
        pk2d.psp.is_log, status)
    check(status)
