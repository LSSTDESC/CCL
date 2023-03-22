from .baryons_base import Baryons
from ..pk2d import Pk2D
import numpy as np


__all__ = ("BaryonsSchneider15",)


class BaryonsSchneider15(Baryons):
    """The BCM model boost factor for baryons.

    .. note:: BCM stands for the "baryonic correction model" of Schneider &
              Teyssier (2015; https://arxiv.org/abs/1510.06034). See the
              `DESC Note <https://github.com/LSSTDESC/CCL/blob/master/doc\
/0000-ccl_note/main.pdf>`_
              for details.

    .. note:: The boost factor is applied multiplicatively so that
              :math:`P_{\\rm corrected}(k, a) = P(k, a)\\, f_{\\rm bcm}(k, a)`.

    Args:
        log10Mc (:obj:`float`): logarithmic mass scale of hot
            gas suppression. Defaults to log10(1.2E14).
        eta_b (:obj:`float`): ratio of escape to ejection radii (see
            Teyssier et al. 2015). Defaults to 0.5.
        k_s (:obj:`float`): Characteristic scale (wavenumber) of
            the stellar component. Defaults to 55.0.
    """
    name = 'Schneider15'
    __repr_attrs__ = ("log10Mc", "eta_b", "k_s")

    def __init__(self, log10Mc=np.log10(1.2E14), eta_b=0.5, k_s=55.0):
        self.log10Mc = log10Mc
        self.eta_b = eta_b
        self.k_s = k_s

    def boost_factor(self, cosmo, k, a):
        """The BCM model boost factor for baryons.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
            k (float or array_like): Wavenumber; Mpc^-1.
            a (float or array_like): Scale factor.

        Returns:
            float or array_like: Correction factor to apply to
                the power spectrum.
        """
        a_use, k_use = map(np.atleast_1d, [a, k])
        a_use, k_use = a_use[:, None], k_use[None, :]

        z = 1/a_use - 1
        kh = k_use / cosmo['h']
        b0 = 0.105*self.log10Mc - 1.27
        bfunc = b0 / (1. + (z/2.3)**2.5)
        kg = 0.7 * (1-bfunc)**4 * self.eta_b**(-1.6)
        gf = bfunc / (1 + (kh/kg)**3) + 1. - bfunc
        scomp = 1 + (kh / self.k_s)**2
        fka = gf * scomp

        if np.ndim(k) == 0:
            fka = np.squeeze(fka, axis=-1)
        if np.ndim(a) == 0:
            fka = np.squeeze(fka, axis=0)
        return fka

    def update_parameters(self, log10Mc=None, eta_b=None, k_s=None):
        """Update BCM parameters.

        Args:
            log10Mc (:obj:`float`): logarithmic mass scale of hot
                gas suppression. Defaults to 14.08.
            eta_b (:obj:`float`): ratio of escape to ejection radii (see
                Teyssier et al. 2015). Defaults to 0.5.
            k_s (:obj:`float`): Characteristic scale (wavenumber) of
                the stellar component. Defaults to 55.0.
        """
        if log10Mc is not None:
            self.log10Mc = log10Mc
        if eta_b is not None:
            self.eta_b = eta_b
        if k_s is not None:
            self.k_s = k_s

    def _include_baryonic_effects(self, cosmo, pk):
        # Applies boost factor
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        k_arr = np.exp(lk_arr)
        fka = self.boost_factor(cosmo, k_arr, a_arr)
        pk_arr *= fka

        if pk.psp.is_log:
            np.log(pk_arr, out=pk_arr)  # in-place log

        return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    is_logp=pk.psp.is_log,
                    extrap_order_lok=pk.extrap_order_lok,
                    extrap_order_hik=pk.extrap_order_hik)
