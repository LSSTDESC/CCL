from ..base import CCLNamedClass
from ..pk2d import Pk2D
import numpy as np
from abc import abstractmethod


__all__ = ("rescale_power_spectrum", "PowerSpectrum", "PowerSpectrumAnalytic",)


def rescale_power_spectrum(cosmo, pk, rescale_mg=False, rescale_s8=False):
    """
    """
    rescale_mg = rescale_mg and cosmo["mu_0"] > 1e-14
    rescale_s8 = rescale_s8 and np.isfinite(cosmo["sigma8"])
    if not (rescale_mg or rescale_s8):
        return pk

    a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
    rescale_extra_musigma = 1.
    rescale_factor = np.ones_like(a_arr, dtype=float)

    # If scale-independent mu/Sigma modified gravity is in use and mu != 0,
    # get the unnormalized growth factor in MG and the one for the GR case
    # to rescale the CLASS power spectrum.
    if rescale_mg:
        # Set up a copy Cosmology in GR (mu_0 = Sigma_0 = 0) to rescale P(k).

        noGR = {"mu_0": 0, "sigma_0": 0,
                "c1_mg": 1, "c2_mg": 1, "lambda_mg": 0}
        cosmo_GR = cosmo.copy(**noGR)

        D_MG = cosmo.growth_factor_unnorm(a_arr)
        D_GR = cosmo_GR.growth_factor_unnorm(a_arr)
        rescale_factor = (D_MG / D_GR)**2
        rescale_extra_musigma = rescale_factor[-1]

    if rescale_s8:
        renorm = (cosmo["sigma8"] / cosmo.sigma8(p_of_k_a=pk))**2
        rescale_factor *= renorm / rescale_extra_musigma

    pk_arr *= rescale_factor[:, None]
    if pk.psp.is_log:
        np.log(pk_arr, out=pk_arr)

    return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                is_logp=pk.psp.is_log,
                extrap_order_lok=pk.extrap_order_lok,
                extrap_order_hik=pk.extrap_order_hik)


class PowerSpectrum(CCLNamedClass):
    """
    """

    @property
    @abstractmethod
    def rescale_s8(self) -> bool:
        """Boolean to indicate whether to perform σ8 rescaling to the
        power spectrum (output of ``get_power_spectrum``).
        """

    @property
    @abstractmethod
    def rescale_mg(self) -> bool:
        """Boolean to indicate whether to perform MG rescaling to the
        power spectrum (output of ``get_power_spectrum``).
        """

    @abstractmethod
    def _get_power_spectrum(self, cosmo) -> Pk2D:
        """Return a :obj:`~pyccl.pk2d.Pk2D` object of the power spectrum."""

    def _get_spline_arrays(self, cosmo):
        """Get the arrays used for sampling."""
        return cosmo.get_pk_spline_a(), cosmo.get_pk_spline_lk()

    def _sigma8_to_As(self, sigma8):
        """Approximate ``A_s`` given ``sigma8``."""
        return 2.43 * (sigma8 / 0.87659)**2

    def get_power_spectrum(self, cosmo) -> Pk2D:
        """
        """
        pk = self._get_power_spectrum(cosmo)
        return rescale_power_spectrum(cosmo, pk,
                                      rescale_mg=self.rescale_mg,
                                      rescale_s8=self.rescale_s8)


class PowerSpectrumAnalytic(PowerSpectrum):
    """
    """
    rescale_mg = False
    rescale_s8 = True

    def _get_analytic_power(self, cosmo, a_arr, lk_arr, pk_arr):
        """Expand an analytic P(k) into the time-dimension, scaling by the
        growth factor, and rescale sigma8.
        """
        if not np.isfinite(cosmo["sigma8"]):
            raise ValueError("sigma8 required for analytic power spectra.")

        out = np.full((a_arr.size, lk_arr.size), np.log(pk_arr))
        out += 2*np.log(cosmo.growth_factor(a_arr))[:, None]

        return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=out, is_logp=True,
                    extrap_order_lok=1, extrap_order_hik=2)
