from .pspec_base import PowerSpectrumAnalytic
from numpy import exp, log, sqrt


__all__ = ("PowerSpectrumBBKS",)


class PowerSpectrumBBKS(PowerSpectrumAnalytic):
    """
    """
    name = "bbks"

    def _transfer(self, k, h, Om, Ob, Th27):
        q = (Th27**2 * k / (Om*h*h * exp(-Ob * (1. + sqrt(2*h) / Om))))
        polynomial = (6.71*q)**4 + (5.46*q)**3 + (16.1*q)**2 + 3.89*q + 1
        return (log(1 + 2.34*q) / (2.34*q))**2 / sqrt(polynomial)

    def _get_power_spectrum(self, cosmo):
        h = cosmo["h"]
        Om, Ob = cosmo["Omega_m"], cosmo["Omega_b"]
        Th27 = cosmo["T_CMB"] / 2.7
        n_s = cosmo["n_s"]

        a_arr, lk_arr = self._get_spline_arrays(cosmo)
        k_arr = exp(lk_arr)
        pk_arr = k_arr**n_s * self._transfer(k_arr, h, Om, Ob, Th27)
        return self._get_analytic_power(cosmo, a_arr, lk_arr, pk_arr)
