from ..pk2d import Pk2D
from .pspec_base import PowerSpectrum
import numpy as np


__all__ = ("PowerSpectrumCLASS",)


class PowerSpectrumCLASS(PowerSpectrum):
    """All precision parameters are listed here.
    https://github.com/lesgourg/class_public/blob/master/include/precisions.h
    """
    name = "boltzmann_class"
    rescale_s8 = True
    rescale_mg = True

    def __init__(self, **precision):
        self.precision = precision

    def _get_power_spectrum(self, cosmo):
        """Run CLASS and return the linear power spectrum."""
        import classy
        sparams = cosmo._spline_params
        a_arr, lk_arr = self._get_spline_arrays(cosmo)

        params = {
            "output": "mPk",
            "non linear": "none",
            "k_min_tau0": sparams.K_MIN,
            "P_k_max_1/Mpc": sparams.K_MAX_SPLINE,
            "z_pk": str((1/a_arr - 1)[::-1].tolist())[1:-1],
            "modes": "s",
            "lensing": "no",
            "h": cosmo["h"],
            "Omega_cdm": cosmo["Omega_c"],
            "Omega_b": cosmo["Omega_b"],
            "Omega_k": cosmo["Omega_k"],
            "n_s": cosmo["n_s"],
            "T_cmb": cosmo["T_CMB"]}
        params.update(self.precision)

        # Dark energy.
        if (cosmo["w0"], cosmo["wa"]) != (-1, 0):
            params["Omega_Lambda"] = 0
            params['w0_fld'] = cosmo['w0']
            params['wa_fld'] = cosmo['wa']

        # Massless neutrinos.
        params["N_ur"] = cosmo["N_nu_rel"] if cosmo["N_nu_rel"] > 1e-4 else 0.

        # Massive neutrinos.
        if cosmo["N_nu_mass"] > 0:
            params["N_ncdm"] = cosmo["N_nu_mass"]
            params["m_ncdm"] = str(cosmo["m_nu"])[1:-1]

        # Power spectrum normalization.
        # If A_s is not given, we just get close and CCL will normalize it.
        A_s, sigma8 = cosmo["A_s"], cosmo["sigma8"]
        A_s_fid = A_s if np.isfinite(A_s) else self._sigma8_to_As(sigma8)
        params["A_s"] = A_s_fid

        try:
            model = classy.Class()
            model.set(params)
            model.compute()
            pk, k, z = model.get_pk_and_k_and_z(nonlinear=False)
        finally:
            if model in locals():
                model.struct_cleanup()
                model.empty()

        # Transform the output
        a_arr = (1/(1+z))
        lk_arr = np.log(k)
        pk_arr = np.log(pk.T)

        return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    is_logp=True, extrap_order_lok=1, extrap_order_hik=2)
