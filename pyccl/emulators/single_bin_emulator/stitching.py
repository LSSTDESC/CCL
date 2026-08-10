import numpy as np
from scipy.interpolate import interp1d
from .utils import check_param_ranges



class BoostEmulator:

    def __init__(
        self,
        linear_emu,
        nonlinear_emu,
        k_low=1e-2,
        k_high=1e-1
    ):

        self.linear = linear_emu
        self.nonlinear = nonlinear_emu

        self.k_low = k_low
        self.k_high = k_high


    # -------------------------------------------------
    # Compute blending weights
    # -------------------------------------------------
    def _compute_weights(self, k):

        logk = np.log(k)

        w = (logk - np.log(self.k_low)) / (
            np.log(self.k_high) - np.log(self.k_low)
        )

        w = np.clip(w, 0.0, 1.0)

        return w


    # -------------------------------------------------
    # Main prediction
    # -------------------------------------------------
    def predict_boost(self, cosmo, mu, eta, bin_index, zs):

        zs = np.atleast_1d(zs)
        
        check_param_ranges(cosmo, mu, eta, bin_index, zs)

        # -----------------------------
        # Linear boost (NN)
        # -----------------------------
        k_lin, boost_lin = self.linear.predict_boost(
            cosmo, mu, eta, bin_index, zs
        )  # (Nz, Nk_nn)

        # -----------------------------
        # Nonlinear boost (GP)
        # -----------------------------
        k_gp, boost_gp = self.nonlinear.predict_boost(
            cosmo, mu, bin_index, zs
        )  # (Nz, Nk_gp)

        # -----------------------------
        # Interpolate GP → NN k-grid
        # -----------------------------
        interp_fn = interp1d(
            k_gp,
            boost_gp,
            axis=1,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate"
        )

        boost_gp_interp = interp_fn(k_lin)  # (Nz, Nk_nn)

        # -----------------------------
        # Compute blending weights
        # -----------------------------
        w = self._compute_weights(k_lin)  # (Nk,)

        # reshape for broadcasting
        w = w[None, :]

        # -----------------------------
        # Blend
        # -----------------------------
        boost = (1 - w) * boost_lin + w * boost_gp_interp

        return k_lin, boost
