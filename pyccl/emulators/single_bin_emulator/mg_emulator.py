import numpy as np
from .linear_nn import LinearBoostNN
from .nonlinear_nn import NonlinearBoostNN
from .stitching import BoostEmulator


class MGEmulator:

    def __init__(self, model_dir="models/"):

        # -----------------------------
        # Linear emulator
        # -----------------------------
        linear_emu = LinearBoostNN(
            f"{model_dir}/linear_boost_nn.pt"
        )

        # -----------------------------
        # Nonlinear emulator (NN, replaces GP)
        # -----------------------------
        # NonlinearBoostNN needs the k-grid as an array, not a path
        # (unlike NonlinearBoostGP, which loaded it internally).
        k_native = np.loadtxt(f"{model_dir}/cola_eg.txt", usecols=0)

        nonlinear_emu = NonlinearBoostNN(
            f"{model_dir}/singlebin_nn_emulator.pt",
            f"{model_dir}/bin5_nn_emulator.pt",
            k_native
        )

        # -----------------------------
        # Final stitched emulator
        # -----------------------------
        self.emulator = BoostEmulator(
            linear_emu,
            nonlinear_emu
        )

    def predict_boost(
        self,
        cosmo,
        mu,
        eta,
        bin_index,
        zs
    ):

        return self.emulator.predict_boost(
            cosmo,
            mu,
            eta,
            bin_index,
            zs
        )

    def __call__(
        self,
        cosmo,
        mu,
        eta,
        bin_index,
        zs
    ):
        """
        Optional shorthand:
        k, boost = emu(...)
        """

        return self.predict_boost(
            cosmo,
            mu,
            eta,
            bin_index,
            zs
        )
