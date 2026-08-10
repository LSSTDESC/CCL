import numpy as np
import torch
from .model_nonlinear import EmulatorNN


class NonlinearBoostNN:
    
    BIN_MAP = {0: 0, 1: 3, 2: 2, 3: 1}  # bin 4 never uses this

    def __init__(
        self,
        checkpoint_path,
        bin5_checkpoint_path,
        k_native,
        input_dim=8,
        output_dim=5,
        bin5_input_dim=7,
        bin5_output_dim=5,
        key_prefix=None,
        bin5_key_prefix=None,
    ):
        # ------------------------------------------
        # "Full" model (bins 0-3)
        # ------------------------------------------
        self.model, self.pca, self.X_mean, self.X_std, self.y_mean, \
            self.y_std = self._load_checkpoint(
                checkpoint_path, input_dim, output_dim,
                key_prefix, state_dict_key="model_state_dict",
                pca_key="pca", x_mean_key="X_mean", x_std_key="X_std",
                y_mean_key="y_mean", y_std_key="y_std",
            )

        # ------------------------------------------
        # "bin5" model (bin 4 only) -- note the
        # bin5_-prefixed key names inside the checkpoint
        # ------------------------------------------
        self.bin5_model, self.bin5_pca, self.bin5_X_mean, \
            self.bin5_X_std, self.bin5_y_mean, self.bin5_y_std = \
            self._load_checkpoint(
                bin5_checkpoint_path, bin5_input_dim, bin5_output_dim,
                bin5_key_prefix, state_dict_key="bin5_model_state_dict",
                pca_key="bin5_pca", x_mean_key="bin5_X_mean",
                x_std_key="bin5_X_std", y_mean_key="bin5_y_mean",
                y_std_key="bin5_y_std",
            )

        self.k_native = np.asarray(k_native)

    # ==================================================
    # Shared checkpoint-loading logic (handles the
    # full-path key-prefix quirk for either checkpoint)
    # ==================================================
    @staticmethod
    def _load_checkpoint(
        checkpoint_path, input_dim, output_dim, key_prefix,
        state_dict_key, pca_key, x_mean_key, x_std_key,
        y_mean_key, y_std_key,
    ):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )

        if key_prefix is None:
            if state_dict_key in checkpoint:
                key_prefix = ""
            else:
                candidates = [
                    k for k in checkpoint
                    if k.endswith(state_dict_key)
                ]
                if not candidates:
                    raise KeyError(
                        f"Could not find a '{state_dict_key}' key "
                        f"(plain or prefixed) in {checkpoint_path}."
                    )
                key_prefix = candidates[0][: -len(state_dict_key)]

        def ckpt_get(name):
            return checkpoint[key_prefix + name]

        model = EmulatorNN(input_dim=input_dim, output_dim=output_dim)
        model.load_state_dict(ckpt_get(state_dict_key))
        model.eval()

        pca = ckpt_get(pca_key)
        X_mean = ckpt_get(x_mean_key)
        X_std = ckpt_get(x_std_key)
        y_mean = ckpt_get(y_mean_key)
        y_std = ckpt_get(y_std_key)

        return model, pca, X_mean, X_std, y_mean, y_std

    # ==================================================
    # Predict on native emulator grid
    # ==================================================
    def predict_native(self, cosmo, mu, bin_index, zs):

        zs = np.atleast_1d(zs)
        N = len(zs)
        bin_index = int(bin_index)

        if bin_index == 4:
            # -------- dedicated bin5 model --------
            X = np.column_stack([
                np.full(N, cosmo["Omega_m"]),
                np.full(N, cosmo["h"]),
                np.full(N, cosmo["Omega_b"]),
                np.full(N, cosmo["n_s"]),
                np.full(N, cosmo["A_s"]),
                np.full(N, mu),
                zs,
            ])

            X_stdized = (X - self.bin5_X_mean) / self.bin5_X_std

            with torch.no_grad():
                pred_std = self.bin5_model(
                    torch.tensor(X_stdized, dtype=torch.float32)
                ).numpy()

            pca_modes = pred_std * self.bin5_y_std + self.bin5_y_mean

            # NO exp() here -- confirmed this model outputs boost
            # directly, unlike the GP's bin5 branch.
            boost = self.bin5_pca.inverse_transform(pca_modes)

        else:
            # -------- shared "full" model, bins 0-3 --------
            internal_bin = self.BIN_MAP[bin_index]

            X = np.column_stack([
                np.full(N, cosmo["Omega_m"]),
                np.full(N, cosmo["h"]),
                np.full(N, cosmo["Omega_b"]),
                np.full(N, cosmo["n_s"]),
                np.full(N, cosmo["A_s"]),
                np.full(N, mu),
                np.full(N, internal_bin),
                zs,
            ])

            X_stdized = (X - self.X_mean) / self.X_std

            with torch.no_grad():
                pred_std = self.model(
                    torch.tensor(X_stdized, dtype=torch.float32)
                ).numpy()

            pca_modes = pred_std * self.y_std + self.y_mean

            boost = self.pca.inverse_transform(pca_modes)

        return boost

    # ==================================================
    # Matches NonlinearBoostGP.predict_boost's signature
    # ==================================================
    def predict_boost(self, cosmo, mu, bin_index, zs):

        boost = self.predict_native(cosmo, mu, bin_index, zs)

        return self.k_native, boost
