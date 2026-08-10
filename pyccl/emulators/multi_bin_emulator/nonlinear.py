import numpy as np
import torch
from scipy.interpolate import interp1d
from .model_nonlinear import EmulatorNN


class NonLinearEmulator:

    def __init__(
        self,
        checkpoint_path,
        k_native
    ):

        # ------------------------------------------
        # Load checkpoint
        # ------------------------------------------

        checkpoint = torch.load(checkpoint_path,
                                map_location="cpu",
                                weights_only=False
                               )

        # ------------------------------------------
        # Build model
        # ------------------------------------------
        self.pca = checkpoint["pca"]

        print(self.pca.n_components_)
        print(self.pca.components_.shape)

        self.model = EmulatorNN(
            input_dim=11, output_dim=self.pca.n_components_
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # ------------------------------------------
        # PCA + normalizations
        # ------------------------------------------


        self.X_mean = checkpoint["X_mean"]
        self.X_std  = checkpoint["X_std"]

        self.y_mean = checkpoint["y_mean"]
        self.y_std  = checkpoint["y_std"]

        print(self.X_mean)

        # ------------------------------------------
        # Native emulator k-grid
        # ------------------------------------------

        self.k_native = np.asarray(k_native)

    # ==================================================
    # Predict on native emulator grid
    # ==================================================

    def predict_native(
        self,
        zs,
        omega_m,
        omega_b,
        h,
        n_s,
        A_s,
        mus
    ):

        zs = np.atleast_1d(zs)

        N = len(zs)

        mus = np.asarray(mus)

        if len(mus) != 5:
            raise ValueError(
                "mus must contain 5 values "
                "[mu1, mu2, mu3, mu4, mu5]"
            )

        # ------------------------------------------
        # Build NN inputs
        # ------------------------------------------

        X = np.column_stack([

            np.full(N, omega_m),
            np.full(N, omega_b),
            np.full(N, h),
            np.full(N, n_s),
            np.full(N, A_s),

            np.full(N, mus[0]),
            np.full(N, mus[1]),
            np.full(N, mus[2]),
            np.full(N, mus[3]),
            np.full(N, mus[4]),

            zs

        ])

        # ------------------------------------------
        # Standardize
        # ------------------------------------------

        X_stdized = (
            X - self.X_mean
        ) / self.X_std

        # ------------------------------------------
        # NN prediction
        # ------------------------------------------

        with torch.no_grad():

            pred_std = self.model(
                torch.tensor(
                    X_stdized,
                    dtype=torch.float32
                )
            ).numpy()

        print(pred_std)

        # ------------------------------------------
        # De-standardize PCA coefficients
        # ------------------------------------------

        pca_modes = (
            pred_std * self.y_std
            + self.y_mean
        )

        # ------------------------------------------
        # Reconstruct boost
        # ------------------------------------------

        boost = self.pca.inverse_transform(
            pca_modes
        )
        print(boost[0,:10])

        return boost

    # ==================================================
    # Predict and interpolate onto user k-grid
    # ==================================================

    def predict(
        self,
        zs,
        omega_m,
        omega_b,
        h,
        n_s,
        A_s,
        mus,
        k
    ):

        boost_native = self.predict_native(
            zs,
            omega_m,
            omega_b,
            h,
            n_s,
            A_s,
            mus
        )

        k = np.asarray(k)
        
        

        boost_interp = np.empty(
            (boost_native.shape[0], len(k))
        )



        for i in range(boost_native.shape[0]):

            interp = interp1d(
                     np.log(self.k_native),
                     np.log(boost_native[i]),
                     kind="cubic",
                     bounds_error=False,
                     fill_value=(
                     np.log(boost_native[i,0]),
                     np.log(boost_native[i,-1])
                     )
            )

            boost_interp[i] = np.exp(
                interp(np.log(k))
            )

        return boost_interp
        