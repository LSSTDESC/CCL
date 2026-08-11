import numpy as np
import torch
from scipy.interpolate import interp1d
from .model_linear import Net


class LinearEmulator:

    def __init__(
        self,
        checkpoint_path
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu", weights_only=False
        )

        self.hidden_layers = checkpoint["hidden_layers"]
        self.output_dim = checkpoint["output_dim"]

        self.model = Net(
            self.hidden_layers,
            self.output_dim
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.mu_param = checkpoint["mu_param"].cpu().numpy()
        self.sigma_param = checkpoint["sigma_param"].cpu().numpy()

        self.y_mu = checkpoint["y_mu"].cpu().numpy()
        self.y_sigma = checkpoint["y_sigma"].cpu().numpy()

        self.k_native = checkpoint["k"]

    # ==================================================
    # Native prediction
    # ==================================================

    def predict_native(
        self,
        omega_m,
        omega_b,
        h,
        n_s,
        ln1e10As,
        mus,
        etas,
        z_pk
    ):

        mus = np.asarray(mus)
        etas = np.asarray(etas)

        X = np.array([[

            omega_m,
            omega_b,
            h,
            n_s,
            ln1e10As,

            mus[0],
            mus[1],
            mus[2],
            mus[3],
            mus[4],

            etas[0],
            etas[1],
            etas[2],
            etas[3],
            etas[4],

            z_pk

        ]])

        X_std = (
            X - self.mu_param
        ) / self.sigma_param

        with torch.no_grad():

            pred_std = self.model(
                torch.tensor(
                    X_std,
                    dtype=torch.float32
                )
            ).numpy()

        pred = (
            pred_std * self.y_sigma
            + self.y_mu
        )

        pred_shape = pred[:, :-1]
        pred_ref = pred[:, -1:]

        pred_log = pred_shape + pred_ref

        boost = np.exp(pred_log)

        return boost[0]

    # ==================================================
    # Interpolate
    # ==================================================

    def predict(
        self,
        omega_m,
        omega_b,
        h,
        n_s,
        ln1e10As,
        mus,
        etas,
        z_pk,
        k
    ):

        boost_native = self.predict_native(
            omega_m,
            omega_b,
            h,
            n_s,
            ln1e10As,
            mus,
            etas,
            z_pk
        )

        interp = interp1d(
            np.log(self.k_native),
            np.log(boost_native),
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate"
        )

        return np.exp(
            interp(np.log(k))
        )
