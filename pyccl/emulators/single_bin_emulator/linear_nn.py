import numpy as np
import torch
from .model import Net


class LinearBoostNN:

    def __init__(self, model_path, device="cpu"):

        checkpoint = torch.load(
            model_path,
            map_location=device,
            weights_only=False)

        self.model = Net(
            hidden_layers=checkpoint["hidden_layers"],
            output_dim=checkpoint["output_dim"]
        ).to(device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.device = device

        # normalization
        self.mu_param = checkpoint["mu_param"].to(device)
        self.sigma_param = checkpoint["sigma_param"].to(device)

        self.y_mu = checkpoint["y_mu"].cpu().numpy()
        self.y_sigma = checkpoint["y_sigma"].cpu().numpy()

        self.k = checkpoint["k"]

    def _map_params(self, cosmo, mu, eta, bin_index, z):

        lnAs = np.log(1e10 * cosmo["A_s"])
        bin_scaled = bin_index / 4.0

        return np.array([
            cosmo["Omega_m"],
            cosmo["Omega_b"],
            cosmo["h"],
            cosmo["n_s"],
            lnAs,
            mu,
            eta,
            z,
            bin_scaled
        ], dtype=np.float32)

    def _standardize(self, x):
        return (x - self.mu_param) / self.sigma_param

    def predict_boost(self, cosmo, mu, eta, bin_index, zs):

        zs = np.atleast_1d(zs)

        boosts = []

        for z in zs:

            params = self._map_params(cosmo, mu, eta, bin_index, z)

            x = torch.tensor(params, dtype=torch.float32).to(self.device)
            x = self._standardize(x)
            x = x.unsqueeze(0)

            with torch.no_grad():
                pred_std = self.model(x).cpu().numpy()

            pred = pred_std * self.y_sigma + self.y_mu

            shape = pred[:, :-1]
            log_ref = pred[:, -1:]

            log_boost = shape + log_ref
            boost = np.exp(log_boost).flatten()

            boosts.append(boost)

        boosts = np.array(boosts)

        return self.k, boosts
