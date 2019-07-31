import numpy as np
import pyccl as ccl


def test_mu_sigma_mg():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        mu_0=0.1,
        sigma_0=0.2)

    assert np.allclose(ccl.mu_MG(cosmo, 1), 0.1)
    assert np.allclose(ccl.Sig_MG(cosmo, 1), 0.2)
