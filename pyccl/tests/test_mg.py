import numpy as np
import pyccl as ccl

import pytest


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


@pytest.mark.parametrize('tf', ['eisenstein_hu', 'bbks'])
def test_mu_sigma_transfer_err(tf):
    with pytest.raises(ccl.CCLError):
        cosmo = ccl.Cosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            A_s=2.1e-9,
            n_s=0.96,
            mu_0=0.1,
            sigma_0=0.2,
            transfer_function=tf,
            matter_power_spectrum='linear'
        )
        ccl.linear_matter_power(cosmo, 1, 1)


@pytest.mark.parametrize('mp', ['emu', 'halofit'])
def test_mu_sigma_matter_power_err(mp):
    with pytest.raises(ccl.CCLError):
        cosmo = ccl.Cosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            A_s=2.1e-9,
            n_s=0.96,
            mu_0=0.1,
            sigma_0=0.2,
            transfer_function=None,
            matter_power_spectrum=mp
        )
        ccl.nonlin_matter_power(cosmo, 1, 1)
