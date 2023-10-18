import pyccl as ccl
from pyccl.modified_gravity import MuSigmaMG, ModifiedGravity

import pytest


@pytest.mark.parametrize('tf', ['eisenstein_hu', 'bbks'])
def test_mu_sigma_transfer_err(tf):
    with pytest.raises(ccl.CCLError):
        cosmo = ccl.Cosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            A_s=2.1e-9,
            n_s=0.96,
            mg_parametrization=MuSigmaMG(mu_0=0.1, sigma_0=0.2),
            transfer_function=tf,
            matter_power_spectrum='linear'
        )
        ccl.linear_matter_power(cosmo, 1, 1)


def test_mg_error():
    class NotMG:
        pass

    class NotMuSigma(ModifiedGravity):
        pass

    with pytest.raises(ValueError):
        ccl.Cosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            mg_parametrization=NotMG(),
            transfer_function="bbks",
        )

    with pytest.raises(NotImplementedError):
        ccl.Cosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            mg_parametrization=NotMuSigma(),
            transfer_function="bbks",
        )
