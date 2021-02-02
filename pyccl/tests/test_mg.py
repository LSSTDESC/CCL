import pyccl as ccl
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
            mu_0=0.1,
            sigma_0=0.2,
            transfer_function=tf,
            matter_power_spectrum='linear'
        )
        ccl.linear_matter_power(cosmo, 1, 1)


@pytest.mark.parametrize('mp', ['emu', 'halofit'])
def test_mu_sigma_matter_power_err(mp):
    from pyccl.pyutils import assert_warns
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
        # Also raises a warning, so catch that.
        assert_warns(ccl.CCLWarning, ccl.nonlin_matter_power, cosmo, 1, 1)
