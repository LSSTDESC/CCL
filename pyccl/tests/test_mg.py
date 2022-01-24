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
        with pytest.warns(ccl.CCLWarning):
            ccl.nonlin_matter_power(cosmo, 1, 1)


def test_planckmg_deprecated_consistent():
    planckMG = {"c1": 1.1, "c2": 1.2, "lambda": 0.05}
    with pytest.warns(ccl.CCLDeprecationWarning):
        cosmo1 = ccl.CosmologyVanillaLCDM(c1_mg=planckMG["c1"],
                                          c2_mg=planckMG["c2"],
                                          lambda_mg=planckMG["lambda"])
    cosmo1.compute_linear_power()
    pk1 = cosmo1.get_linear_power()

    cosmo2 = ccl.CosmologyVanillaLCDM(extra_parameters={"PlanckMG": planckMG})
    cosmo2.compute_linear_power()
    pk2 = cosmo2.get_linear_power()
    assert pk1 == pk2
