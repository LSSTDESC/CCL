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


@pytest.mark.parametrize('tf', ['boltzmann_camb', 'boltzmann_class', 'bbks',
                                  'eisenstein_hu'])
def test_mg_scale_dep_transfer_err(tf):
    """Test that scale-dependent MG params raise error with non-isitgr TF.

    Issue #1191: CCL should complain when mu-Sigma scale-dependent parameters
    are non-zero but transfer function isn't 'boltzmann_isitgr'.
    """
    with pytest.raises(ValueError):
        ccl.Cosmology(
            Omega_c=0.25,
            Omega_b=0.05,
            h=0.7,
            sigma8=0.8,
            n_s=0.96,
            mg_parametrization=MuSigmaMG(
                mu_0=0.1, sigma_0=0.2,
                c1_mg=0.5, c2_mg=0.5, lambda_mg=0.5
            ),
            transfer_function=tf,
            matter_power_spectrum='linear'
        )


def test_mg_scale_dep_allowed_cases():
    """Test that scale-dependent MG params work with isitgr.

    Also test that GR cases (c1=c2=1 or all zeros) work with any TF.
    """
    # Test 1: Scale-dependent MG with boltzmann_isitgr should work
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        mg_parametrization=MuSigmaMG(
            mu_0=0.1, sigma_0=0.2,
            c1_mg=0.5, c2_mg=0.5, lambda_mg=0.5
        ),
        transfer_function='boltzmann_isitgr',
        matter_power_spectrum='linear'
    )
    assert cosmo is not None

    # Test 2: GR case (c1=c2=1) with boltzmann_camb should work
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        mg_parametrization=MuSigmaMG(
            mu_0=0.1, sigma_0=0.2,
            c1_mg=1, c2_mg=1, lambda_mg=0.5
        ),
        transfer_function='boltzmann_camb',
        matter_power_spectrum='linear'
    )
    assert cosmo is not None

    # Test 3: All zeros case with boltzmann_camb should work
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        mg_parametrization=MuSigmaMG(
            mu_0=0.1, sigma_0=0.2,
            c1_mg=0, c2_mg=0, lambda_mg=0
        ),
        transfer_function='boltzmann_camb',
        matter_power_spectrum='linear'
    )
    assert cosmo is not None
