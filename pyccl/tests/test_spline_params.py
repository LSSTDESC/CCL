import pytest
import pyccl as ccl


def test_spline_params():
    cosmo = ccl.Cosmology(
                Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
                transfer_function='bbks', matter_power_spectrum='linear')
    cosmo.cosmo.spline_params.A_SPLINE_MAX = 0.9
    with pytest.raises(ccl.CCLError):
        ccl.angular_diameter_distance(cosmo, a1=1.0)

    with pytest.raises(ccl.CCLError):
        ccl.growth_factor(cosmo, a=1.0)

    with pytest.raises(ccl.CCLError):
        ccl.linear_matter_power(cosmo, k=1.0, a=1.0)
