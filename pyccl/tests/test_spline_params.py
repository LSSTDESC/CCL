import pytest
import pyccl as ccl


def test_spline_params():
    cosmo = ccl.Cosmology(
                Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
                transfer_function='bbks', matter_power_spectrum='linear')

    assert cosmo.cosmo.spline_params.A_SPLINE_MAX == 1.0

    with pytest.raises(RuntimeError):
        cosmo.cosmo.spline_params.A_SPLINE_MAX = 0.9
