import numpy as np
import pyccl as ccl
import pytest

from pyccl.errors import CCLError

COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halofit')

COSMO_LOWS8 = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.1, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halofit')


@pytest.mark.parametrize("cosmo", [COSMO, COSMO_LOWS8])
def test_halofit_highz(cosmo):
    vals = [(25, 75)] + list(zip(range(0, 98), range(1, 99)))
    for zl, zh in vals:
        al = 1.0/(1 + zl)
        ah = 1.0/(1 + zh)

        k = np.logspace(0, 2, 10)
        pkratl = (
            ccl.nonlin_matter_power(cosmo, k, al)
            / ccl.linear_matter_power(cosmo, k, al)
        )
        pkrath = (
            ccl.nonlin_matter_power(cosmo, k, ah)
            / ccl.linear_matter_power(cosmo, k, ah)
        )

        assert np.all(pkratl >= pkrath), (zl, zh, pkratl, pkrath)


def test_halofit_background_check():
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                          n_s=0.97,
                          sigma8=0.8,
                          w0=-1.04, wa=-0.1,
                          matter_power_spectrum="halofit",
                          transfer_function="eisenstein_hu")

    cosmo.cosmo.spline_params.A_SPLINE_MIN = 0.4
    cosmo.cosmo.spline_params.A_SPLINE_MINLOG = 0.3
    cosmo.cosmo.spline_params.A_SPLINE_MIN_PK = 0.4
    cosmo.cosmo.spline_params.A_SPLINE_MINLOG_PK = 0.3

    k = np.geomspace(1e-3, 1, 10)

    with pytest.raises(CCLError):
        ccl.nonlin_matter_power(cosmo, k, a=0.5)
