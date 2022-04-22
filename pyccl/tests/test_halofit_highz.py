import numpy as np
import pyccl as ccl

COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halofit')


def test_halofit_highz():
    vals = [(25, 75)] + list(zip(range(0, 90), range(1, 99)))
    for zl, zh in vals:
        al = 1.0/(1 + zl)
        ah = 1.0/(1 + zh)

        k = np.logspace(0, 2, 10)
        pkratl = (
            ccl.nonlin_matter_power(COSMO, k, al)
            / ccl.linear_matter_power(COSMO, k, al)
        )
        pkrath = (
            ccl.nonlin_matter_power(COSMO, k, ah)
            / ccl.linear_matter_power(COSMO, k, ah)
        )

        assert np.all(pkratl >= pkrath), (zl, zh, pkratl, pkrath)
