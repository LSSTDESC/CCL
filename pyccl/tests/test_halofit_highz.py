import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halofit')


def test_halofit_highz():
    a0 = 1
    a95 = 1.0/(1 + 95)

    k = 10
    pkrat0 = (
        ccl.nonlin_matter_power(COSMO, k, a0)
        / ccl.linear_matter_power(COSMO, k, a0)
    )
    pkrat95 = (
        ccl.nonlin_matter_power(COSMO, k, a95)
        / ccl.linear_matter_power(COSMO, k, a95)
    )

    assert pkrat0 > pkrat95
