import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='halofit')


def test_halofit_highz():
    a0 = 1.0/(1 + 25)
    a1 = 1.0/(1 + 50)

    k = 10
    pkrat0 = (
        ccl.nonlin_matter_power(COSMO, k, a0)
        / ccl.linear_matter_power(COSMO, k, a0)
    )
    pkrat1 = (
        ccl.nonlin_matter_power(COSMO, k, a1)
        / ccl.linear_matter_power(COSMO, k, a1)
    )

    print(pkrat0, pkrat1, flush=True)
    assert pkrat0 > pkrat1
