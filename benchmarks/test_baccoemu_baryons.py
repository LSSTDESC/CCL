import numpy as np
import pyccl as ccl

BEMBAR_TOLERANCE = 1e-3


def test_baccoemu_baryons():
    baryons = ccl.BaccoemuBaryons()
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        Neff=3.046,
        mass_split='normal',
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0)

    data = np.loadtxt("./benchmarks/data/baccoemu_baryons_fk.txt")

    k = data[:, 0] * cosmo['h']
    fk = data[:, 1]
    a = 1

    cclfk = baryons.boost_factor(cosmo, k, a)
    err = np.abs(fk / cclfk - 1)
    assert np.allclose(err, 0, atol=BEMBAR_TOLERANCE, rtol=0)


def test_baccoemu_baryons_boost():
    baryons = ccl.BaccoemuBaryons()
    nlpkemu = ccl.BaccoemuNonlinear()
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.83,
        n_s=0.96,
        Neff=3.046,
        mass_split='normal',
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0,
        matter_power_spectrum=nlpkemu)

    k = np.logspace(-2, 0.5, 100)
    cclfk = baryons.boost_factor(cosmo, k, 1)
    pk_gro = cosmo.get_nonlin_power()
    pk_bcm = baryons.include_baryonic_effects(cosmo, pk_gro)
    fk = pk_bcm(k, 1) / pk_gro(k, 1)
    err = np.abs(fk / cclfk - 1)
    print(err)
    assert np.allclose(err, 0, atol=BEMBAR_TOLERANCE, rtol=0)