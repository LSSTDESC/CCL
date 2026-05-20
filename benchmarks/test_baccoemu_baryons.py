import numpy as np
import pyccl as ccl

BEMBAR_TOLERANCE = 1e-3


def test_baccoemu_baryons():
    baryons = ccl.BaryonsBaccoemu()
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


def test_baccoemu_baryons_A_s():
    baryons = ccl.BaryonsBaccoemu()
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        A_s=2.2194e-9,
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
