import numpy as np
import pyccl as ccl

BEMULIN_TOLERANCE = 1e-3


def test_baccoemu_linear():
    bemu = ccl.BaccoemuLinear()
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
        transfer_function=bemu)

    data = np.loadtxt("./benchmarks/data/baccoemu_linear.txt")

    k = data[:, 0] * cosmo['h']
    pk = data[:, 1] / cosmo['h']**3
    a = 1

    linpk = cosmo.get_linear_power()
    err = np.abs(pk / linpk(k, a) - 1)
    assert np.allclose(err, 0, atol=BEMULIN_TOLERANCE, rtol=0)

    ktest, pktest = bemu.get_pk_at_a(cosmo, 1.0)
    pktest = np.exp(np.interp(np.log(k), np.log(ktest), np.log(pktest)))
    err = np.abs(pktest / pk - 1)
    assert np.allclose(err, 0, atol=BEMULIN_TOLERANCE, rtol=0)


def test_baccoemu_linear_A_s():
    bemu = ccl.BaccoemuLinear()
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
        wa=0,
        transfer_function=bemu)

    data = np.loadtxt("./benchmarks/data/baccoemu_linear.txt")

    k = data[:, 0] * cosmo['h']
    pk = data[:, 1] / cosmo['h']**3
    a = 1

    linpk = cosmo.get_linear_power()
    err = np.abs(pk / linpk(k, a) - 1)
    assert np.allclose(err, 0, atol=BEMULIN_TOLERANCE, rtol=0)

    ktest, pktest = bemu.get_pk_at_a(cosmo, 1.0)
    pktest = np.exp(np.interp(np.log(k), np.log(ktest), np.log(pktest)))
    err = np.abs(pktest / pk - 1)
    assert np.allclose(err, 0, atol=BEMULIN_TOLERANCE, rtol=0)
