import numpy as np
import pyccl as ccl

EE2_TOLERANCE = 5e-3


def test_ee2_nonlinear():
    bemu = ccl.EuclidEmulator2Nonlinear()
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.05,
        h=0.67,
        A_s=2.1204e-9,
        n_s=0.96,
        Neff=3.046,
        mass_split='normal',
        m_nu=0.1,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0,
        matter_power_spectrum=bemu)

    data = np.loadtxt("./benchmarks/data/baccoemu_nonlinear.txt")

    k = data[:, 0] * cosmo['h']
    pk = data[:, 1] / cosmo['h']**3
    a = 1

    nlpk = cosmo.get_nonlin_power()
    err = np.abs(pk / nlpk(k, a) - 1)
    assert np.allclose(err, 0, atol=EE2_TOLERANCE, rtol=0)

    ktest, pktest = bemu.get_pk_at_a(1, cosmo)
    pktest = np.exp(np.interp(np.log(k), np.log(ktest), np.log(pktest)))
    err = np.abs(pktest / pk - 1)
    assert np.allclose(err, 0, atol=EE2_TOLERANCE, rtol=0)
test_ee2_nonlinear()
