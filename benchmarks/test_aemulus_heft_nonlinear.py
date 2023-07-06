import numpy as np
import pyccl as ccl

AEMNL_TOLERANCE = 5e-3


def test_baccoemu_nonlinear():
    aem = ccl.AemulusHEFTNonlinear(n_sampling_a=20)
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
        matter_power_spectrum=aem)

    data = np.loadtxt("./benchmarks/data/aemulus_heft_nonlinear.txt")

    k = data[:, 0] * cosmo['h']
    pk = data[:, 1] / cosmo['h']**3
    a = 1

    nlpk = cosmo.get_nonlin_power()
    err = np.abs(pk / nlpk(k, a) - 1)
    assert np.allclose(err, 0, atol=AEMNL_TOLERANCE, rtol=0)

    ktest, pktest = aem.get_pk_at_a(1, cosmo)
    pktest = np.exp(np.interp(np.log(k), np.log(ktest), np.log(pktest)))
    # let's exclude scales that here would be dominated by np.interp
    mask = k > 0.05
    err = np.abs(pktest[mask] / pk[mask] - 1)
    assert np.allclose(err, 0, atol=AEMNL_TOLERANCE, rtol=0)
