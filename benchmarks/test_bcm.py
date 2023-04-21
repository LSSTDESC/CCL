import numpy as np
import pyccl as ccl

BCM_TOLERANCE = 1e-4


def test_bcm():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.2e-9,
        n_s=0.96,
        Neff=3.046,
        mass_split='normal',
        m_nu=0.0,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0)

    data = np.loadtxt("./benchmarks/data/w_baryonspk_nl.dat")
    data_nobar = np.loadtxt("./benchmarks/data/wo_baryonspk_nl.dat")

    k = data[:, 0] * cosmo['h']
    a = 1

    bar = ccl.BaryonsSchneider15(log10Mc=14.)
    fbcm = bar.boost_factor(cosmo, k, a)
    err = np.abs(data[:, 1]/data_nobar[:, 1]/fbcm - 1)
    assert np.allclose(err, 0, atol=BCM_TOLERANCE, rtol=0)

    cosmo.compute_nonlin_power()
    pk_nobar = cosmo.get_nonlin_power()
    pk_wbar = bar.include_baryonic_effects(cosmo, pk_nobar)
    ratio = pk_wbar(k, a)/pk_nobar(k, a)
    err = np.abs(data[:, 1]/data_nobar[:, 1]/ratio - 1)
    assert np.allclose(err, 0, atol=BCM_TOLERANCE, rtol=0)
