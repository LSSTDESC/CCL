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
        m_nu_type='normal',
        m_nu=0.0,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0,
        bcm_log10Mc=14.0,
        baryons_power_spectrum='bcm')

    cosmo_nobar = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.2e-9,
        n_s=0.96,
        Neff=3.046,
        m_nu_type='normal',
        m_nu=0.0,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0)

    data = np.loadtxt("./benchmarks/data/w_baryonspk_nl.dat")
    data_nobar = np.loadtxt("./benchmarks/data/wo_baryonspk_nl.dat")

    k = data[:, 0] * cosmo['h']
    a = 1

    fbcm = ccl.bcm_model_fka(cosmo, k, a)
    err = np.abs(data[:, 1]/data_nobar[:, 1]/fbcm - 1)
    assert np.allclose(err, 0, atol=BCM_TOLERANCE, rtol=0)

    ratio = (
        ccl.nonlin_matter_power(cosmo, k, a) /
        ccl.nonlin_matter_power(cosmo_nobar, k, a))
    err = np.abs(data[:, 1]/data_nobar[:, 1]/ratio - 1)
    assert np.allclose(err, 0, atol=BCM_TOLERANCE, rtol=0)
