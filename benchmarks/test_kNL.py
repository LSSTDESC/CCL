import numpy as np
import pyccl as ccl

KNL_TOLERANCE = 1.0e-5


def test_kNL():
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=0,
        m_nu=0.0,
        w0=-1.,
        wa=0.,
        T_CMB=2.7,
        mass_split='normal',
        Omega_g=0,
        Omega_k=0,
        transfer_function='bbks',
        matter_power_spectrum='linear')

    data = np.loadtxt('./benchmarks/data/kNL.txt')
    a = data[:, 0]
    kNL = data[:, 1]
    kNL_ccl = ccl.kNL(cosmo, a)
    for i in range(len(a)):
        err = np.abs(kNL_ccl[i]/kNL[i] - 1)
        assert np.allclose(err, 0, rtol=0, atol=KNL_TOLERANCE)
