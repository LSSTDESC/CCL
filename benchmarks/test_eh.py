import numpy as np
import pyccl as ccl

EH_TOLERANCE = 1.0e-5


def test_eh():
    model = 1
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=0,
        m_nu=0.0,
        w0=-1.0,
        wa=0.0,
        m_nu_type='normal',
        Omega_g=0,
        Omega_k=0,
        transfer_function='eisenstein_hu',
        matter_power_spectrum='linear')

    data = np.loadtxt('./benchmarks/data/model%d_pk_eh.txt' % model)

    k = data[:, 0] * cosmo['h']
    for i in range(1):
        a = 1.0 / (1.0 + i)
        pk = data[:, i+1] / (cosmo['h']**3)
        pk_ccl = ccl.linear_matter_power(cosmo, k, a)
        err = np.abs(pk_ccl/pk - 1)
        assert np.allclose(err, 0, rtol=0, atol=EH_TOLERANCE)
