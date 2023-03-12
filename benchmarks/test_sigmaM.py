import numpy as np
import pyccl as ccl
import pytest

SIGMAM_TOLERANCE = 3.0E-5


@pytest.mark.parametrize(
    'model,w0,wa',
    [(1, -1.0, 0.0),
     (2, -0.9, 0.0),
     (3, -0.9, 0.1)])
def test_sigmaM(model, w0, wa):
    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=0,
        m_nu=0.0,
        w0=w0,
        wa=wa,
        T_CMB=2.7,
        m_nu_type='normal',
        Omega_g=0,
        Omega_k=0,
        transfer_function='bbks',
        matter_power_spectrum='linear')

    data = np.loadtxt('./benchmarks/data/model%d_sm.txt' % model)

    for i in range(data.shape[0]):
        m = data[i, 0] / cosmo['h']
        sm = ccl.sigmaM(cosmo, m, 1)
        err = sm / data[i, 1] - 1
        np.allclose(err, 0, rtol=0, atol=SIGMAM_TOLERANCE)
