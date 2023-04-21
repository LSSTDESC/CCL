import numpy as np
import pyccl as ccl
import pytest

BBKS_TOLERANCE = 1.0e-5


@pytest.mark.parametrize(
    'model,w0,wa',
    [(1, -1.0, 0.0),
     (2, -0.9, 0.0),
     (3, -0.9, 0.1)])
def test_bbks(model, w0, wa):
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
        mass_split='normal',
        Omega_g=0,
        Omega_k=0,
        transfer_function='bbks',
        matter_power_spectrum='linear')

    data = np.loadtxt('./benchmarks/data/model%d_pk.txt' % model)

    k = data[:, 0] * cosmo['h']
    for i in range(6):
        a = 1.0 / (1.0 + i)
        pk = data[:, i+1] / (cosmo['h']**3)
        pk_ccl = ccl.linear_matter_power(cosmo, k, a)
        err = np.abs(pk_ccl/pk - 1)
        assert np.allclose(err, 0, rtol=0, atol=BBKS_TOLERANCE)
