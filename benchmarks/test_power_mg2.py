import numpy as np
import pytest

import pyccl as ccl

POWER_MG_TOL = 1e-4


@pytest.mark.parametrize('model', list(range(5)))
def test_power_mg(model):
    mu_0 = [0., 0.1, -0.1, 0.1, -0.1]
    sigma_0 = [0., 0.1, -0.1, -0.1, 0.1]
    h0=0.7
    cosmo = ccl.Cosmology(
        Omega_c=0.112/h0**2,
        Omega_b=0.0226/h0**2,
        h=h0,
        A_s=2.1e-9,
        n_s=0.96,
        Neff=3.046,
        mu_0=mu_0[model],
        sigma_0=sigma_0[model],
        Omega_k=0,
        m_nu=0,
        T_CMB=2.7255,
        matter_power_spectrum='linear',
        transfer_function='boltzmann_isitgr')

    data = np.loadtxt("./data/model%d_pk_MG_matterpower.dat" % model)

    a = 1
    k = data[:, 0] * cosmo['h']
    pk = data[:, 1] / (cosmo['h']**3)
    pk_ccl = ccl.linear_matter_power(cosmo, k, a)
    err = np.abs(pk_ccl/pk - 1)

    cut = data[:,0] > 1e-04   #cut two points due to cosmic variance
    assert np.allclose(err[cut], 0, rtol=0, atol=POWER_MG_TOL)
