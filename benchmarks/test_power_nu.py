import numpy as np
import pytest
import warnings

import pyccl as ccl

POWER_NU_TOL = 1.0E-3


@pytest.mark.parametrize('model', [0, 1, 2])
def test_power_nu(model):
    mnu = [[0.04, 0., 0.],
           [0.05, 0.01, 0.],
           [0.03, 0.02, 0.04]]
    w_0 = [-1.0, -0.9, -0.9]
    w_a = [0.0, 0.0, 0.1]

    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        Neff=3.046,
        Omega_k=0,
        w0=w_0[model],
        wa=w_a[model],
        m_nu=mnu[model],
        m_nu_type='list',
        transfer_function='boltzmann_class')

    a = 1

    data_lin = np.loadtxt("./benchmarks/data/model%d_pk_nu.txt" % (model+1))
    k_lin = data_lin[:, 0] * cosmo['h']
    pk_lin = data_lin[:, 1] / (cosmo['h']**3)

    with warnings.catch_warnings():
        # Linear power with massive neutrinos raises a warning.
        # Ignore it.
        # XXX: Do you really want to be raising a warning for this?
        #      This seems spurious to me.  (MJ)
        warnings.simplefilter("ignore")
        pk_lin_ccl = ccl.linear_matter_power(cosmo, k_lin, a)

    err = np.abs(pk_lin_ccl/pk_lin - 1)
    assert np.allclose(err, 0, rtol=0, atol=POWER_NU_TOL)

    data_nl = np.loadtxt("./benchmarks/data/model%d_pk_nl_nu.txt" % (model+1))
    k_nl = data_nl[:, 0] * cosmo['h']
    pk_nl = data_nl[:, 1] / (cosmo['h']**3)
    pk_nl_ccl = ccl.nonlin_matter_power(cosmo, k_nl, a)
    err = np.abs(pk_nl_ccl/pk_nl - 1)
    assert np.allclose(err, 0, rtol=0, atol=POWER_NU_TOL)
