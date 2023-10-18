import numpy as np
import pytest

import pyccl as ccl

# NOTE: We now check up to kmax=23
# because CLASS v3 has a slight mismatch
# (6e-3) at higher k wavenumbers.
KMAX = 23
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
        Neff=3.046, T_CMB=2.725,
        Omega_k=0,
        w0=w_0[model],
        wa=w_a[model],
        m_nu=mnu[model],
        mass_split='list',
        transfer_function='boltzmann_class')

    a = 1

    data_lin = np.loadtxt("./benchmarks/data/model%d_pk_nu.txt" % (model+1))
    k_lin = data_lin[:, 0] * cosmo['h']
    pk_lin = data_lin[:, 1] / (cosmo['h']**3)

    pk_lin_ccl = ccl.linear_matter_power(cosmo, k_lin, a)

    assert np.allclose(pk_lin_ccl[k_lin < KMAX],
                       pk_lin[k_lin < KMAX],
                       rtol=POWER_NU_TOL)

    data_nl = np.loadtxt("./benchmarks/data/model%d_pk_nl_nu.txt" % (model+1))
    k_nl = data_nl[:, 0] * cosmo['h']
    pk_nl = data_nl[:, 1] / (cosmo['h']**3)
    pk_nl_ccl = ccl.nonlin_matter_power(cosmo, k_nl, a)
    assert np.allclose(pk_nl_ccl, pk_nl, rtol=POWER_NU_TOL)
