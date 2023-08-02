import numpy as np
import pytest
import pyccl as ccl
from pyccl.modified_gravity import MuSigmaMG


POWER_MG_TOL = 1e-2


@pytest.mark.parametrize('model', list(range(5)))
def test_power_mg(model):
    mu_0 = [0., 0.1, -0.1, 0.1, -0.1]
    sigma_0 = [0., 0.1, -0.1, -0.1, 0.1]

    cosmo = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        A_s=2.1e-9,
        n_s=0.96,
        Neff=3.046,
        Omega_k=0,
        mg_parametrization=MuSigmaMG(
            mu_0=mu_0[model],
            sigma_0=sigma_0[model]),
        matter_power_spectrum='linear',
        transfer_function='boltzmann_class')

    data = np.loadtxt("./benchmarks/data/model%d_pk_MG.dat" % model)

    a = 1
    k = data[:, 0] * cosmo['h']
    pk = data[:, 1] / (cosmo['h']**3)
    pk_ccl = ccl.linear_matter_power(cosmo, k, a)
    err = np.abs(pk_ccl/pk - 1)

    if model == 0:
        assert np.allclose(err, 0, rtol=0, atol=POWER_MG_TOL)
    else:
        msk = data[:, 0] > 5e-3
        assert np.allclose(err[msk], 0, rtol=0, atol=POWER_MG_TOL)
