import numpy as np
import pytest
import pyccl as ccl
from pyccl.modified_gravity import MuSigmaMG

POWER_MG_TOL = 1e-2


@pytest.mark.parametrize('model', list(range(5)))
def test_power_mg(model):
    mu_0 = [0., 0.1, -0.1, 0.1, -0.1]
    sigma_0 = [0., 0.1, -0.1, -0.1, 0.1]

    h0 = 0.6736

    # --- CCL cosmology for this model ---
    cosmoMG = ccl.Cosmology(
        Omega_c=0.1200 / h0**2,
        Omega_b=0.02237 / h0**2,
        h=h0,
        A_s=2.100e-9,
        n_s=0.9649,
        Neff=3.046,
        Omega_k=0,
        m_nu=0,
        T_CMB=2.7255,
        T_ncdm=(4/11)**(1/3),
        mass_split='equal',
        mg_parametrization=MuSigmaMG(mu_0=mu_0[model], sigma_0=sigma_0[model]),
        matter_power_spectrum='linear',
        transfer_function='boltzmann_isitgr',
    )

    # --- load benchmark for this model ---
    fname = f"./benchmarks/data/model{model:d}_pk_isitgr_matterpower.dat"
    data = np.loadtxt(fname)
    k_hmpc = data[:, 0]
    pk_bm_h3 = data[:, 1]

    a = 1.0

    # Bench file: k [h/Mpc], Pk [(Mpc/h)^3]
    # CCL expects k [1/Mpc] and returns Pk [Mpc^3]
    k = k_hmpc * cosmoMG["h"]
    pk_bm = pk_bm_h3 / (cosmoMG["h"]**3)
    pk_ccl = ccl.linear_matter_power(cosmoMG, k, a)

    frac = pk_ccl / pk_bm - 1.0
    err = np.abs(frac)

    cut = k_hmpc > 1e-04
    passed = np.allclose(err[cut], 0.0, rtol=0.0, atol=POWER_MG_TOL)

    assert passed
