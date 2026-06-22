import numpy as np
import pytest

import pyccl as ccl
from classy import Class

KMIN = 1e-5
KMAX = 50
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
        Neff=3.046,  # In CLASSv3, Neff=3.044.
        T_CMB=2.725,
        Omega_k=0,
        w0=w_0[model],
        wa=w_a[model],
        m_nu=mnu[model],
        mass_split='list',
        transfer_function='boltzmann_class')

    a = 1
    z = 1/a - 1
    k_arr = np.logspace(np.log10(KMIN), np.log10(KMAX), 200)

    # Instantiate CLASS for comparison
    # Generate pk from CLASS
    N_ncdm = np.nonzero(mnu[model])[0].size
    # Values in explanatori.ini
    # N_ur = [3.044, 2.0308, 1.0176, 0.00441][N_ncdm]
    # N_ur ~ N_eff - N_ncdm * T_ncdm^4 * (11/4)^(4/3),
    # with T_ncdm = 0.71611 in CLASS
    N_ur = cosmo['Neff'] - 1.0132 * N_ncdm
    params = {'Omega_Lambda': 0,
              # 'Omega_fld': 0,  # Left unespecified to use w0, wa
              # Dark energy parameters
              'w0_fld': w_0[model],
              'wa_fld': w_a[model],
              # Cosmological parameters
              'h': cosmo['h'],
              'Omega_cdm': cosmo['Omega_c'],
              'Omega_b': cosmo['Omega_b'],
              'A_s': cosmo['A_s'],
              'n_s': cosmo['n_s'],
              'T_cmb': cosmo['T_CMB'],
              'Omega_k': cosmo['Omega_k'],
              # Neutrinos
              "N_ur": N_ur,  # Neff
              "N_ncdm": N_ncdm,  # Number of massive neutrino species
              "m_ncdm": ",".join([str(m) for m in mnu[model][:N_ncdm]]),
              # Matter power spectrum
              'non_linear': 'halofit',
              'output': 'mPk',
              'P_k_max_1/Mpc': KMAX,
              'z_pk': z
              }

    cosmo_classy = Class()
    cosmo_classy.set(params)
    cosmo_classy.compute()

    # Generate pk from CLASS and compare to CCL
    pk_lin = np.array([cosmo_classy.pk_lin(k_i, z) for k_i in k_arr])
    pk_nl = np.array([cosmo_classy.pk(k_i, z) for k_i in k_arr])
    cosmo_classy.struct_cleanup()  # Free memory used by CLASS

    # Linear Pk
    pk_lin_ccl = ccl.linear_matter_power(cosmo, k_arr, a)
    assert np.allclose(pk_lin_ccl, pk_lin,
                       rtol=POWER_NU_TOL)

    # Non-linear Pk
    pk_nl_ccl = ccl.nonlin_matter_power(cosmo, k_arr, a)
    assert np.allclose(pk_nl_ccl, pk_nl, rtol=POWER_NU_TOL)
