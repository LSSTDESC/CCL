import os
import numpy as np
import pandas as pd
import pyccl as ccl


def test_mass_function():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'data/haloprofile_models.txt'),
        sep=' ',
        names=['r', 'rho_nfw', 'rho_ein', 'rho_hern', 'sigma_nfw'],
        skiprows=1)

    with ccl.Cosmology(Omega_c=0.2603, Omega_b=0.0486, Omega_g=0, Omega_k=0,
                       h=0.6774, sigma8=0.8159, n_s=0.9667, Neff=3.046, m_nu=0.0,
                       w0=-1, wa=0) as c:

        concentration = 5.0
        halo_mass = 6e13
        overdensity_factor = 200.0
        a = 1.0

        for i, r in enumerate(df['r']):
            rho_nfw = ccl.nfw_profile_3d(
                c, concentration, halo_mass, overdensity_factor, a, r)
            rho_ein = ccl.einasto_profile_3d(
                c, concentration, halo_mass, overdensity_factor, a, r)
            rho_hern = ccl.hernquist_profile_3d(
                c, concentration, halo_mass, overdensity_factor, a, r)
            sigma_nfw = ccl.nfw_profile_2d(
                c, concentration, halo_mass, overdensity_factor, a, r)

            assert np.allclose(
                rho_nfw, df['rho_nfw'].values[i], rtol=1e-3)
            assert np.allclose(
                rho_ein, df['rho_ein'].values[i], rtol=1e-3)
            assert np.allclose(
                rho_hern, df['rho_hern'].values[i], rtol=1e-3)
            assert np.allclose(
                sigma_nfw, df['sigma_nfw'].values[i], rtol=1e-3)
