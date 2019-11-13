import os
import numpy as np
import pandas as pd
import pyccl as ccl


def test_mass_function():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), 'data/model1_hmf.txt'),
        sep=' ',
        names=['logmass', 'sigma', 'invsigma', 'logmf'],
        skiprows=1)

    with ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, Omega_g=0, Omega_k=0,
                       h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                       w0=-1, wa=0, T_CMB=2.7, transfer_function='bbks',
                       mass_function='tinker') as c:
        c.cosmo.params.T_CMB = 2.7

        rho_m = (
            ccl.physical_constants.RHO_CRITICAL *
            c['Omega_m'] *
            c['h']**2)
        for i, logmass in enumerate(df['logmass']):
            mass = 10**logmass
            sigma = ccl.sigmaM(c, mass, 1.0)
            loginvsigma = np.log10(1.0 / sigma)
            logmf = np.log10(
                ccl.massfunc(c, mass, 1, 200) * mass / rho_m / np.log(10))

            assert np.allclose(
                sigma, df['sigma'].values[i], rtol=3e-5)
            assert np.allclose(
                loginvsigma, df['invsigma'].values[i], rtol=1e-3)
            assert np.allclose(
                logmf, df['logmf'].values[i], rtol=5e-5)
