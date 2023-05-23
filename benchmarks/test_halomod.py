import numpy as np
import pyccl as ccl

import pytest

HALOMOD_TOLERANCE = 1E-3


@pytest.mark.parametrize('model', list(range(3)))
def test_halomod(model):
    Omega_c = [0.2500, 0.2265, 0.2685]
    Omega_b = [0.0500, 0.0455, 0.0490]
    h = [0.7000, 0.7040, 0.6711]
    sigma8 = [0.8000, 0.8100, 0.8340]
    n_s = [0.9600, 0.9670, 0.9624]

    cosmo = ccl.Cosmology(
        Omega_c=Omega_c[model],
        Omega_b=Omega_b[model],
        h=h[model],
        sigma8=sigma8[model],
        n_s=n_s[model],
        Neff=0,
        m_nu=0,
        Omega_k=0,
        transfer_function='eisenstein_hu',
        matter_power_spectrum='linear')

    data_z0 = np.loadtxt("./benchmarks/data/pk_hm_c%d_z0.txt" % (model+1))
    data_z1 = np.loadtxt("./benchmarks/data/pk_hm_c%d_z1.txt" % (model+1))

    mass_def = ccl.halos.MassDef('vir', 'matter')
    hmf = ccl.halos.MassFuncSheth99(mass_def=mass_def,
                                    mass_def_strict=False,
                                    use_delta_c_fit=True)
    hbf = ccl.halos.HaloBiasSheth99(mass_def=mass_def,
                                    mass_def_strict=False)
    cM = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
    prf = ccl.halos.HaloProfileNFW(mass_def=mass_def, concentration=cM)
    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf,
                                 mass_def=mass_def)

    z = 0.
    k = data_z0[:, 0] * cosmo['h']
    pk = data_z0[:, -1] / (cosmo['h']**3)
    pk_ccl = ccl.halos.halomod_power_spectrum(cosmo, hmc, k, 1./(1+z), prf)
    tol = pk * HALOMOD_TOLERANCE
    err = np.abs(pk_ccl - pk)
    assert np.all(err <= tol)

    z = 1.
    k = data_z1[:, 0] * cosmo['h']
    pk = data_z1[:, -1] / (cosmo['h']**3)
    pk_ccl = ccl.halos.halomod_power_spectrum(cosmo, hmc, k, 1./(1+z), prf)
    tol = pk * HALOMOD_TOLERANCE
    err = np.abs(pk_ccl - pk)
    assert np.all(err <= tol)
