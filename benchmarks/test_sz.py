import numpy as np
import pyccl as ccl
import pytest


model_params = {'P13': 'Planck13'}


@pytest.mark.parametrize('model', ['P13'])
def test_szcl(model):
    par = model_params[model]
    bm = np.loadtxt("benchmarks/data/sz_cl_" +
                    model + "_szpowerspectrum.txt",
                    unpack=True)
    l_bm = bm[0]
    cl_bm = bm[1]
    cl_bm *= (2*np.pi) / (1E12*l_bm*(l_bm+1))
    cosmo = ccl.Cosmology(
        Omega_b=0.05,
        Omega_c=0.25,
        h=0.67,
        n_s=0.9645,
        A_s=2.02E-9,
        w0=-1,
        wa=0,
        m_nu=0,
        m_nu_type='normal',
        Neff=3.046,
        Omega_k=0)
    mass_def = ccl.halos.MassDef(500, 'critical')
    hmf = ccl.halos.MassFuncTinker08(cosmo, mass_def=mass_def)
    hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mass_def)
    hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mass_def)
    prf = ccl.halos.HaloProfilePressureGNFW(mass_bias=1./1.41,
                                            profile_params=par)
    pk = ccl.halos.halomod_Pk2D(cosmo, hmc, prf, get_2h=False)
    tr = ccl.tSZTracer(cosmo, z_max=4.)
    cl = ccl.angular_cl(cosmo, tr, tr, l_bm, p_of_k_a=pk)

    assert np.all(np.fabs(cl/cl_bm-1) < 2E-2)
