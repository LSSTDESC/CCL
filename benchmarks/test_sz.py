import numpy as np
import pyccl as ccl


def test_szcl():
    COSMO = ccl.Cosmology(
        Omega_b=0.05,
        Omega_c=0.25,
        h=0.67,
        n_s=0.9645,
        A_s=2.02E-9,
        Neff=3.046)
    bm = np.loadtxt("benchmarks/data/sz_cl_P13_szpowerspectrum.txt",
                    unpack=True)
    l_bm = bm[0]
    cl_bm = bm[1]
    cl_bm *= (2*np.pi) / (1E12*l_bm*(l_bm+1))
    mass_def = ccl.halos.MassDef(500, 'critical')
    hmf = ccl.halos.MassFuncTinker08(COSMO, mass_def=mass_def)
    hbf = ccl.halos.HaloBiasTinker10(COSMO, mass_def=mass_def)
    hmc = ccl.halos.HMCalculator(COSMO, hmf, hbf, mass_def)
    prf = ccl.halos.HaloProfilePressureGNFW(mass_bias=1./1.41)
    pk = ccl.halos.halomod_Pk2D(COSMO, hmc, prf, get_2h=False)
    tr = ccl.tSZTracer(COSMO, z_max=4.)
    cl = ccl.angular_cl(COSMO, tr, tr, l_bm, p_of_k_a=pk)

    assert np.all(np.fabs(cl/cl_bm-1) < 2E-2)
