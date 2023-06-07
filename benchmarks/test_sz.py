import numpy as np
import pyccl as ccl


def test_szcl():
    fsky = 1.
    COSMO = ccl.Cosmology(
        Omega_b=0.05,
        Omega_c=0.25,
        h=0.7,
        n_s=0.9645,
        A_s=2.02E-9,
        Neff=3.046, T_CMB=2.725,
        transfer_function='boltzmann_class')
    bm = np.loadtxt("benchmarks/data/sz_cl_P13_szpowerspectrum.txt",
                    unpack=True)
    l_bm = bm[0]
    cl_bm = bm[1]
    tll_bm = np.loadtxt("benchmarks/data/tSZ_trispectrum_ref_for_cobaya.txt")
    fac = 2*np.pi/(l_bm*(l_bm+1)*1E12)

    cl_bm *= fac
    tll_bm *= fac[:, None]*fac[None, :]/(4*np.pi*fsky)
    mass_def = ccl.halos.MassDef(500, 'critical')
    hmf = ccl.halos.MassFuncTinker08(mass_def=mass_def)
    hbf = ccl.halos.HaloBiasTinker10(mass_def=mass_def)
    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf,
                                 mass_def=mass_def)
    prf = ccl.halos.HaloProfilePressureGNFW(mass_def=mass_def)
    prf.update_parameters(mass_bias=1./1.41, x_out=6.)
    tr = ccl.tSZTracer(COSMO, z_max=3.)

    # Power spectrum
    pk = ccl.halos.halomod_Pk2D(COSMO, hmc, prf, get_2h=False)
    cl = ccl.angular_cl(COSMO, tr, tr, ell=l_bm, p_of_k_a=pk)

    # Covariance
    lk_arr = np.log(np.geomspace(1E-4, 1E2, 256))
    a_arr = 1./(1+np.linspace(0, 3., 20))[::-1]
    tkk = ccl.halos.halomod_Tk3D_1h(COSMO, hmc, prf,
                                    lk_arr=lk_arr, a_arr=a_arr,
                                    use_log=True)
    tll = ccl.angular_cl_cov_cNG(COSMO, tr, tr, ell=l_bm, t_of_kk_a=tkk,
                                 fsky=fsky)

    assert np.all(np.fabs(cl/cl_bm-1) < 2E-2)
    assert np.all(np.fabs(tll/tll_bm-1) < 5E-2)
