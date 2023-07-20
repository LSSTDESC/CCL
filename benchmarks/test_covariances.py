import os
import numpy as np
import pyccl as ccl


def test_ssc_WL():
    # Compare against Benjamin Joachimi's code. An overview of the methodology
    # is given in appendix E.2 of 2007.01844.
    data_dir = os.path.join(os.path.dirname(__file__), "data/covariances/")

    h = 0.7
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=h, n_s=0.97,
                          sigma8=0.8, m_nu=0.0)

    mass_def = ccl.halos.MassDef200m
    hmf = ccl.halos.MassFuncTinker10(mass_def=mass_def)
    hbf = ccl.halos.HaloBiasTinker10(mass_def=mass_def)
    con = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
    nfw = ccl.halos.HaloProfileNFW(mass_def=mass_def, concentration=con,
                                   fourier_analytic=True)
    hmc = ccl.halos.HMCalculator(mass_function=hmf, halo_bias=hbf,
                                 mass_def=mass_def)

    n_z = 100

    n_k = 200
    k_min = 1e-4
    k_max = 1e2

    a = np.linspace(1/(1+6), 1, n_z)
    k = np.geomspace(k_min, k_max, n_k)

    tk3D = ccl.halos.halomod_Tk3D_SSC(cosmo=cosmo, hmc=hmc,
                                      prof=nfw, prof2=nfw, prof12_2pt=None,
                                      lk_arr=np.log(k), a_arr=a,
                                      use_log=True)

    z, nofz = np.loadtxt(os.path.join(data_dir, "ssc_WL_nofz.txt"),
                         unpack=True)
    WL_tracer = ccl.WeakLensingTracer(cosmo, dndz=(z, nofz))

    ell = np.loadtxt(os.path.join(data_dir, "ssc_WL_ell.txt"))

    fsky = 0.05

    sigma2_B = ccl.sigma2_B_disc(cosmo, a_arr=a, fsky=fsky)
    cov_ssc = ccl.covariances.angular_cl_cov_SSC(cosmo,
                                                 tracer1=WL_tracer,
                                                 tracer2=WL_tracer,
                                                 ell=ell, t_of_kk_a=tk3D,
                                                 sigma2_B=(a, sigma2_B),
                                                 fsky=None)
    var_ssc_ccl = np.diag(cov_ssc)
    off_diag_1_ccl = np.diag(cov_ssc, k=1)

    cov_ssc_bj = np.loadtxt(os.path.join(data_dir, "ssc_WL_cov_matrix.txt"))

    # At large scales, CCL uses a different convention for the Limber
    # approximation. This factor accounts for this difference
    ccl_limber_shear_fac = np.sqrt((ell-1)*ell*(ell+1)*(ell+2))/(ell+1/2)**2
    cov_ssc_bj_corrected = cov_ssc_bj * np.outer(ccl_limber_shear_fac**2,
                                                 ccl_limber_shear_fac**2)
    var_bj = np.diag(cov_ssc_bj_corrected)
    off_diag_1_bj = np.diag(cov_ssc_bj_corrected, k=1)

    assert np.all(np.fabs(var_ssc_ccl/var_bj - 1) < 3e-2)
    assert np.all(np.fabs(off_diag_1_ccl/off_diag_1_bj - 1) < 3e-2)
    assert np.all(np.fabs(cov_ssc/cov_ssc_bj_corrected - 1) < 3e-2)
