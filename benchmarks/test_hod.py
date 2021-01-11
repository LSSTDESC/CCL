import numpy as np
import pyccl as ccl


def test_hodcl():
    # With many thanks to Ryu Makiya, Eiichiro Komatsu
    # and Shin'ichiro Ando for providing this benchmark.
    # HOD params
    lMcut = 11.8
    lM1 = 11.73
    sigma_Ncen = 0.15
    alp_Nsat = 0.77
    rmax = 4.39
    rgs = 1.17

    # Input power spectrum
    ks = np.loadtxt("benchmarks/data/k_hod.txt")
    zs = np.loadtxt("benchmarks/data/z_hod.txt")
    pks = np.loadtxt("benchmarks/data/pk_hod.txt")
    l_bm, cl_bm = np.loadtxt("benchmarks/data/cl_hod.txt",
                             unpack=True)

    # Set N(z)
    def _nz_2mrs(z):
        # From 1706.05422
        m = 1.31
        beta = 1.64
        x = z / 0.0266
        return x**m * np.exp(-x**beta)
    z1 = 1e-5
    z2 = 0.1
    z_arr = np.linspace(z1, z2, 1024)
    dndz = _nz_2mrs(z_arr)

    # CCL prediction
    # Make sure we use the same P(k)
    cosmo = ccl.CosmologyCalculator(
        Omega_b=0.05,
        Omega_c=0.25,
        h=0.67,
        n_s=0.9645,
        A_s=2.0E-9,
        m_nu=0.00001,
        m_nu_type='equal',
        pk_linear={'a': 1./(1.+zs[::-1]),
                   'k': ks,
                   'delta_matter:delta_matter': pks[::-1, :]})
    cosmo.compute_growth()

    # Halo model setup
    mass_def = ccl.halos.MassDef(200, 'critical')
    cm = ccl.halos.ConcentrationDuffy08(mass_def)
    hmf = ccl.halos.MassFuncTinker08(cosmo, mass_def=mass_def)
    hbf = ccl.halos.HaloBiasTinker10(cosmo, mass_def=mass_def)
    hmc = ccl.halos.HMCalculator(cosmo, hmf, hbf, mass_def)
    prf = ccl.halos.HaloProfileHOD(cm,
                                   lMmin_0=np.log10(10.**lMcut/cosmo['h']),
                                   siglM_0=sigma_Ncen,
                                   lM0_0=np.log10(10.**lMcut/cosmo['h']),
                                   lM1_0=np.log10(10.**lM1/cosmo['h']),
                                   alpha_0=alp_Nsat,
                                   bg_0=rgs,
                                   bmax_0=rmax)
    prf2pt = ccl.halos.Profile2ptHOD()
    # P(k)
    k_arr = np.geomspace(1E-4, 1E2, 512)
    a_arr = np.linspace(0.8, 1, 32)
    pk_hod = ccl.halos.halomod_Pk2D(cosmo, hmc, prf, prof_2pt=prf2pt,
                                    normprof1=True, lk_arr=np.log(k_arr),
                                    a_arr=a_arr)
    # C_ell
    tr = ccl.NumberCountsTracer(cosmo, False, (z_arr, dndz),
                                (z_arr, np.ones(len(dndz))))
    cl_hod = ccl.angular_cl(cosmo, tr, tr, l_bm, p_of_k_a=pk_hod)

    assert np.all(np.fabs(cl_hod/cl_bm-1) < 0.005)
