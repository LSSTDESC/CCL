import numpy as np
import pyccl as ccl


def test_satellite_shear_profile():
    k_arr, gamma_k_CosmoSIS = np.loadtxt(
        './benchmarks/data/satellite_shear_profile_cosmoSIS_dev.txt',
        unpack=True)
    cosmo = ccl.Cosmology(Omega_c=0.278 - 0.0391,
                          Omega_b=0.0391, h=0.7,
                          A_s=1.70150098142e-09, n_s=0.978)

    z_eval = 0.2
    mass_eval = 1e15  # Msun

    cM = ccl.halos.ConcentrationDuffy08(mass_def='200m')

    sat_gamma_HOD_simps = ccl.halos.SatelliteShearHOD(concentration=cM, lmax=6,
                                                      a1h=0.000989,
                                                      mass_def='200m')
    gamma_k_CCL = sat_gamma_HOD_simps._usat_fourier(cosmo, k_arr, mass_eval,
                                                    1 / (1 + z_eval))
    # This benchmark is not accurate enough and only serves as a
    # qualitative test, which is why the accuracy criterion is so low.
    assert np.all(np.fabs(-gamma_k_CCL / gamma_k_CosmoSIS - 1) < 0.8)

    # An in-house benchmark test has been developed, testing the accuracy
    # of the FFTLog method against a highly accurate integration method as
    # described in benchmarks/data/codes/halomod_IA_FFTLog_accuracy.ipynb.
    l_arr, cl_GI_benchmark, cl_II_benchmark = np.loadtxt(
        './benchmarks/data/IA_halomodel_Cell_test.txt',
        delimiter=',', unpack=True)
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045,
                          h=0.67, sigma8=0.83, n_s=0.96)
    k_arr = np.geomspace(1E-3, 1e2, 128)  # For evaluating
    a_arr = np.linspace(0.1, 1, 16)

    cM = ccl.halos.ConcentrationDuffy08(mass_def="200m")
    nM = ccl.halos.MassFuncTinker08(mass_def="200m")
    bM = ccl.halos.HaloBiasTinker10(mass_def="200m")
    hmc = ccl.halos.HMCalculator(mass_function=nM,
                                 halo_bias=bM,
                                 mass_def="200m", nM=64)

    sat_gamma_HOD = ccl.halos.SatelliteShearHOD(concentration=cM,
                                                mass_def='200m')
    NFW = ccl.halos.HaloProfileNFW(concentration=cM, truncated=True,
                                   fourier_analytic=True, mass_def='200m')

    pk_GI_1h = ccl.halos.halomod_Pk2D(cosmo, hmc, NFW,
                                      prof2=sat_gamma_HOD,
                                      get_2h=False,
                                      lk_arr=np.log(k_arr),
                                      a_arr=a_arr)
    pk_II_1h = ccl.halos.halomod_Pk2D(cosmo, hmc, sat_gamma_HOD,
                                      get_2h=False,
                                      lk_arr=np.log(k_arr),
                                      a_arr=a_arr)

    z_arr = np.linspace(0., 3., 256)
    z0 = 0.1
    pz = 1. / (2. * z0) * (z_arr / z0) ** 2. * np.exp(-z_arr / z0)

    b_IA = np.ones(len(z_arr))  # A_IA = 1 in the NLA model
    ia_tracer = ccl.WeakLensingTracer(cosmo, dndz=(z_arr, pz), has_shear=False,
                                      ia_bias=(z_arr, b_IA), use_A_ia=False)
    wl_tracer = ccl.WeakLensingTracer(cosmo, dndz=(z_arr, pz))

    cl_GI_1h = ccl.angular_cl(cosmo, wl_tracer, ia_tracer, l_arr,
                              p_of_k_a=pk_GI_1h)
    cl_II_1h = ccl.angular_cl(cosmo, ia_tracer, ia_tracer, l_arr,
                              p_of_k_a=pk_II_1h)

    assert np.all(np.fabs(cl_GI_1h / cl_GI_benchmark - 1) < 0.01)
    assert np.all(np.fabs(cl_II_1h / cl_II_benchmark - 1) < 0.02)

    # Below we are benchmarking the normalization factor of the
    # delta-E power spectrum Eq. (17) in Fortuna et al. 2021.
    # Specifically, the term f_s(z)*<N_s|M>/n_s(z).
    mass_benchmark, norm_benchmark = np.loadtxt(
        './benchmarks/data/IA_halomodel_norm_term.dat', unpack=True)
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, Omega_k=0,
                          sigma8=0.81, n_s=0.96, h=1.)
    cosmo.compute_sigma()
    Ns_M_mean = sat_gamma_HOD._Ns(mass_benchmark, 1.)
    norm = Ns_M_mean / sat_gamma_HOD.get_normalization(cosmo, 1, hmc=hmc)
    assert np.all(np.fabs(norm / norm_benchmark - 1) < 0.005)
