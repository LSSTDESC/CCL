import numpy as np
import pytest

import pyccl as ccl

HALOPROFILE_TOLERANCE = 1E-3

COSMO = ccl.Cosmology(
    Omega_b=0.0486,
    Omega_c=0.2603,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1,
    wa=0,
    m_nu=0,
    m_nu_type='normal',
    Neff=3.046,
    Omega_k=0,
    transfer_function='eisenstein_hu',
    mass_function='shethtormen')


def test_profile_Hernquist():
    data = np.loadtxt("./benchmarks/data/haloprofile_hernquist_colossus.txt")
    a = 1.0
    halomass = 6e13
    concentration = 5
    mDelta = 200
    rmin = 0.01
    rmax = 100
    r = np.exp(
        np.log(rmin) +
        np.log(rmax/rmin) * np.arange(data.shape[0]) / (data.shape[0]-1))

    mdef = ccl.halos.MassDef(mDelta, 'matter')
    c = ccl.halos.ConcentrationConstant(c=concentration, mass_def=mdef)
    p = ccl.halos.HaloProfileHernquist(concentration=c, truncated=False)

    prof = p.real(COSMO, r, halomass, a, mass_def=mdef)

    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 1]), 1e-12, np.inf)
    err = np.abs(prof - data[:, 1])
    assert np.all(err <= tol)


def test_profile_Einasto():
    data = np.loadtxt("./benchmarks/data/haloprofile_einasto_colossus.txt")
    a = 1.0
    halomass = 6e13
    concentration = 5
    mDelta = 200
    rmin = 0.01
    rmax = 100
    r = np.exp(
        np.log(rmin) +
        np.log(rmax/rmin) * np.arange(data.shape[0]) / (data.shape[0]-1))

    mdef = ccl.halos.MassDef(mDelta, 'matter')
    c = ccl.halos.ConcentrationConstant(c=concentration, mass_def=mdef)
    mdef = ccl.halos.MassDef(mDelta, 'matter', concentration=c)
    p = ccl.halos.HaloProfileEinasto(concentration=c, truncated=False)

    prof = p.real(COSMO, r, halomass, a, mass_def=mdef)

    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 1]), 1e-12, np.inf)
    err = np.abs(prof - data[:, 1])
    assert np.all(err <= tol)


def test_profile_NFW():
    data = np.loadtxt("./benchmarks/data/haloprofile_nfw_colossus.txt")
    a = 1.0
    halomass = 6e13
    concentration = 5
    mDelta = 200
    rmin = 0.01
    rmax = 100
    r = np.exp(
        np.log(rmin) +
        np.log(rmax/rmin) * np.arange(data.shape[0]) / (data.shape[0]-1))

    mdef = ccl.halos.MassDef(mDelta, 'matter')
    c = ccl.halos.ConcentrationConstant(c=concentration, mass_def=mdef)
    p = ccl.halos.HaloProfileNFW(concentration=c, truncated=False)

    prof = p.real(COSMO, r, halomass, a, mass_def=mdef)

    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 1]), 1e-12, np.inf)
    err = np.abs(prof - data[:, 1])
    assert np.all(err <= tol)


@pytest.mark.parametrize(
    'model', ['nfw', 'projected_nfw', 'einasto', 'hernquist'])
def test_haloprofile(model):

    data = np.loadtxt("./benchmarks/data/haloprofile_%s_colossus.txt" % model)
    a = 1.0
    concentration = 5
    halomass = 6e13
    halomassdef = 200
    rmin = 0.01
    rmax = 100
    r = np.exp(
        np.log(rmin) +
        np.log(rmax/rmin) * np.arange(data.shape[0]) / (data.shape[0]-1))

    mdef = ccl.halos.MassDef(halomassdef, 'matter')
    c = ccl.halos.ConcentrationConstant(c=concentration, mass_def=mdef)

    if model == 'nfw':
        p = ccl.halos.HaloProfileNFW(concentration=c, truncated=False)
        prof = p.real(COSMO, r, halomass, a, mass_def=mdef)
    elif model == 'projected_nfw':
        p = ccl.halos.HaloProfileNFW(concentration=c, truncated=False,
                                     projected_analytic=True)
        prof = p.projected(COSMO, r, halomass, a, mass_def=mdef)
    elif model == 'einasto':
        mdef = ccl.halos.MassDef(halomassdef, 'matter', concentration=c)
        p = ccl.halos.HaloProfileEinasto(concentration=c, truncated=False)
        prof = p.real(COSMO, r, halomass, a, mass_def=mdef)
    elif model == 'hernquist':
        p = ccl.halos.HaloProfileHernquist(concentration=c, truncated=False)
        prof = p.real(COSMO, r, halomass, a, mass_def=mdef)

    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 1]), 1e-12, np.inf)
    err = np.abs(prof - data[:, 1])
    assert np.all(err <= tol)


def test_weak_lensing_functions():
    data = np.loadtxt("./benchmarks/data/haloprofile_nfw_wl_numcosmo.txt")
    z_lens = 1.0
    z_source = 2.0
    a_lens = 1.0 / (1.0 + z_lens)
    a_source = 1.0 / (1.0 + z_source)
    halomass = 1.0e15
    concentration = 5
    mDelta = "vir"  # 200
    rmin = 0.01
    rmax = 100
    r = np.exp(
        np.log(rmin) +
        np.log(rmax/rmin) * np.arange(data.shape[0]) / (data.shape[0]-1))

    r_al = r / a_lens
    len_r = len(r)
    a_source = [a_source]*len_r

    mdef = ccl.halos.MassDef(mDelta, 'matter')
    c = ccl.halos.ConcentrationConstant(c=concentration, mass_def=mdef)
    p = ccl.halos.HaloProfileNFW(
        concentration=c,
        truncated=False, projected_analytic=True, cumul2d_analytic=True
    )

    kappa = p.convergence(COSMO, r_al, halomass,
                          a_lens=a_lens, a_source=a_source, mass_def=mdef)
    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 1]), 1e-12, np.inf)
    err_kappa = np.abs(kappa - data[:, 1])
    assert np.all(err_kappa <= tol)

    gamma = p.shear(COSMO, r_al, halomass,
                    a_lens=a_lens, a_source=a_source, mass_def=mdef)
    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 2]), 1e-12, np.inf)
    err_gamma = np.abs(gamma - data[:, 2])
    assert np.all(err_gamma <= tol)

    gt = p.reduced_shear(COSMO, r_al, halomass,
                         a_lens=a_lens, a_source=a_source, mass_def=mdef)
    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 3]), 1e-12, np.inf)
    err_gt = np.abs(gt - data[:, 3])
    assert np.all(err_gt <= tol)

    mu = p.magnification(COSMO, r_al, halomass,
                         a_lens=a_lens, a_source=a_source, mass_def=mdef)
    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 4]), 1e-12, np.inf)
    err_mu = np.abs(mu - data[:, 4])
    assert np.all(err_mu <= tol)


def test_satellite_shear_profile():
    k_arr, gamma_k_CosmoSIS = np.loadtxt(
        './benchmarks/data/satellite_shear_profile_cosmoSIS_dev.txt',
        unpack=True)
    cosmo = ccl.Cosmology(Omega_c=0.278 - 0.0391,
                          Omega_b=0.0391, h=0.7,
                          A_s=1.70150098142e-09, n_s=0.978)

    z_eval = 0.2
    mass_eval = 1e15  # Msun

    hmd_200m = ccl.halos.MassDef200m()
    cM = ccl.halos.ConcentrationDuffy08(hmd_200m)

    sat_gamma_HOD_simps = ccl.halos.SatelliteShearHOD(cM, lmax=6,
                                                      a1h=0.000989)
    gamma_k_CCL = sat_gamma_HOD_simps._usat_fourier(cosmo,
                                                    k_arr,
                                                    mass_eval,
                                                    1 / (1 + z_eval),
                                                    hmd_200m)
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

    hmd_200m = ccl.halos.MassDef200m()
    cM = ccl.halos.ConcentrationDuffy08(hmd_200m)
    nM = ccl.halos.MassFuncTinker08(cosmo, mass_def=hmd_200m)
    bM = ccl.halos.HaloBiasTinker10(cosmo, mass_def=hmd_200m)
    hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd_200m, nlog10M=64)

    sat_gamma_HOD = ccl.halos.SatelliteShearHOD(cM)
    NFW = ccl.halos.HaloProfileNFW(cM, truncated=True, fourier_analytic=True)

    pk_GI_1h = ccl.halos.halomod_Pk2D(cosmo, hmc, NFW,
                                      normprof1=True,
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
    assert np.all(np.fabs(cl_II_1h / cl_II_benchmark - 1) < 0.01)

    # Below I am benchmarking the normalization factor of the
    # delta-E power spectrum Eq. (17) in Fortuna et al. 2021.
    # Specifically, the term f_s(z)*<N_s|M>/n_s(z).
    mass_benchmark, norm_benchmark = np.loadtxt(
        './benchmarks/data/IA_halomodel_norm_term.dat', unpack=True)
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, Omega_k=0,
                          sigma8=0.81, n_s=0.96, h=1.)
    cosmo.compute_sigma()
    hmc = ccl.halos.HMCalculator(cosmo, nM, bM, hmd_200m)
    Ns_M_mean = sat_gamma_HOD._Ns(mass_benchmark, 1.)
    norm = Ns_M_mean * sat_gamma_HOD._get_prefactor(cosmo, 1, hmc)
    assert np.all(np.fabs(norm / norm_benchmark - 1) < 0.005)
