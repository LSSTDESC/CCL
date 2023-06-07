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
    mass_split='normal',
    Neff=3.046, T_CMB=2.725,
    Omega_k=0,
    transfer_function='eisenstein_hu')


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
    p = ccl.halos.HaloProfileHernquist(mass_def=mdef, concentration=c,
                                       truncated=False)

    prof = p.real(COSMO, r, halomass, a)

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
    p = ccl.halos.HaloProfileEinasto(mass_def=mdef, concentration=c,
                                     truncated=False)

    prof = p.real(COSMO, r, halomass, a)

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
    p = ccl.halos.HaloProfileNFW(mass_def=mdef, concentration=c,
                                 truncated=False)

    prof = p.real(COSMO, r, halomass, a)

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
        p = ccl.halos.HaloProfileNFW(mass_def=mdef, concentration=c,
                                     truncated=False)
        prof = p.real(COSMO, r, halomass, a)
    elif model == 'projected_nfw':
        p = ccl.halos.HaloProfileNFW(mass_def=mdef, concentration=c,
                                     truncated=False, projected_analytic=True)
        prof = p.projected(COSMO, r, halomass, a)
    elif model == 'einasto':
        mdef = ccl.halos.MassDef(halomassdef, 'matter')
        p = ccl.halos.HaloProfileEinasto(mass_def=mdef, concentration=c,
                                         truncated=False)
        prof = p.real(COSMO, r, halomass, a)
    elif model == 'hernquist':
        p = ccl.halos.HaloProfileHernquist(mass_def=mdef, concentration=c,
                                           truncated=False)
        prof = p.real(COSMO, r, halomass, a)

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
    p = ccl.halos.HaloProfileNFW(mass_def=mdef, concentration=c,
                                 truncated=False, projected_analytic=True,
                                 cumul2d_analytic=True)

    kappa = p.convergence(COSMO, r_al, halomass,
                          a_lens=a_lens, a_source=a_source)
    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 1]), 1e-12, np.inf)
    err_kappa = np.abs(kappa - data[:, 1])
    assert np.all(err_kappa <= tol)

    gamma = p.shear(COSMO, r_al, halomass,
                    a_lens=a_lens, a_source=a_source)
    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 2]), 1e-12, np.inf)
    err_gamma = np.abs(gamma - data[:, 2])
    assert np.all(err_gamma <= tol)

    gt = p.reduced_shear(COSMO, r_al, halomass,
                         a_lens=a_lens, a_source=a_source)
    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 3]), 1e-12, np.inf)
    err_gt = np.abs(gt - data[:, 3])
    assert np.all(err_gt <= tol)

    mu = p.magnification(COSMO, r_al, halomass,
                         a_lens=a_lens, a_source=a_source)
    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 4]), 1e-12, np.inf)
    err_mu = np.abs(mu - data[:, 4])
    assert np.all(err_mu <= tol)
