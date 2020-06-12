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
    c = ccl.halos.ConcentrationConstant(c=concentration, mdef=mdef)
    p = ccl.halos.HaloProfileHernquist(c, truncated=False)

    prof = p.real(COSMO, r, halomass, a, mdef)

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
    c = ccl.halos.ConcentrationConstant(c=concentration, mdef=mdef)
    mdef = ccl.halos.MassDef(mDelta, 'matter',
                             c_m_relation=c)
    p = ccl.halos.HaloProfileEinasto(c, truncated=False)

    prof = p.real(COSMO, r, halomass, a, mdef)

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
    c = ccl.halos.ConcentrationConstant(c=concentration, mdef=mdef)
    p = ccl.halos.HaloProfileNFW(c, truncated=False)

    prof = p.real(COSMO, r, halomass, a, mdef)

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
    c = ccl.halos.ConcentrationConstant(c=concentration, mdef=mdef)

    if model == 'nfw':
        p = ccl.halos.HaloProfileNFW(c, truncated=False)
        prof = p.real(COSMO, r, halomass, a, mdef)
    elif model == 'projected_nfw':
        p = ccl.halos.HaloProfileNFW(c, truncated=False,
                                     projected_analytic=True)
        prof = p.projected(COSMO, r, halomass, a, mdef)
    elif model == 'einasto':
        mdef = ccl.halos.MassDef(halomassdef, 'matter', c_m_relation=c)
        p = ccl.halos.HaloProfileEinasto(c, truncated=False)
        prof = p.real(COSMO, r, halomass, a, mdef)
    elif model == 'hernquist':
        p = ccl.halos.HaloProfileHernquist(c, truncated=False)
        prof = p.real(COSMO, r, halomass, a, mdef)

    tol = np.clip(np.abs(HALOPROFILE_TOLERANCE * data[:, 1]), 1e-12, np.inf)
    err = np.abs(prof - data[:, 1])
    assert np.all(err <= tol)
