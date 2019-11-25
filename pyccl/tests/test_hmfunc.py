import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
HMFS = [ccl.halos.MassFuncPress74,
        ccl.halos.MassFuncSheth99,
        ccl.halos.MassFuncJenkins01,
        ccl.halos.MassFuncAngulo12,
        ccl.halos.MassFuncTinker08,
        ccl.halos.MassFuncTinker10,
        ccl.halos.MassFuncWatson13,
        ccl.halos.MassFuncDespali16,
        ccl.halos.MassFuncBocquet16]
MS = [1E13, [1E12, 1E15], np.array([1E12, 1E15])]
MFOF = ccl.halos.MassDef('fof', 'matter')
MVIR = ccl.halos.MassDef('vir', 'critical')
M100 = ccl.halos.MassDef(100, 'matter')
M200c = ccl.halos.MassDef(200, 'critical')
M200m = ccl.halos.MassDef(200, 'matter')
M500c = ccl.halos.MassDef(500, 'critical')
M500m = ccl.halos.MassDef(500, 'matter')
MDFS = [MVIR, MVIR, MVIR, MVIR,
        MFOF, MFOF, MVIR, MFOF, MFOF]


@pytest.mark.parametrize('nM_class', HMFS)
def test_nM_subclasses_smoke(nM_class):
    nM = nM_class(COSMO)
    for m in MS:
        n = nM.get_mass_function(COSMO, m, 0.9)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)


@pytest.mark.parametrize('nM_pair', zip(HMFS, MDFS))
def test_nM_mdef_raises(nM_pair):
    nM_class, mdef = nM_pair
    with pytest.raises(ValueError):
        nM_class(COSMO, mdef)


@pytest.mark.parametrize('nM_class', [ccl.halos.MassFuncTinker08,
                                      ccl.halos.MassFuncTinker10])
def test_nM_mdef_bad_delta(nM_class):
    with pytest.raises(ValueError):
        nM_class(COSMO, M100)


def test_nM_despali_smoke():
    nM = ccl.halos.MassFuncDespali16(COSMO,
                                     ellipsoidal=True)
    for m in MS:
        n = nM.get_mass_function(COSMO, m, 0.9)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)


@pytest.mark.parametrize('mdef', [MFOF, M200m])
def test_nM_watson_smoke(mdef):
    nM = ccl.halos.MassFuncWatson13(COSMO,
                                    mdef)
    for m in MS:
        n = nM.get_mass_function(COSMO, m, 0.9)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)
    for m in MS:
        n = nM.get_mass_function(COSMO, m, 0.1)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)


@pytest.mark.parametrize('with_hydro', [True, False])
def test_nM_bocquet_smoke(with_hydro):
    with pytest.raises(ValueError):
        ccl.halos.MassFuncBocquet16(COSMO, M500m,
                                    hydro=with_hydro)

    for md in [M500c, M200c, M200m]:
        nM = ccl.halos.MassFuncBocquet16(COSMO, md,
                                         hydro=with_hydro)
        for m in MS:
            n = nM.get_mass_function(COSMO, m, 0.9)
            assert np.all(np.isfinite(n))
            assert np.shape(n) == np.shape(m)


@pytest.mark.parametrize('name', ['Press74', 'Tinker08',
                                  'Despali16', 'Angulo12'])
def test_nM_from_string(name):
    nM_class = ccl.halos.mass_function_from_name(name)
    nM = nM_class(COSMO)
    for m in MS:
        n = nM.get_mass_function(COSMO, m, 0.9)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)


def test_nM_from_string_raises():
    with pytest.raises(ValueError):
        ccl.halos.mass_function_from_name('Tinker09')


def test_nM_default():
    nM = ccl.halos.MassFunc(COSMO)
    with pytest.raises(NotImplementedError):
        nM._get_fsigma(COSMO, 1., 1., 8)

    M_in = 1E12
    lM_out = nM._get_consistent_mass(COSMO,
                                     M_in, 1., nM.mdef)
    assert np.fabs(np.log10(M_in) - lM_out) < 1E-10
