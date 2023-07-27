import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
HMFS = [ccl.halos.MassFuncPress74,
        ccl.halos.MassFuncSheth99,
        ccl.halos.MassFuncJenkins01,
        ccl.halos.MassFuncAngulo12,
        ccl.halos.MassFuncTinker08,
        ccl.halos.MassFuncTinker10,
        ccl.halos.MassFuncWatson13,
        ccl.halos.MassFuncDespali16,
        ccl.halos.MassFuncBocquet16,
        ccl.halos.MassFuncBocquet20]
MS = [1E13, [1E12, 1E15], np.array([1E12, 1E15])]
MFOF = ccl.halos.MassDef('fof', 'matter')
MVIR = ccl.halos.MassDef('vir', 'critical')
M100 = ccl.halos.MassDef(100, 'matter')
M200c = ccl.halos.MassDef(200, 'critical')
M200m = ccl.halos.MassDef(200, 'matter')
M500c = ccl.halos.MassDef(500, 'critical')
M500m = ccl.halos.MassDef(500, 'matter')
MDFS = [MVIR, MVIR, MVIR, MVIR,
        MFOF, MFOF, MVIR, MFOF, MFOF, MFOF]


@pytest.mark.parametrize('nM_class', HMFS)
def test_nM_subclasses_smoke(nM_class):
    nM = nM_class()
    for m in MS:
        n = nM(COSMO, m, 0.9)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)


@pytest.mark.parametrize('nM_pair', zip(HMFS, MDFS))
def test_nM_mdef_raises(nM_pair):
    nM_class, mdef = nM_pair
    with pytest.raises(ValueError):
        nM_class(mass_def=mdef)


@pytest.mark.parametrize('nM_class', [ccl.halos.MassFuncTinker08,
                                      ccl.halos.MassFuncTinker10])
def test_nM_mdef_bad_delta(nM_class):
    with pytest.raises(ValueError):
        nM_class(mass_def=MFOF)


@pytest.mark.parametrize('nM_class', [ccl.halos.MassFuncTinker08,
                                      ccl.halos.MassFuncTinker10])
def test_nM_SO_allgood(nM_class):
    nM = nM_class(mass_def=MVIR)
    for m in MS:
        n = nM(COSMO, m, 0.9)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)


def test_nM_despali_smoke():
    nM = ccl.halos.MassFuncDespali16(ellipsoidal=True)
    for m in MS:
        n = nM(COSMO, m, 0.9)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)


@pytest.mark.parametrize('mdef', [MFOF, M200m])
def test_nM_watson_smoke(mdef):
    nM = ccl.halos.MassFuncWatson13(mass_def=mdef)
    for m in MS:
        n = nM(COSMO, m, 0.9)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)
    for m in MS:
        n = nM(COSMO, m, 0.1)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)


@pytest.mark.parametrize('with_hydro', [True, False])
def test_nM_bocquet_smoke(with_hydro):
    with pytest.raises(ValueError):
        ccl.halos.MassFuncBocquet16(mass_def=M500m, hydro=with_hydro)

    for md in [M500c, M200c, M200m]:
        nM = ccl.halos.MassFuncBocquet16(mass_def=md, hydro=with_hydro)
        for m in MS:
            n = nM(COSMO, m, 0.9)
            assert np.all(np.isfinite(n))
            assert np.shape(n) == np.shape(m)


@pytest.mark.parametrize('name', ['Press74', 'Tinker08',
                                  'Despali16', 'Angulo12'])
def test_nM_from_string(name):
    nM_class = ccl.halos.MassFunc.from_name(name)
    nM = nM_class()
    for m in MS:
        n = nM(COSMO, m, 0.9)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)


def test_nM_from_string_raises():
    with pytest.raises(KeyError):
        ccl.halos.MassFunc.from_name('Tinker09')


@pytest.mark.parametrize('mf', [ccl.halos.MassFuncTinker08,
                                ccl.halos.MassFuncTinker10])
def test_nM_tinker_crit(mf):
    a = 0.5
    om = ccl.omega_x(COSMO, a, 'matter')
    oc = ccl.omega_x(COSMO, a, 'critical')
    delta_c = 500.
    delta_m = delta_c * oc / om
    mdef_c = ccl.halos.MassDef(delta_c, 'critical')
    mdef_m = ccl.halos.MassDef(delta_m, 'matter')
    nM_c = mf(mass_def=mdef_c)
    nM_m = mf(mass_def=mdef_m)
    assert np.allclose(nM_c(COSMO, 1E13, a), nM_m(COSMO, 1E13, a))


def test_nM_tinker10_norm():
    from scipy.integrate import quad

    md = ccl.halos.MassDef(300, rho_type='matter')
    mf = ccl.halos.MassFuncTinker10(mass_def=md, norm_all_z=True)
    bf = ccl.halos.HaloBiasTinker10(mass_def=md)

    def integrand(lnu, z):
        nu = np.exp(lnu)
        a = 1./(1+z)
        gnu = mf._get_fsigma(COSMO, 1.686/nu, a, 1)
        bnu = bf._get_bsigma(COSMO, bf.dc/nu, a)
        return gnu*bnu

    def norm(z):
        return quad(integrand, -13, 2, args=(z,))[0]

    zs = np.linspace(0, 1, 4)
    ns = np.array([norm(z) for z in zs])
    assert np.all(np.fabs(ns-1) < 0.005)


def test_mass_function_mass_def_strict_always_raises():
    # Verify that when the property `_mass_def_strict_always` is set to True,
    # the `mass_def_strict` check cannot be relaxed.
    mdef = ccl.halos.MassDef(400, "critical")
    with pytest.raises(ValueError):
        ccl.halos.MassFuncBocquet16(mass_def=mdef, mass_def_strict=False)


def test_nM_bocquet20_raises():
    mf = ccl.halos.MassFuncBocquet20(mass_def='200c')
    Ms = np.geomspace(1E12, 1E17, 128)

    # Need A_s
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                          A_s=2E-9, n_s=0.96)
    with pytest.raises(ValueError):
        mf(cosmo, Ms, 1.0)

    # Cosmo parameters out of bounds
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                          sigma8=0.8, n_s=2.0)
    with pytest.raises(ValueError):
        mf(cosmo, Ms, 1.0)

    # Redshift out of bounds
    with pytest.raises(ValueError):
        mf(cosmo, Ms, 0.3)
