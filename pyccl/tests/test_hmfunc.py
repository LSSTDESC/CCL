import numpy as np
import pytest
import pyccl as ccl


COSMO = ccl.Cosmology(
    Omega_c=0.27, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96,
    transfer_function='bbks', matter_power_spectrum='linear')
# Dark Emulator needs A_s not sigma8, so cosmological params are redifined.
COSMO_DE = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                         A_s=2.2e-9, n_s=0.96, w0=-1)
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
# These are kinds of slow to initialize, so let's do it only once
MF_emu = ccl.halos.MassFuncBocquet20(mass_def='200c')
# Dark Emulator needs A_s not sigma8, so cosmological params are defined later.
MF_demu = ccl.halos.MassFuncNishimichi19(mass_def='200m', extrapolate=True)


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


def test_nM_bocquet20_compare():
    # Check that the values are sensible (they don't depart from other
    # parametrisations by more than ~10%
    Ms = np.geomspace(1E14, 1E15, 128)
    mf1 = MF_emu
    mf2 = ccl.halos.MassFuncTinker08(mass_def='200c')

    nM1 = mf1(COSMO, Ms, 1.0)
    nM2 = mf2(COSMO, Ms, 1.0)
    assert np.allclose(nM1, nM2, atol=0, rtol=0.1)


def test_nM_bocquet20_raises():
    Ms = np.geomspace(1E12, 1E17, 128)

    # Need sigma8
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                          A_s=2E-9, n_s=0.96)
    with pytest.raises(ValueError):
        MF_emu(cosmo, Ms, 1.0)

    # Cosmo parameters out of bounds
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                          sigma8=0.8, n_s=2.0)
    with pytest.raises(ValueError):
        MF_emu(cosmo, Ms, 1.0)

    # Redshift out of bounds
    with pytest.raises(ValueError):
        MF_emu(cosmo, Ms, 0.3)


def test_nM_nishimichi_smoke():
    for m in MS:
        n = MF_demu(COSMO_DE, m, 0.9)
        assert np.all(np.isfinite(n))
        assert np.shape(n) == np.shape(m)


def test_nM_nishimichi19_compare():
    # Check that the values are sensible (they don't depart from other
    # parametrisations by more than ~4%
    # Msun, under supported range(10^12-16 Msun/h)
    Ms = np.geomspace(1.5E12, 1E15, 128)
    mf1 = MF_demu
    mf2 = ccl.halos.MassFuncTinker10(mass_def='200m')

    nM1 = mf1(COSMO_DE, Ms, 1.0)
    nM2 = mf2(COSMO_DE, Ms, 1.0)
    assert np.allclose(nM1, nM2, atol=0, rtol=0.04)


def test_nM_nishimichi19_raises():
    Ms = np.geomspace(1.5E12, 1E15, 128)
    # mdef raise
    with pytest.raises(ValueError):
        ccl.halos.MassFuncNishimichi19(mass_def=MFOF)

    # contains sigma8 not A_s
    cosmo_s = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                            sigma8=0.8, n_s=0.96)
    with pytest.raises(ValueError):
        MF_demu(cosmo_s, Ms, 1.0)

    # Cosmo parameters out of bounds
    cosmo_wr = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                             A_s=2.2e-9, n_s=2.0)
    with pytest.raises(RuntimeError):
        MF_demu(cosmo_wr, Ms, 1.0)

    # contain unsupported range
    # you can pass it when you set "extrapolate=True" in input of mass
    # function definition even you use unsupported range.
    # default is "extrapolate=False"
    Ms = np.geomspace(1E10, 1E15, 128)
    MF_demu_exFal = ccl.halos.MassFuncNishimichi19(mass_def=M200m,
                                                   extrapolate=False)
    with pytest.raises(RuntimeError):
        MF_demu_exFal(COSMO_DE, Ms, 1.0)
