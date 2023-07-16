import pytest
import numpy as np
import pyccl as ccl

COSMO = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, Omega_g=0, Omega_k=0,
                      h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                      w0=-1, wa=0, T_CMB=2.7)


def test_mdef_eq():
    hmd_200m = ccl.halos.MassDef200m
    hmd_200m_b = ccl.halos.MassDef(200, 'matter')
    assert hmd_200m == hmd_200m_b


def test_concentration_translation():
    c_old = np.array([9., 10., 11.])
    Delta_old = 200.

    # No change expected
    Delta_new = 200.
    c_new = ccl.halos.massdef.convert_concentration(
        COSMO, c_old=c_old, Delta_old=Delta_old, Delta_new=Delta_new)
    assert np.all(c_old == c_new)

    c_new = ccl.halos.massdef.convert_concentration(
        COSMO, c_old=c_old, Delta_old=Delta_old, Delta_new=Delta_new,
        model="Einasto", alpha=0.25)
    assert np.all(c_old == c_new)

    c_new = ccl.halos.massdef.convert_concentration(
        COSMO, c_old=c_old, Delta_old=Delta_old, Delta_new=Delta_new,
        model="Hernquist")
    assert np.all(c_old == c_new)

    with pytest.raises(ValueError):
        c_new = ccl.halos.massdef.convert_concentration(
            COSMO, c_old=c_old, Delta_old=Delta_old, Delta_new=Delta_new,
            model="Einasto")

    with pytest.raises(ValueError):
        c_new = ccl.halos.massdef.convert_concentration(
            COSMO, c_old=c_old, Delta_old=Delta_old, Delta_new=Delta_new,
            model="NotValid")

    # Test against numerical solutions from Mathematica.
    Delta_new = 500.
    c_new = ccl.halos.massdef.convert_concentration(
        COSMO, c_old=c_old, Delta_old=Delta_old, Delta_new=Delta_new,
        model="NFW")
    c_new_expected = np.array([6.121936239564, 6.829509425616, 7.537971574322])
    assert np.all(np.fabs(c_new/c_new_expected-1) < 1E-12)

    c_new = ccl.halos.massdef.convert_concentration(
        COSMO, c_old=c_old, Delta_old=Delta_old, Delta_new=Delta_new,
        model="Einasto", alpha=0.25)
    c_new_expected = np.array([6.254469943236, 6.98975396734, 7.72676126963])
    assert np.all(np.fabs(c_new/c_new_expected-1) < 1E-12)

    c_new = ccl.halos.massdef.convert_concentration(
        COSMO, c_old=c_old, Delta_old=Delta_old, Delta_new=Delta_new,
        model="Hernquist")
    c_new_expected = np.array([6.463225850159, 7.199309242066, 7.935520760879])
    assert np.all(np.fabs(c_new/c_new_expected-1) < 1E-12)


def test_init_raises():
    with pytest.raises(ValueError):
        ccl.halos.MassDef('bir', 'matter')

    with pytest.raises(ValueError):
        ccl.halos.MassDef('vir', 'radiation')

    with pytest.raises(ValueError):
        ccl.halos.MassDef(-100, 'matter')


def test_get_Delta():
    for Delta in [200, 500, 121]:
        hmd = ccl.halos.MassDef(Delta, 'critical')
        assert int(hmd.get_Delta(COSMO, 1.)) == Delta

    hmd = ccl.halos.MassDef('vir', 'critical')
    assert np.isfinite(hmd.get_Delta(COSMO, 1.))

    hmd = ccl.halos.MassDef('fof', 'critical')
    with pytest.raises(ValueError):
        hmd.get_Delta(COSMO, 1.)


def test_get_mass():
    hmd = ccl.halos.MassDef(200, 'critical')
    for R in [1., [1., 2.], np.array([1., 2.])]:
        m = hmd.get_mass(COSMO, R, 1.)
        assert np.all(np.isfinite(m))
        assert np.shape(m) == np.shape(R)


def test_get_radius():
    hmd = ccl.halos.MassDef(200, 'critical')
    for M in [1E12, [1E12, 2E12],
              np.array([1E12, 2E12])]:
        r = hmd.get_radius(COSMO, M, 1.)
        assert np.all(np.isfinite(r))
        assert np.shape(r) == np.shape(M)


def test_translate_mass():
    hmd = ccl.halos.MassDef200m
    hmdb = ccl.halos.MassDef200c
    cm = ccl.halos.Concentration.create_instance("Duffy08", mass_def=hmd)
    translator = ccl.halos.mass_translator(mass_in=hmd, mass_out=hmdb,
                                           concentration=cm)
    for M in [1E12, [1E12, 2E12],
              np.array([1E12, 2E12])]:
        m = translator(COSMO, M, 1)
        assert np.all(np.isfinite(m))
        assert np.shape(m) == np.shape(M)


@pytest.mark.parametrize('scls', [ccl.halos.MassDef200m,
                                  ccl.halos.MassDef200c,
                                  ccl.halos.MassDef500c,
                                  ccl.halos.MassDefVir])
def test_subclasses_smoke(scls):
    hmd = scls
    assert np.isfinite(hmd.get_Delta(COSMO, 1.))


@pytest.mark.parametrize('name', ['200m', '200c', '500c', 'vir', '350m'])
def test_massdef_from_string_smoke(name):
    hmd = ccl.halos.MassDef.from_name(name)
    assert np.isfinite(hmd.get_radius(COSMO, 1e14, 1))


def test_massdef_from_string_raises():
    with pytest.raises(ValueError):
        ccl.halos.MassDef.from_name("my_mass_def")


def test_mass_translator():
    # Check that the mass translator complains for inconsistent masses.
    cm = ccl.halos.Concentration.create_instance("Duffy08")
    mdef1 = ccl.halos.MassDef.create_instance("250c")
    mdef2 = ccl.halos.MassDef.create_instance("500c")
    with pytest.raises(ValueError):
        ccl.halos.mass_translator(mass_in=mdef1, mass_out=mdef2,
                                  concentration=cm)

    # Check that if we pass the same mass definition, it returns the same M.
    mdef1 = mdef2 = cm.mass_def
    translator = ccl.halos.mass_translator(mass_in=mdef1, mass_out=mdef2,
                                           concentration=cm)
    cosmo = ccl.CosmologyVanillaLCDM()
    assert translator(cosmo, 1e14, 1) == 1e14
