import pytest
import numpy as np
import pyccl as ccl

COSMO = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, Omega_g=0, Omega_k=0,
                      h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                      w0=-1, wa=0, T_CMB=2.7)


def test_mdef_eq():
    hmd_200m = ccl.halos.MassDef200m()
    hmd_200m_b = ccl.halos.MassDef(200, 'matter')
    assert hmd_200m == hmd_200m_b


def test_concentration_translation():
    c_old = np.array([9., 10., 11.])
    Delta_old = 200.

    # No change expected
    Delta_new = 200.
    c_new = ccl.halos.massdef.convert_concentration(COSMO,
                                                    c_old, Delta_old,
                                                    Delta_new)
    assert np.all(c_old == c_new)

    # Test against numerical solutions from Mathematica.
    Delta_new = 500.
    c_new = ccl.halos.massdef.convert_concentration(COSMO,
                                                    c_old, Delta_old,
                                                    Delta_new)
    c_new_expected = np.array([6.12194, 6.82951, 7.53797])
    assert np.all(np.fabs(c_new/c_new_expected-1) < 1E-4)


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


def test_get_concentration():
    hmd = ccl.halos.MassDef200m()
    for M in [1E12, [1E12, 2E12],
              np.array([1E12, 2E12])]:
        c = hmd._get_concentration(COSMO, M, 1.)
        assert np.all(np.isfinite(c))
        assert np.shape(c) == np.shape(M)


def test_get_concentration_raises():
    hmd = ccl.halos.MassDef(200, 'matter')
    with pytest.raises(RuntimeError):
        hmd._get_concentration(COSMO, 1E12, 1.)


def test_translate_mass():
    hmd = ccl.halos.MassDef200m()
    hmdb = ccl.halos.MassDef200c()
    for M in [1E12, [1E12, 2E12],
              np.array([1E12, 2E12])]:
        m = hmd.translate_mass(COSMO, M,
                               1., hmdb)
        assert np.all(np.isfinite(m))
        assert np.shape(m) == np.shape(M)


def test_translate_mass_raises():
    hmd = ccl.halos.MassDef(200, 'matter')
    hmdb = ccl.halos.MassDef(200, 'critical')
    with pytest.raises(RuntimeError):
        hmd.translate_mass(COSMO, 1E12,
                           1., hmdb)


@pytest.mark.parametrize('scls', [ccl.halos.MassDef200m,
                                  ccl.halos.MassDef200c,
                                  ccl.halos.MassDefVir])
def test_subclasses_smoke(scls):
    hmd = scls()
    assert np.isfinite(hmd.get_Delta(COSMO, 1.))
