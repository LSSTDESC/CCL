import os
import numpy as np
import pyccl as ccl

cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, Omega_g=0, Omega_k=0,
                      h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                      w0=-1, wa=0, T_CMB=2.7)
dirdat = os.path.dirname(__file__) + '/data/'
Ms, Rs_200m, Rs_500c, Ms_500c = np.loadtxt(dirdat + 'mdef_bm.txt',
                                           unpack=True)
Ms, cs_200m_d, cs_200c_d, cs_200m_b, cs_200c_b = np.loadtxt(dirdat +
                                                            'conc_bm.txt',
                                                            unpack=True)
hmd_200m = ccl.HMDef200mat()
hmd_200m_b = ccl.HMDef200mat('Bhattacharya11')
hmd_200c = ccl.HMDef200crit()
hmd_200c_b = ccl.HMDef200crit('Bhattacharya11')
hmd_500c = ccl.HMDef(500, 'critical')


def test_mdef_eq():
    hmd_200m_b = ccl.HMDef(200, 'matter')
    assert hmd_200m == hmd_200m_b


def test_mdef_get_radius():
    Rs_200m_h = hmd_200m.get_radius(cosmo, Ms, 1.)
    Rs_500c_h = hmd_500c.get_radius(cosmo, Ms, 1.)
    assert np.all(np.fabs(Rs_200m_h/Rs_200m-1) < 1E-6)
    assert np.all(np.fabs(Rs_500c_h/Rs_500c-1) < 1E-6)


def test_mdef_get_mass():
    Ms_h = hmd_200m.get_mass(cosmo, Rs_200m, 1.)
    assert np.all(np.fabs(Ms_h/Ms-1) < 1E-6)
    Ms_h = hmd_500c.get_mass(cosmo, Rs_500c, 1.)
    assert np.all(np.fabs(Ms_h/Ms-1) < 1E-6)


def test_mdef_concentration():
    cs_200m_dh = hmd_200m.get_concentration(cosmo, Ms, 1.)
    cs_200m_bh = hmd_200m_b.get_concentration(cosmo, Ms, 1.)
    cs_200c_dh = hmd_200c.get_concentration(cosmo, Ms, 1.)
    cs_200c_bh = hmd_200c_b.get_concentration(cosmo, Ms, 1.)
    assert np.all(np.fabs(cs_200m_dh/cs_200m_d-1) < 1E-6)
    assert np.all(np.fabs(cs_200c_dh/cs_200c_d-1) < 1E-6)
    assert np.all(np.fabs(cs_200m_bh/cs_200m_b-1) < 3E-2)
    assert np.all(np.fabs(cs_200c_bh/cs_200c_b-1) < 3E-2)


def test_mdef_translate_mass():
    Ms_500c_h = hmd_200m.translate_mass(cosmo, Ms, 1., hmd_500c)
    assert np.all(np.fabs(Ms_500c_h/Ms_500c-1) < 1E-6)
