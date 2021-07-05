import os
import numpy as np
import pyccl as ccl

cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, Omega_g=0, Omega_k=0,
                      h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                      w0=-1, wa=0, T_CMB=2.7255,
                      transfer_function='eisenstein_hu')
dirdat = os.path.join(os.path.dirname(__file__), 'data')
Ms, Rs_200m, Rs_500c, Ms_500c = np.loadtxt(os.path.join(dirdat,
                                                        'mdef_bm.txt'),
                                           unpack=True)
dc = np.loadtxt(os.path.join(dirdat, 'conc_bm.txt'), unpack=True)
Ms = dc[0]
cs_200m_d = dc[1]
cs_200c_d = dc[2]
cs_200m_b = dc[3]
cs_200c_b = dc[4]
cs_vir_k = dc[5]
cs_vir_b = dc[6]
cs_200c_p = dc[7]
cs_200c_di = dc[8]

hmd_vir = ccl.halos.MassDefVir()
hmd_vir_b = ccl.halos.MassDefVir('Bhattacharya13')
hmd_200m = ccl.halos.MassDef200m()
hmd_200m_b = ccl.halos.MassDef200m('Bhattacharya13')
hmd_200c = ccl.halos.MassDef200c()
hmd_200c_b = ccl.halos.MassDef200c('Bhattacharya13')
hmd_200c_p = ccl.halos.MassDef200c('Prada12')
hmd_200c_di = ccl.halos.MassDef200c('Diemer15')
hmd_500c = ccl.halos.MassDef(500, 'critical')


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
    # Duffy 200 matter
    cs_200m_dh = hmd_200m._get_concentration(cosmo, Ms, 1.)
    # Bhattacharya 200 matter
    cs_200m_bh = hmd_200m_b._get_concentration(cosmo, Ms, 1.)
    # Duffy 200 critical
    cs_200c_dh = hmd_200c._get_concentration(cosmo, Ms, 1.)
    # Bhattacharya 200 critical
    cs_200c_bh = hmd_200c_b._get_concentration(cosmo, Ms, 1.)
    # Klypin virial
    cs_vir_kh = hmd_vir._get_concentration(cosmo, Ms, 1.)
    # Bhattacharya virial
    cs_vir_bh = hmd_vir_b._get_concentration(cosmo, Ms, 1.)
    # Prada 200 critical
    cs_200c_ph = hmd_200c_p._get_concentration(cosmo, Ms, 1.)
    # Diemer 200 critical
    cs_200c_dih = hmd_200c_di._get_concentration(cosmo, Ms, 1.)
    assert np.all(np.fabs(cs_200m_dh/cs_200m_d-1) < 1E-6)
    assert np.all(np.fabs(cs_200c_dh/cs_200c_d-1) < 1E-6)
    assert np.all(np.fabs(cs_200m_bh/cs_200m_b-1) < 3E-3)
    assert np.all(np.fabs(cs_200c_bh/cs_200c_b-1) < 3E-3)
    assert np.all(np.fabs(cs_200c_ph/cs_200c_p-1) < 1E-3)
    assert np.all(np.fabs(cs_200c_dih/cs_200c_di-1) < 1E-2)
    assert np.all(np.fabs(cs_vir_kh/cs_vir_k-1) < 1E-3)
    assert np.all(np.fabs(cs_vir_bh/cs_vir_b-1) < 3E-3)


def test_mdef_translate_mass():
    Ms_500c_h = hmd_200m.translate_mass(cosmo, Ms, 1., hmd_500c)
    assert np.all(np.fabs(Ms_500c_h/Ms_500c-1) < 1E-6)
