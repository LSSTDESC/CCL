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

hmd_vir = ccl.halos.MassDefVir
hmd_200m = ccl.halos.MassDef200m
hmd_200c = ccl.halos.MassDef200c
hmd_500c = ccl.halos.MassDef500c


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


def get_cm(name, mass_def):
    return ccl.halos.Concentration.create_instance(name, mass_def=mass_def)


def test_mdef_concentration():
    cs_200m_dh = get_cm("Duffy08", "200m")(cosmo, Ms, 1)
    cs_200m_bh = get_cm("Bhattacharya13", "200m")(cosmo, Ms, 1)
    assert np.allclose(cs_200m_dh, cs_200m_d, atol=0, rtol=1e-6)
    assert np.allclose(cs_200m_bh, cs_200m_b, atol=0, rtol=3e-3)

    cs_200c_dh = get_cm("Duffy08", "200c")(cosmo, Ms, 1)
    cs_200c_bh = get_cm("Bhattacharya13", "200c")(cosmo, Ms, 1)
    assert np.allclose(cs_200c_dh, cs_200c_d, atol=0, rtol=1e-6)
    assert np.allclose(cs_200c_bh, cs_200c_b, atol=0, rtol=3e-3)

    cs_vir_kh = get_cm("Klypin11", "vir")(cosmo, Ms, 1)
    cs_vir_bh = get_cm("Bhattacharya13", "vir")(cosmo, Ms, 1)
    assert np.allclose(cs_vir_kh, cs_vir_k, atol=0, rtol=1e-6)
    assert np.allclose(cs_vir_bh, cs_vir_b, atol=0, rtol=3e-3)

    cs_200c_ph = get_cm("Prada12", "200c")(cosmo, Ms, 1)
    cs_200c_dih = get_cm("Diemer15", "200c")(cosmo, Ms, 1)
    assert np.allclose(cs_200c_ph, cs_200c_p, atol=0, rtol=1e-3)
    assert np.allclose(cs_200c_dih, cs_200c_di, atol=0, rtol=1e-2)


def test_mdef_translate_mass():
    translator = ccl.halos.mass_translator(mass_in=hmd_200m, mass_out=hmd_500c,
                                           concentration="Duffy08")
    Ms_500c_h = translator(cosmo, Ms, 1.)
    assert np.all(np.fabs(Ms_500c_h/Ms_500c-1) < 1E-6)
