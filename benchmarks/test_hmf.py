import os
import numpy as np
import pyccl as ccl

# Set cosmology
cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, Omega_g=0, Omega_k=0,
                      h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                      w0=-1, wa=0, T_CMB=2.7255,
                      transfer_function='eisenstein_hu')
# Read data
dirdat = os.path.join(os.path.dirname(__file__), 'data')

# Redshifts
zs = np.array([0., 0.5, 1.])


def test_hmf_despali16():
    hmd = ccl.halos.MassDef('vir', 'critical')
    mf = ccl.halos.MassFuncDespali16(cosmo, hmd)
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_despali16.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_bocquet16():
    hmd = ccl.halos.MassDef200c()
    mf = ccl.halos.MassFuncBocquet16(cosmo, hmd)
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_bocquet16.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_watson13():
    mf = ccl.halos.MassFuncWatson13(cosmo)
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_watson13.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_tinker08():
    mf = ccl.halos.MassFuncTinker08(cosmo)
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_tinker08.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_press74():
    mf = ccl.halos.MassFuncPress74(cosmo)
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_press74.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_angulo12():
    mf = ccl.halos.MassFuncAngulo12(cosmo)
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_angulo12.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_sheth99():
    mf = ccl.halos.MassFuncSheth99(cosmo)
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_sheth99.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_jenkins01():
    mf = ccl.halos.MassFuncJenkins01(cosmo)
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_jenkins01.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)
