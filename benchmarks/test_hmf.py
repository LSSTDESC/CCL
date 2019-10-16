import os
import numpy as np
import pyccl as ccl
import pytest

# Set cosmology
cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, Omega_g=0, Omega_k=0,
                      h=0.7, sigma8=0.8, n_s=0.96, Neff=0, m_nu=0.0,
                      w0=-1, wa=0, T_CMB=2.7255, transfer_function='eisenstein_hu')
# Read data
dirdat = os.path.dirname(__file__) + '/data/'

# Redshifts
zs = np.array([0., 0.5, 1.])

def test_hmf_despali16():
    hmd = ccl.HMDef('vir','critical')
    mf=ccl.MassFuncDespali16(cosmo,hmd)
    d_hmf = np.loadtxt(dirdat + 'hmf_despali16.txt', unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.01)

def test_hmf_bocquet16():
    hmd = ccl.HMDef200crit()
    mf=ccl.MassFuncBocquet16(cosmo,hmd)
    d_hmf = np.loadtxt(dirdat + 'hmf_bocquet16.txt', unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.01)

def test_hmf_watson13():
    hmd = ccl.HMDef200mat()
    mf=ccl.MassFuncWatson13(cosmo,hmd)
    d_hmf = np.loadtxt(dirdat + 'hmf_watson13.txt', unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.01)

def test_hmf_tinker08():
    hmd = ccl.HMDef200mat()
    mf=ccl.MassFuncTinker08(cosmo,hmd)
    d_hmf = np.loadtxt(dirdat + 'hmf_tinker08.txt', unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.01)

def test_hmf_press74():
    hmd = ccl.HMDef('fof', 'matter')
    mf=ccl.MassFuncPress74(cosmo,hmd)
    d_hmf = np.loadtxt(dirdat + 'hmf_press74.txt', unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.05)

def test_hmf_angulo12():
    hmd = ccl.HMDef('fof', 'matter')
    mf=ccl.MassFuncAngulo12(cosmo,hmd)
    d_hmf = np.loadtxt(dirdat + 'hmf_angulo12.txt', unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.05)

def test_hmf_sheth99():
    hmd = ccl.HMDef('fof', 'matter')
    mf=ccl.MassFuncSheth99(cosmo,hmd)
    d_hmf = np.loadtxt(dirdat + 'hmf_sheth99.txt', unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.05)

def test_hmf_jenkins01():
    hmd = ccl.HMDef('fof', 'matter')
    mf=ccl.MassFuncJenkins01(cosmo,hmd)
    d_hmf = np.loadtxt(dirdat + 'hmf_jenkins01.txt', unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.01)
