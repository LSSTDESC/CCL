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
d_hmf = {}
for model in ['press74', 'jenkins01', 'tinker08', 'sheth99',
              'despali16', 'bocquet16']:
    d_hmf[model] = np.loadtxt(dirdat + 'hmf_' + model + '.txt',
                              unpack=True)
# Redshifts
zs = np.array([0., 0.5, 1.])

def test_hmf_despali16():
    hmd = ccl.HMDef('vir','critical')
    mf=ccl.MassFuncDespali16(cosmo,hmd)
    m = d_hmf['despali16'][0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf['despali16'][iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.01)

def test_hmf_bocquet16():
    hmd = ccl.HMDef200crit()
    mf=ccl.MassFuncBocquet16(cosmo,hmd)
    m = d_hmf['bocquet16'][0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf['bocquet16'][iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.01)

def test_hmf_tinker08():
    hmd = ccl.HMDef200mat()
    mf=ccl.MassFuncTinker08(cosmo,hmd)
    m = d_hmf['tinker08'][0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf['tinker08'][iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.01)

def test_hmf_press74():
    hmd = ccl.HMDef('fof', 'matter')
    mf=ccl.MassFuncPress74(cosmo,hmd)
    m = d_hmf['press74'][0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf['press74'][iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.05)

def test_hmf_sheth99():
    hmd = ccl.HMDef('fof', 'matter')
    mf=ccl.MassFuncSheth99(cosmo,hmd)
    m = d_hmf['sheth99'][0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf['sheth99'][iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.05)

def test_hmf_jenkins01():
    hmd = ccl.HMDef('fof', 'matter')
    mf=ccl.MassFuncJenkins01(cosmo,hmd)
    m = d_hmf['jenkins01'][0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf['jenkins01'][iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.01)
