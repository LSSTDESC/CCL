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
for model in ['tinker08', 'sheth99']:
    d_hmf[model] = np.loadtxt(dirdat + 'hmf_' + model + '.txt',
                              unpack=True)
# Set mass definition
hmd_200m = ccl.HMDef200mat()
zs = np.array([0., 0.5, 1.])

def test_hmf_tinker08():
    mf=ccl.MassFuncTinker08(cosmo,hmd_200m)
    m = d_hmf['tinker08'][0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf['tinker08'][iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.01)

def test_hmf_sheth99():
    mf=ccl.MassFuncShethTormen(cosmo,hmd_200m)
    m = d_hmf['sheth99'][0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf['sheth99'][iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(nm_h/nm_d-1) < 0.04)
