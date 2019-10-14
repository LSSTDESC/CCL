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
d_hbf = {}
for model in ['tinker10']:
    d_hbf[model] = np.loadtxt(dirdat + 'hbf_' + model + '.txt',
                              unpack=True)
# Set mass definition
hmd_200m = ccl.HMDef200mat()
zs = np.array([0., 0.5, 1.])

def test_hbf_tinker10():
    mf=ccl.HBiasFuncTinker10(cosmo,hmd_200m)
    m = d_hbf['tinker10'][0]
    for iz, z in enumerate(zs):
        hb_d = d_hbf['tinker10'][iz+1]
        hb_h = mf.get_halo_bias(cosmo, m, 1./(1+z))
        assert np.all(np.fabs(hb_h/hb_d-1) < 0.01)
