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


def test_hbf_tinker10():
    d_hbf = np.loadtxt(os.path.join(dirdat, 'hbf_tinker10.txt'),
                       unpack=True)
    mf = ccl.halos.HaloBiasTinker10(cosmo)
    m = d_hbf[0]
    for iz, z in enumerate(zs):
        hb_d = d_hbf[iz+1]
        hb_h = mf.get_halo_bias(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(hb_h / hb_d - 1) < 1E-3)


def test_hbf_sheth01():
    d_hbf = np.loadtxt(os.path.join(dirdat, 'hbf_sheth01.txt'),
                       unpack=True)
    mf = ccl.halos.HaloBiasSheth01(cosmo)
    m = d_hbf[0]
    for iz, z in enumerate(zs):
        hb_d = d_hbf[iz+1]
        hb_h = mf.get_halo_bias(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(hb_h / hb_d - 1) < 1E-3)


def test_hbf_bhattacharya11():
    d_hbf = np.loadtxt(os.path.join(dirdat, 'hbf_bhattacharya11.txt'),
                       unpack=True)
    mf = ccl.halos.HaloBiasBhattacharya11(cosmo)
    m = d_hbf[0]
    for iz, z in enumerate(zs):
        hb_d = d_hbf[iz+1]
        hb_h = mf.get_halo_bias(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(hb_h / hb_d - 1) < 1E-3)
