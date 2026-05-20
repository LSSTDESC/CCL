import os
import numpy as np
import pyccl as ccl

# Set cosmology
# based on fiducial cosmology (Planck 2015)
cosmo = ccl.Cosmology(Omega_c=0.2650, Omega_b=0.0492, h=0.6724,
                      A_s=2.2065e-9, n_s=0.9645, w0=-1)
# Read data
dirdat = os.path.join(os.path.dirname(__file__), 'data')

# Redshifts
zs = np.array([0., 0.5, 1.])


def test_hmf_nishimichi19():
    hmd = ccl.halos.MassDef200m
    mf = ccl.halos.MassFuncNishimichi19(mass_def=hmd, extrapolate=True)
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_nishimichi19.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 1E-4)
