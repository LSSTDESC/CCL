import numpy as np
import pyccl as ccl


def test_bocquet20_mf():
    d = np.load("./benchmarks/data/mf_bocquet20mf.py.npz")
    m_nu = ccl.nu_masses(Omega_nu_h2=float(d['Omnuh2']),
                         mass_split='equal')
    h = float(d['h'])
    cosmo = ccl.Cosmology(Omega_c=(float(d['Ommh2']) -
                                   float(d['Ombh2']) -
                                   float(d['Omnuh2']))/h**2,
                          Omega_b=float(d['Ombh2'])/h**2,
                          h=h, n_s=float(d['n_s']),
                          sigma8=float(d['sigma_8']),
                          m_nu=m_nu,
                          w0=float(d['w_0']),
                          wa=float(d['w_a']))
    mf = ccl.halos.MassFuncBocquet20()
    Ms = d['m']/h
    for z, mfp in zip(d['z'], d['mf']):
        mf_here = mf(cosmo, Ms, 1/(1+z))
        assert np.allclose(mfp*h**3*np.log(10), mf_here,
                           atol=0, rtol=1E-5)
