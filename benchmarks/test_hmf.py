import os
import warnings
import numpy as np
from . import pyccl as ccl

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
    mf = ccl.halos.MassFuncDespali16(mass_def=hmd)
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_despali16.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_bocquet16():
    hmd = ccl.halos.MassDef200c()
    mf = ccl.halos.MassFuncBocquet16(mass_def=hmd)
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_bocquet16.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_watson13():
    mf = ccl.halos.MassFuncWatson13()
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_watson13.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_tinker08():
    mf = ccl.halos.MassFuncTinker08()
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_tinker08.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_press74():
    mf = ccl.halos.MassFuncPress74()
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_press74.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_angulo12():
    mf = ccl.halos.MassFuncAngulo12()
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_angulo12.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_sheth99():
    mf = ccl.halos.MassFuncSheth99()
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_sheth99.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_jenkins01():
    mf = ccl.halos.MassFuncJenkins01()
    d_hmf = np.loadtxt(os.path.join(dirdat, 'hmf_jenkins01.txt'),
                       unpack=True)
    m = d_hmf[0]
    for iz, z in enumerate(zs):
        nm_d = d_hmf[iz+1]
        nm_h = mf.get_mass_function(cosmo, m, 1. / (1 + z))
        assert np.all(np.fabs(nm_h / nm_d - 1) < 0.01)


def test_hmf_bocquet20():
    EMU_ACCURACY_AND_DATA_SCATTER = 0.2
    z_arr = np.array([0., 0.29, 0.58, 0.87, 1.15, 1.44, 1.73, 2.02])
    d_hmf = np.genfromtxt(
        os.path.join(dirdat,
                     "hmf_bocquet20_digitized_from_webplotdigitizer.csv"),
        delimiter=",", skip_header=2)

    # cosmology used in the example
    fid = {'Ommh2': 0.3 * 0.7**2,
           'Ombh2': .022,
           'Omnuh2': .0006,
           'n_s': 0.96,
           'h': 0.7,
           'w_0': -1,
           'w_a': 0,
           'sigma_8': 0.8}

    # translate to CCL cosmology
    ccl_cosmo = {
        # Neutrinos are treated as a background quantity,
        # so we don't include it in `Omega_c`.
        "Omega_c": (fid["Ommh2"]-fid["Ombh2"])/fid["h"]**2,
        "Omega_b": fid["Ombh2"]/fid["h"]**2,
        "h": fid["h"],
        "n_s": fid["n_s"],
        "sigma8": fid["sigma_8"],
        "w0": fid["w_0"],
        "wa": fid["w_a"],
        "m_nu": ccl.nu_masses(Om_nu_h2=fid["Omnuh2"], mass_split="equal")}

    with warnings.catch_warnings():
        # filter CCL neutrino-cosmologies warnings
        warnings.simplefilter("ignore")
        cosmo = ccl.Cosmology(**ccl_cosmo)
        mf = ccl.halos.MassFuncBocquet20(extrapolate=False)

    for i, z in enumerate(z_arr):
        data = d_hmf[:, 2*i: 2*(i+1)]
        M = data[:, 0] / cosmo["h"]
        nm_d = data[:, 1] * cosmo["h"]**3
        # remove nans
        M = M[~np.isnan(M)]
        nm_d = nm_d[~np.isnan(nm_d)]

        nm_h = mf.get_mass_function(cosmo, M, 1/(1+z))
        assert np.all(np.abs(1 - nm_h/nm_d) < EMU_ACCURACY_AND_DATA_SCATTER)
