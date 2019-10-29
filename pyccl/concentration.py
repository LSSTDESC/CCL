from . import ccllib as lib
from .background import growth_factor
from .hmfunc import sigmaM
from .massdef import mass2radius_lagrangian
import numpy as np
from .power import linear_matter_power
from scipy.interpolate import InterpolatedUnivariateSpline


def concentration_diemer15_200crit(cosmo, M, a):
    M_use = np.atleast_1d(M)

    # Compute power spectrum slope
    DIEMER15_KAPPA = 1.0
    R = mass2radius_lagrangian(cosmo, M_use)
    lk_R = np.log10(2.0 * np.pi / R * DIEMER15_KAPPA)
    lkmin = np.amin(lk_R-0.05)
    lkmax = np.amax(lk_R+0.05)
    logk = np.arange(lkmin, lkmax, 0.01)
    lpk = np.log10(linear_matter_power(cosmo, 10**logk, a))
    interp = InterpolatedUnivariateSpline(logk, lpk)
    n = interp(lk_R, nu=1)

    status = 0
    delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
    sig = sigmaM(cosmo, M_use, a)
    nu = delta_c / sig

    DIEMER15_MEDIAN_PHI_0 = 6.58
    DIEMER15_MEDIAN_PHI_1 = 1.27
    DIEMER15_MEDIAN_ETA_0 = 7.28
    DIEMER15_MEDIAN_ETA_1 = 1.56
    DIEMER15_MEDIAN_ALPHA = 1.08
    DIEMER15_MEDIAN_BETA = 1.77

    floor = DIEMER15_MEDIAN_PHI_0 + n * DIEMER15_MEDIAN_PHI_1
    nu0 = DIEMER15_MEDIAN_ETA_0 + n * DIEMER15_MEDIAN_ETA_1
    alpha = DIEMER15_MEDIAN_ALPHA
    beta = DIEMER15_MEDIAN_BETA
    c = 0.5 * floor * ((nu0 / nu)**alpha + (nu / nu0)**beta)
    if np.isscalar(M):
        c = c[0]

    return c


def concentration_bhattacharya13_generic(cosmo, M, a, A, B, C):
    gz = growth_factor(cosmo, a)
    status = 0
    delta_c, status = lib.dc_NakamuraSuto(cosmo.cosmo, a, status)
    sig = sigmaM(cosmo, M, a)
    nu = delta_c / sig
    return A * gz**B * nu**C


def concentration_bhattacharya13_200crit(cosmo, M, a):
    return concentration_bhattacharya13_generic(cosmo, M, a,
                                                5.9, 0.54, -0.35)


def concentration_bhattacharya13_200mat(cosmo, M, a):
    return concentration_bhattacharya13_generic(cosmo, M, a,
                                                9.0, 1.15, -0.29)


def concentration_bhattacharya13_vir(cosmo, M, a):
    return concentration_bhattacharya13_generic(cosmo, M, a,
                                                7.7, 0.9, -0.29)


def concentration_prada12_200crit(cosmo, M, a):
    def cmin(x):
        c0 = 3.681
        c1 = 5.033
        al = 6.948
        x0 = 0.424
        return c0 + (c1 - c0) * (np.arctan(al * (x - x0)) / np.pi + 0.5)

    def imin(x):
        i0 = 1.047
        i1 = 1.646
        be = 7.386
        x1 = 0.526
        return i0 + (i1 - i0) * (np.arctan(be * (x - x1)) / np.pi + 0.5)

    sig = sigmaM(cosmo, M, a)
    om = cosmo.cosmo.params.Omega_m
    ol = cosmo.cosmo.params.Omega_l
    x = a * (ol / om)**(1. / 3.)
    B0 = cmin(x)/cmin(1.393)
    B1 = imin(x)/imin(1.393)
    sig_p = B1 * sig
    Cc = 2.881 * ((sig_p / 1.257)**1.022 + 1) * np.exp(0.060 / sig_p**2)
    return B0 * Cc


def concentration_klypin11_vir(cosmo, M, a):
    M_pivot_inv = cosmo.cosmo.params.h * 1E-12
    return 9.6 * (M * M_pivot_inv)**-0.075


def concentration_duffy08_generic(cosmo, M, a, A, B, C):
    M_pivot_inv = cosmo.cosmo.params.h * 5E-13
    return A * (M * M_pivot_inv)**B * a**(-C)


def concentration_duffy08_200mat(cosmo, M, a):
    return concentration_duffy08_generic(cosmo, M, a,
                                         10.14,   # A
                                         -0.081,  # B
                                         -1.01)   # C


def concentration_duffy08_200crit(cosmo, M, a):
    return concentration_duffy08_generic(cosmo, M, a,
                                         5.71,    # A
                                         -0.084,  # B
                                         -0.47)   # C
