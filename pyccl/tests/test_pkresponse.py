import numpy as np
import pyccl as ccl
from pyccl import CCLWarning
import pytest
from dark_emulator import darkemu
from pyccl.pkresponse import (
    Pmm_resp,
    darkemu_Pgm_resp,
    darkemu_Pgg_resp,
    _get_phh_massthreshold_mass,
    _b2H17,
    _darkemu_set_cosmology,
    _darkemu_set_cosmology_forAsresp,
    _set_hmodified_cosmology,
)
from .test_cclobject import check_eq_repr_hash

ccl.update_warning_verbosity("high")

# Set cosmology
Om = 0.3156
ob = 0.02225
oc = 0.1198
OL = 0.6844
As = 2.2065e-9
ns = 0.9645
h = 0.6727

cosmo = ccl.Cosmology(
    Omega_c=oc / (h**2),
    Omega_b=ob / (h**2),
    h=h,
    n_s=ns,
    A_s=As,
    m_nu=0.06,
    transfer_function="boltzmann_camb",
    extra_parameters={"camb": {"halofit_version": "takahashi"}},
)

deltah = 0.02
deltalnAs = 0.03
lk_arr = np.log(np.geomspace(1e-3, 1e1, 100))
k_use = np.exp(lk_arr)
k_emu = k_use / h  # [h/Mpc]
a_arr = np.array([1.0])


# HOD parameters
logMhmin = 13.94
logMh1 = 14.46
alpha = 1.192
kappa = 0.60
sigma_logM = 0.5
sigma_lM = sigma_logM * np.log(10)
logMh0 = logMhmin + np.log10(kappa)

logMmin = np.log10(10**logMhmin / h)
logM0 = np.log10(10**logMh0 / h)
logM1 = np.log10(10**logMh1 / h)

# Construct HOD
mass_def = ccl.halos.MassDef(200, "matter")
cm = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
prof_hod = ccl.halos.HaloProfileHOD(
    mass_def=mass_def,
    concentration=cm,
    log10Mmin_0=logMmin,
    siglnM_0=sigma_lM,
    log10M0_0=logM0,
    log10M1_0=logM1,
    alpha_0=alpha,
)

# initialize dark emulator class
emu = darkemu.de_interface.base_class()

cosmo.compute_linear_power()
pk2dlin = cosmo.get_linear_power("delta_matter:delta_matter")


def test_Pmm_resp_init():
    # lk_arr and a_arr should be specified
    with pytest.raises(ValueError):
        Pmm_resp(cosmo, deltah=deltah, lk_arr=None, a_arr=None)


def test_Pgm_resp_init():
    # lk_arr and a_arr should be specified
    with pytest.raises(ValueError):
        darkemu_Pgm_resp(
            cosmo, prof_hod, deltah=deltah, lk_arr=None, a_arr=None
        )
    with pytest.raises(TypeError):
        darkemu_Pgm_resp(
            cosmo, None, deltah=deltah, lk_arr=lk_arr, a_arr=a_arr
        )

    # dark emulator is valid for z<=1.48
    with pytest.warns(CCLWarning):
        darkemu_Pgm_resp(
            cosmo,
            prof_hod,
            deltah=deltah,
            lk_arr=lk_arr,
            a_arr=np.array([0.3]),
        )

    # dark emulator support range is 10^12 <= M200m <= 10^16 Msun/h
    with pytest.warns(CCLWarning):
        darkemu_Pgm_resp(
            cosmo,
            prof_hod,
            deltah=deltah,
            lk_arr=lk_arr,
            a_arr=a_arr,
            log10Mh_min=11.9,
            log10Mh_max=16.1,
        )


def test_Pgg_resp_init():
    # lk_arr and a_arr should be specified
    with pytest.raises(ValueError):
        darkemu_Pgg_resp(
            cosmo, prof_hod, deltalnAs=deltalnAs, lk_arr=None, a_arr=None
        )

    with pytest.raises(TypeError):
        darkemu_Pgg_resp(
            cosmo, None, deltalnAs=deltalnAs, lk_arr=lk_arr, a_arr=a_arr
        )

    # dark emulator is valid for z<=1.48
    with pytest.warns(CCLWarning):
        darkemu_Pgg_resp(
            cosmo,
            prof_hod,
            deltalnAs=deltalnAs,
            lk_arr=lk_arr,
            a_arr=np.array([0.3]),
        )

    # dark emulator support range is 10^12 <= M200m <= 10^16 Msun/h
    with pytest.warns(CCLWarning):
        darkemu_Pgg_resp(
            cosmo,
            prof_hod,
            deltalnAs=deltalnAs,
            lk_arr=lk_arr,
            a_arr=a_arr,
            log10Mh_min=11.9,
            log10Mh_max=16.1,
        )


def test_Pmm_resp():
    response = Pmm_resp(cosmo, deltah=deltah, lk_arr=lk_arr, a_arr=a_arr)
    valid = (k_emu > 1e-2) & (k_emu < 4)

    assert np.all(response[0][valid] > 0)

    response = Pmm_resp(
        cosmo, deltah=deltah, lk_arr=lk_arr, a_arr=a_arr, use_log=True
    )
    if np.any(response <= 0):
        with pytest.warns(CCLWarning):
            response = Pmm_resp(
                cosmo, deltah=deltah, lk_arr=lk_arr, a_arr=a_arr, use_log=True
            )


def test_Pgm_resp():
    response = darkemu_Pgm_resp(
        cosmo, prof_hod, deltah=deltah, lk_arr=lk_arr, a_arr=a_arr
    )
    valid = (k_emu > 1e-2) & (k_emu < 4)

    assert np.all(response[0][valid] > 0)

    with pytest.warns(CCLWarning):
        response = darkemu_Pgm_resp(
            cosmo,
            prof_hod,
            deltah=deltah,
            lk_arr=lk_arr,
            a_arr=a_arr,
            use_log=True,
        )


def test_Pgg_resp():
    response = darkemu_Pgg_resp(
        cosmo, prof_hod, deltalnAs=deltalnAs, lk_arr=lk_arr, a_arr=a_arr
    )
    valid = (k_emu > 1e-2) & (k_emu < 4)

    assert np.all(response[0][valid] < 0)

    with pytest.warns(CCLWarning):
        response = darkemu_Pgg_resp(
            cosmo,
            prof_hod,
            deltalnAs=deltalnAs,
            lk_arr=lk_arr,
            a_arr=a_arr,
            use_log=True,
        )


# Tests for the utility functions
def test_get_phh_massthreshold_mass():
    # set cosmology for dark emulator
    _darkemu_set_cosmology(emu, cosmo)
    dens1 = 1e-3
    Mbin = 1e13
    redshift = 0.0
    phh = _get_phh_massthreshold_mass(emu, k_emu, dens1, Mbin, redshift)
    pklin = pk2dlin(k_use, 1.0, cosmo) * h**3

    valid = (k_emu > 1e-3) & (k_emu < 1e-2)

    # phh is power spectrum of biased tracer
    assert np.all(phh[valid] > pklin[valid])


def test_b2H17():
    b2 = _b2H17(0.0)

    assert b2 == 0.77


def test_darkemu_set_cosmology():
    # Cosmo parameters out of bounds
    cosmo_wr = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.67, A_s=2.2e-9, n_s=2.0
    )
    with pytest.raises(ValueError):
        _darkemu_set_cosmology(emu, cosmo_wr)

    # A_s must be provided
    cosmo_noAs = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96
    )
    with pytest.raises(ValueError):
        _darkemu_set_cosmology(emu, cosmo_noAs)


def test_darkemu_set_cosmology_forAsresp():
    # Cosmo parameters out of bounds
    cosmo_wr = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.67, A_s=2.2e-9, n_s=2.0
    )
    with pytest.raises(ValueError):
        _darkemu_set_cosmology_forAsresp(emu, cosmo_wr, deltalnAs=100.0)

    # A_s must be provided
    cosmo_noAs = ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96
    )
    with pytest.raises(ValueError):
        _darkemu_set_cosmology(emu, cosmo_noAs)


def test_set_hmodified_cosmology():
    cosmo_hp, cosmo_hm = _set_hmodified_cosmology(
        cosmo, deltah, extra_parameters=None
    )

    for cosmo_test in [cosmo_hp, cosmo_hm]:
        cosmo_dict = cosmo_test.to_dict()
        cosmo_dict["h"] = cosmo["h"]
        cosmo_dict["Omega_c"] = cosmo["Omega_c"]
        cosmo_dict["Omega_b"] = cosmo["Omega_b"]
        cosmo_dict["extra_parameters"] = cosmo["extra_parameters"]
        cosmo_new = ccl.Cosmology(**cosmo_dict)

        # make sure the output cosmologies are exactly the same as
        # the input one, except for the modified parameters.
        assert check_eq_repr_hash(cosmo, cosmo_new)
