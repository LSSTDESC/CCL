import numpy as np
import pyccl as ccl
import pytest
from pyccl.pkresponse import *
from pyccl.pkresponse import _mass_to_dens, _get_phh_massthreshold_mass , _b2H17, _darkemu_set_cosmology, _darkemu_set_cosmology_forAsresp, _set_hmodified_cosmology
from .test_cclobject import check_eq_repr_hash

cosmo = ccl.Cosmology(
    Omega_c=0.27,
    Omega_b=0.048,
    h=0.67,
    A_s=2.0e-9,
    n_s=0.96,
    transfer_function="boltzmann_camb",
)

deltah = 0.02
deltalnAs = 0.03
lk_arr = np.log(np.geomspace(1e-3, 1e1, 100))
k_use = np.exp(lk_arr)
a_arr = np.array([1.0])
mass_def = ccl.halos.MassDef(200, "matter")
cm = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
prof_hod = ccl.halos.HaloProfileHOD(mass_def=mass_def, concentration=cm)

# initialize dark emulator class
emu = darkemu.de_interface.base_class()

cosmo.compute_linear_power()
pk2dlin = cosmo.get_linear_power("delta_matter:delta_matter")


def test_Pmm_resp():
    response = Pmm_resp(cosmo, deltah=deltah, lk_arr=lk_arr, a_arr=a_arr)

    assert np.all(np.isfinite(response))


def test_Pgm_resp():
    response = darkemu_Pgm_resp(
        cosmo, prof_hod, deltah=deltah, lk_arr=lk_arr, a_arr=a_arr
    )

    assert np.all(np.isfinite(response))


def test_Pgg_resp():
    response = darkemu_Pgg_resp(
        cosmo, prof_hod, deltalnAs=deltalnAs, lk_arr=lk_arr, a_arr=a_arr
    )

    assert np.all(np.isfinite(response))

# Tests for the utility functions
def test_mass_to_dens():
    def dndlog10m(logM):
        return np.ones_like(logM)
    
    mass_thre = 1e13
    dens = _mass_to_dens(dndlog10m, cosmo, mass_thre)

    assert dens > 0

def test_get_phh_massthreshold_mass():
    # set cosmology for dark emulator
    _darkemu_set_cosmology(emu, cosmo)
    h = cosmo["h"]
    k_emu = k_use / h  # [h/Mpc]
    dens1 = 1e-3
    Mbin = 1e13
    redshift = 0.0    
    phh = _get_phh_massthreshold_mass(emu, k_emu, dens1, Mbin, redshift)
    pklin = pk2dlin(k_use, 1.0, cosmo)*h**3 

    valid = (k_emu > 1e-3) & (k_emu < 1e-2)

    # phh is power spectrum of biased tracer
    assert np.all(phh[valid] > pklin[valid])

def test_b2H17():
    b2 = _b2H17(0.0)

    assert b2 == 0.77

def test_darkemu_set_cosmology():
    # Cosmo parameters out of bounds
    cosmo_wr = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                             A_s=2.2e-9, n_s=2.0)
    with pytest.raises(ValueError):
        _darkemu_set_cosmology(emu, cosmo_wr)

def test_darkemu_set_cosmology_forAsresp():
    # Cosmo parameters out of bounds
    cosmo_wr = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.67,
                             A_s=2.2e-9, n_s=2.0)
    with pytest.raises(ValueError):
        _darkemu_set_cosmology_forAsresp(emu, cosmo_wr, deltalnAs=100.0)


def test_set_hmodified_cosmology():
    cosmo_hp, cosmo_hm = _set_hmodified_cosmology(cosmo, deltah, extra_parameters=None)
    
    for cosmo_test in [cosmo_hp, cosmo_hm]:
        cosmo_dict = cosmo_test.to_dict()
        cosmo_dict["h"] = cosmo["h"]
        cosmo_dict["Omega_c"] = cosmo["Omega_c"]
        cosmo_dict["Omega_b"] = cosmo["Omega_b"]
        cosmo_dict["extra_parameters"] = cosmo["extra_parameters"]
        cosmo_new = cosmology.Cosmology(**cosmo_dict)
        
        # make sure the output cosmologies are exactly the same as the input one, except for the modified parameters.
        assert check_eq_repr_hash(cosmo, cosmo_new)
    
