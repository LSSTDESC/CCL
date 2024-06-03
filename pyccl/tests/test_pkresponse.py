import numpy as np
import pyccl as ccl
from pkresponse import Pmm_resp, darkemu_Pgm_resp, darkemu_Pgg_resp
import pytest

def test_Pmm_resp():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
    deltah = 0.02
    lk_arr = np.log(np.geomspace(1e-4, 1e1, 100))
    a_arr = np.linspace(0.1, 1.0, 10)
    response = Pmm_resp(cosmo, deltah=deltah, lk_arr=lk_arr, a_arr=a_arr)
    
    assert np.all(np.isfinite(response)), "Pmm_resp produced infinity values."

def test_Pgm_resp():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
    hmc = ccl.halos.MassFunc(ccl.halos.MassDef(200, 'critical'))
    prof_hod = ccl.halos.HaloProfileHOD()
    deltah = 0.02
    log10Mh_min = 12.0
    log10Mh_max = 15.9
    lk_arr = np.log(np.geomspace(1e-4, 1e1, 100))
    a_arr = np.linspace(0.1, 1.0, 10)
    response = darkemu_Pgm_resp(cosmo, hmc, prof_hod, deltah=deltah, log10Mh_min=log10Mh_min, log10Mh_max=log10Mh_max, lk_arr=lk_arr, a_arr=a_arr)

    assert np.all(np.isfinite(response)), "darkemu_Pgm_resp produced infinity values."

def test_Pgg_resp():
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
    hmc = ccl.halos.MassFunc(ccl.halos.MassDef(200, 'critical'))
    prof_hod = ccl.halos.HaloProfileHOD()
    deltalnAs = 0.03
    log10Mh_min = 12.0
    log10Mh_max = 15.9
    lk_arr = np.log(np.geomspace(1e-4, 1e1, 100))
    a_arr = np.linspace(0.1, 1.0, 10)
    response = darkemu_Pgg_resp(cosmo, hmc, prof_hod, deltalnAs=deltalnAs, log10Mh_min=log10Mh_min, log10Mh_max=log10Mh_max, lk_arr=lk_arr, a_arr=a_arr)
    
    assert np.all(np.isfinite(response)), "darkemu_Pgg_resp produced infinity values."

if __name__ == "__main__":
    pytest.main()
