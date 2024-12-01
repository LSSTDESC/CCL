import numpy as np
import pyccl as ccl
from pyccl.pkresponse import Pmm_resp, darkemu_Pgm_resp, darkemu_Pgg_resp

cosmo = ccl.Cosmology(
    Omega_c=0.27,
    Omega_b=0.045,
    h=0.67,
    A_s=2.0e-9,
    n_s=0.96,
    transfer_function="boltzmann_camb",
)

deltah = 0.02
deltalnAs = 0.03
lk_arr = np.log(np.geomspace(1e-3, 1e1, 100))
a_arr = np.array([1.0])
mass_def = ccl.halos.MassDef(200, "matter")
cm = ccl.halos.ConcentrationDuffy08(mass_def=mass_def)
prof_hod = ccl.halos.HaloProfileHOD(mass_def=mass_def, concentration=cm)


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