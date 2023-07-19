"""
Benchmark for the mass-concentration relations derived
using the Uchuu simulations (arXiv:2007.14720).

With many thanks to Tomoaki Ishiyama for kindly providing
the benchmarks.
"""
import os
import pyccl as ccl
import pytest
import numpy as np

dirdat = os.path.join(os.path.dirname(__file__), "data")

# Planck18 used in Uchuu simulations
COSMO = ccl.Cosmology(Omega_c=0.2589, Omega_b=0.0486, h=0.6774,
                      sigma8=0.8159, n_s=0.9667)
COSMO.compute_sigma()
H100 = COSMO["h"]

Z = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
# All parametrizations are accurate at to 5% except (500, Vmax, relaxed)
# for which the model is accurate to 10%. This may be due to low mass
# precision error. We thus clip the masses at 8 < log10(M) < 16.
UCHUU_DATA_SCATTER = 0.035
M_min, M_max = 1e8, 1e16


@pytest.mark.parametrize("pars",
                         [{"Delta": 200, "relaxed": False, "Vmax": False},
                          {"Delta": 200, "relaxed": False, "Vmax": True},
                          {"Delta": 200, "relaxed": True, "Vmax": False},
                          {"Delta": 200, "relaxed": True, "Vmax": True},
                          {"Delta": "vir", "relaxed": False, "Vmax": False},
                          {"Delta": "vir", "relaxed": False, "Vmax": True},
                          {"Delta": "vir", "relaxed": True, "Vmax": False},
                          {"Delta": "vir", "relaxed": True, "Vmax": True},
                          {"Delta": 500, "relaxed": False, "Vmax": False},
                          {"Delta": 500, "relaxed": True, "Vmax": False}])
def test_concentration_Ishiyama21(pars):
    Delta = pars["Delta"]
    key1 = "cvmax" if pars["Vmax"] else "crs"
    key2 = "relaxed" if pars["relaxed"] else "all"
    key0 = f"Uchuu_m{Delta}-c{Delta}.{key1}.{key2}"
    fname = os.path.join(dirdat, key0)
    data = np.loadtxt(fname)
    M = data[:, 0]

    # mass cutoff at CCL integration boundaries
    data = data[(M > M_min/H100) & (M < M_max/H100)]
    M_use = M[(M > M_min/H100) & (M < M_max/H100)]

    hmd = ccl.halos.MassDef(Delta, "critical")

    cm = ccl.halos.ConcentrationIshiyama21(mass_def=hmd,
                                           relaxed=pars["relaxed"],
                                           Vmax=pars["Vmax"])

    for i, zz in enumerate(Z):
        dat = data[:, i+1]
        mod = cm(COSMO, M_use/H100, 1/(1+zz))
        assert np.allclose(mod, dat, rtol=UCHUU_DATA_SCATTER)
