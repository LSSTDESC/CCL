"""Script to make benchmark data for Van Daalen et al. (2019) baryon model in CCL."""

from pathlib import Path

import numpy as np

import pyccl as ccl

# This file is in benchmarks/data/codes/, so parent is benchmarks/data/
HERE = Path(__file__).resolve()
DATADIR = HERE.parent.parent

# Same-ish cosmology as other baryon benchmarks
cosmo = ccl.Cosmology(
    Omega_c=0.25,
    Omega_b=0.05,
    h=0.7,
    A_s=2.2e-9,
    n_s=0.96,
    Neff=3.046,
    mass_split="normal",
    m_nu=0.0,
    Omega_g=0.0,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0,
)

baryons_500c = ccl.BaryonsvanDaalen19(fbar=0.7, mass_def="500c")

# Define k and a
k_hmpc = np.logspace(-2, 0, 200)  # 0.01–1 h/Mpc
a = 1.0

# Convert to 1/Mpc for the model
k_1mpc = k_hmpc * cosmo["h"]

fk_500c = baryons_500c.boost_factor(cosmo, k_1mpc, a)
data_500c = np.column_stack([k_hmpc, fk_500c])

out_500c = DATADIR / "baryons_vd19_fk_500c.txt"
np.savetxt(out_500c, data_500c)
print("Wrote", out_500c)
