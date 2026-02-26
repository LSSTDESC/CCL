import os
import numpy as np
import pytest
import pyccl as ccl
from pyccl.modified_gravity import MuSigmaMG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


POWER_MG_TOL = 1e-2
PLOT_DIR = "/n/home12/cgarciaquintero/CCL/ccl_benchmarking/results"


@pytest.mark.parametrize('model', list(range(5)))
def test_power_mg(model):
    mu_0 = [0., 0.1, -0.1, 0.1, -0.1]
    sigma_0 = [0., 0.1, -0.1, -0.1, 0.1]

    h0 = 0.6736

    # --- CCL cosmology for this model ---
    cosmoMG = ccl.Cosmology(
        Omega_c=0.1200 / h0**2,
        Omega_b=0.02237 / h0**2,
        h=h0,
        A_s=2.100e-9,
        n_s=0.9649,
        Neff=3.046,
        Omega_k=0,
        m_nu=0,
        T_CMB=2.7255,
        T_ncdm=(4/11)**(1/3),
        mass_split='equal',
        mg_parametrization=MuSigmaMG(mu_0=mu_0[model], sigma_0=sigma_0[model]),
        matter_power_spectrum='linear',
        transfer_function='boltzmann_isitgr',
    )

    # --- CCL baseline (mu0=sigma0=0) ---
    cosmoGR = ccl.Cosmology(
        Omega_c=0.1200 / h0**2,
        Omega_b=0.02237 / h0**2,
        h=h0,
        A_s=2.100e-9,
        n_s=0.9649,
        Neff=3.046,
        Omega_k=0,
        m_nu=0,
        T_CMB=2.7255,
        T_ncdm=(4/11)**(1/3),
        mass_split='equal',
        mg_parametrization=MuSigmaMG(mu_0=0.0, sigma_0=0.0),
        matter_power_spectrum='linear',
        transfer_function='boltzmann_isitgr',
    )

    # --- load benchmark for this model ---
    data = np.loadtxt(f"./benchmarks/data/model{model:d}_pk_isitgr_matterpower.dat")
    k_hmpc = data[:, 0]
    pk_bm_h3 = data[:, 1]

    # --- load baseline benchmark (model 0: mu0=sigma0=0) ---
    data0 = np.loadtxt("./benchmarks/data/model0_pk_isitgr_matterpower.dat")
    k0_hmpc = data0[:, 0]
    pk0_bm_h3 = data0[:, 1]

    a = 1.0

    # Bench file: k [h/Mpc], Pk [(Mpc/h)^3]
    # CCL expects k [1/Mpc] and returns Pk [Mpc^3]
    k = k_hmpc * cosmoMG["h"]
    pk_bm = pk_bm_h3 / (cosmoMG["h"]**3)
    pk_ccl = ccl.linear_matter_power(cosmoMG, k, a)

    # Baseline curve in same k-grid as model's plot
    pk0_ccl = ccl.linear_matter_power(cosmoGR, k, a) * (cosmoGR["h"]**3)  # to (Mpc/h)^3

    frac = pk_ccl / pk_bm - 1.0
    err = np.abs(frac)

    cut = k_hmpc > 1e-04
    passed = np.allclose(err[cut], 0.0, rtol=0.0, atol=POWER_MG_TOL)

    if not passed:
        os.makedirs(PLOT_DIR, exist_ok=True)
        out = os.path.join(
            PLOT_DIR,
            f"pk_fail_model{model:d}_mu{mu_0[model]:+.3f}_sigma{sigma_0[model]:+.3f}.png",
        )

        fig = plt.figure(figsize=(7.0, 6.0))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

        ax = fig.add_subplot(gs[0])

        # model curves
        ax.loglog(k_hmpc[cut], pk_bm_h3[cut], lw=1, linestyle="-",
                  label="Benchmark (ISiTGR file)")
        ax.loglog(k_hmpc[cut], pk_ccl[cut] * (cosmoMG["h"]**3), lw=1, linestyle="--",
                  label="CCL (converted)")

        # baseline curves in BLACK
        # 1) baseline benchmark (may have slightly different k-grid)
        ax.loglog(k0_hmpc[k0_hmpc > 1e-04], pk0_bm_h3[k0_hmpc > 1e-04],
                  linestyle=":", color="k", label="Baseline bm (mu0=sigma0=0)")

        # 2) baseline CCL evaluated on this model's k-grid
        ax.loglog(k_hmpc[cut], pk0_ccl[cut],
                  linestyle="--", color="k", label="Baseline CCL (mu0=sigma0=0)")

        ax.set_ylabel(r"$P_{\rm lin}(k)\,[(\mathrm{Mpc}/h)^3]$")
        ax.set_title(f"model {model}  (mu0={mu_0[model]:+.3f}, sigma0={sigma_0[model]:+.3f})")
        ax.legend(loc="best", fontsize=8)

        ax2 = fig.add_subplot(gs[1], sharex=ax)
        ax2.axhline(0.0, linestyle="-")
        ax2.semilogx(k_hmpc[cut], frac[cut], marker=".", linestyle="none")
        ax2.axhline(+POWER_MG_TOL, linestyle="--")
        ax2.axhline(-POWER_MG_TOL, linestyle="--")
        ax2.set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        ax2.set_ylabel(r"$P_{\rm CCL}/P_{\rm bm}-1$")

        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved failure plot to: {out}")

    assert passed