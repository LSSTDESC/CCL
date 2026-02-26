import os
import numpy as np
import matplotlib.pyplot as plt
import isitgr

print('Using CAMB-ISiTGR %s installed at %s' % (isitgr.__version__, os.path.dirname(isitgr.__file__)))

# ---------------- Config ----------------
H0, ombh2, omch2 = 67.36, 0.02237, 0.1200
mnu, omk, tau, nnu    = 0.0, 0.0, 0.06, 3.046
As, ns           = 2.100e-9, 0.9649

mu_0    = [0.0,  0.1, -0.1,  0.1, -0.1]
sigma_0 = [0.0,  0.1, -0.1, -0.1,  0.1]

z_pk = 0.0

# Match your benchmark-style sampling range (adjust if you want denser / wider)
minkh   = 1e-4
maxkh   = 15.0
npoints = 400

# Output directory (your requested path pattern)
outdir = "/n/home12/cgarciaquintero/CCL/benchmarks/data"
os.makedirs(outdir, exist_ok=True)

# ---------------- Calculation + Plot ----------------
plt.figure(figsize=(7, 5))

for model in range(5):
    mu0 = mu_0[model]
    s0  = sigma_0[model]

    # --- ISiTGR params ---
    pars = isitgr.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau, nnu=nnu,
    MG_parameterization="muSigma", mu0=mu0, Sigma0=s0
    )

    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_matter_power(redshifts=[z_pk], kmax=maxkh)
    pars.NonLinear = isitgr.model.NonLinear_none

    results_isitgr = isitgr.get_results(pars)

    # Get P(k) in the usual CAMB/ISiTGR convention: k in h/Mpc, P in (Mpc/h)^3
    k_isitgr, z_isitgr, pk_isitgr = results_isitgr.get_matter_power_spectrum(
        minkh=minkh, maxkh=maxkh, npoints=npoints,
        var1="delta_nonu", var2="delta_nonu"
    )
    pk0 = pk_isitgr[0]  # z index 0 since we requested only z_pk

    # --- Save benchmark file for CCL test ---
    # Your CCL test loads: k = data[:,0]*h and pk = data[:,1]/h^3
    # => so data[:,0] must be in h/Mpc and data[:,1] in (Mpc/h)^3 (this is what CAMB returns here).
    fout = os.path.join(outdir, f"model{model:d}_pk_isitgr_matterpower.dat")
    np.savetxt(
        fout,
        np.column_stack([k_isitgr, pk0]),
        fmt="%.18e",
    )

    # --- Plot ---
    plt.loglog(k_isitgr, pk0, lw=1.6, label=fr"model {model}: $\mu_0={mu0}$, $\Sigma_0={s0}$")

    # quick sanity print
    print(f"model {model}: wrote {fout}")
    print(f"  k-range [h/Mpc] = ({k_isitgr.min():.3e}, {k_isitgr.max():.3e})  |  P(k) range = ({pk0.min():.3e}, {pk0.max():.3e})")

# ---------------- Save combined plot ----------------
plotdir = "/n/home12/cgarciaquintero/CCL/ccl_benchmarking"
os.makedirs(plotdir, exist_ok=True)
plotpath = os.path.join(plotdir, f"isitgr_pk_musigma_z{z_pk:g}.png")

plt.xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
plt.ylabel(r"$P_{\mathrm{lin}}(k)\ [(\mathrm{Mpc}/h)^3]$")
plt.title(fr"ISiTGR linear matter power spectrum at z={z_pk:g}")
plt.legend(fontsize=9)
plt.tight_layout()

plt.savefig(plotpath, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved plot to: {plotpath}")