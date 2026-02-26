#!/usr/bin/env python3
from __future__ import annotations

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from cobaya.model import get_model


# ----------------------------
# Settings
# ----------------------------
YAML_FILE = "ccl.yaml"

# Where CCL later expects to read the predictions
DIRDAT = Path("/n/home12/cgarciaquintero/CCL/benchmarks/data")

# CCL test uses this theta grid (arcmin)
THETA_FILE = DIRDAT / "theta_corr_MG.dat"

# Output names expected by your pytest loader
FN_W = DIRDAT / "wtheta_isitgr_linear_scale_dependence_prediction.dat"
FN_T = DIRDAT / "gammat_isitgr_linear_scale_dependence_prediction.dat"
FN_P = DIRDAT / "Xip_isitgr_linear_scale_dependence_prediction.dat"
FN_M = DIRDAT / "Xim_isitgr_linear_scale_dependence_prediction.dat"

# Plot output (keep local)
PLOT_OUTDIR = Path.cwd()


def _require(arr, name: str):
    if arr is None:
        raise RuntimeError(f"Missing required theory array: {name}")


def _as1d(x) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _take_exact(x, n: int, name: str) -> np.ndarray:
    x = _as1d(x)
    if x.size != n:
        raise RuntimeError(f"{name} has size {x.size}, expected {n}")
    return x


def _load_p0(info: dict) -> dict:
    """Support both plain scalars and {value: ...} style."""
    p0 = {}
    for name, cfg in info.get("params", {}).items():
        if isinstance(cfg, dict):
            if "value" in cfg:
                p0[name] = cfg["value"]
        else:
            p0[name] = cfg
    return p0


def main():
    info = yaml.safe_load(open(YAML_FILE, "r"))
    model = get_model(info)

    print("Likelihoods loaded:", list(model.likelihood.keys()))
    des_like = model.likelihood["ccl.joint"]

    # Build p0 from YAML (your YAML uses plain scalars)
    p0 = _load_p0(info)
    print("p0 sanity:", {k: p0.get(k, None) for k in ["H0", "ombh2", "omch2", "As", "ns", "mu0", "Sigma0"]})

    # Force one evaluation (wires provider)
    _ = model.logposterior(p0)

    # --- Build the same interpolators your likelihood uses ---
    use_MG = getattr(des_like, "use_MG", False)
    use_Weyl = getattr(des_like, "use_Weyl", False)
    acc = float(getattr(des_like, "acc", 1.0))

    extrap_kmax = 8000

    if use_MG:
        if not use_Weyl:
            raise RuntimeError("In your likelihood, use_MG=True requires use_Weyl=True.")
        PKdelta = des_like.provider.get_Pk_interpolator(
            ("delta_tot", "delta_tot"), nonlinear=False, extrap_kmax=extrap_kmax * acc
        )
        PKdeltaWeyl = des_like.provider.get_Pk_interpolator(
            ("delta_tot", "Weyl"), nonlinear=False, extrap_kmax=extrap_kmax * acc
        )
        PKWeyl = des_like.provider.get_Pk_interpolator(
            ("Weyl", "Weyl"), nonlinear=False, extrap_kmax=extrap_kmax * acc
        )
    else:
        PKdelta = des_like.provider.get_Pk_interpolator(
            ("delta_tot", "delta_tot"), nonlinear=False, extrap_kmax=extrap_kmax * acc
        )
        PKdeltaWeyl = None
        PKWeyl = None
        if use_Weyl:
            PKWeyl = des_like.provider.get_Pk_interpolator(
                ("Weyl", "Weyl"), nonlinear=False, extrap_kmax=extrap_kmax * acc
            )

    # ----------------------------
    # Extra plot: P(k) from interpolators
    # ----------------------------
    # Choose a couple of z's to inspect (within DES z-range)
    z_plot = [0.3, 0.6, 1.0]

    # Build a k-array (1/Mpc). Use a safe upper bound and also respect each interpolator kmax.
    # PK*.kmax exists for cobaya Pk_interpolator objects.
    kmin = 1e-4
    # pick a "requested" max, then we'll clip per interpolator
    kmax_req = 50.0
    nk = 400
    k_full = np.logspace(np.log10(kmin), np.log10(kmax_req), nk)

    def _pk_raw(PK, z, k):
        if PK is None:
            return None, None
        return k, PK.P(z, k, grid=False)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.8))

    for z0 in z_plot:
        # delta-delta
        k_dd, p_dd = _pk_raw(PKdelta, z0, k_full)
        ax.loglog(k_dd, np.abs(p_dd), label=fr"$P_{{\delta\delta}}(z={z0})$")

        # delta-Weyl (MG only)
        if PKdeltaWeyl is not None:
            k_dw, p_dw = _pk_raw(PKdeltaWeyl, z0, k_full)
            ax.loglog(k_dw, np.abs(p_dw), ls="--", label=fr"$|P_{{\delta W}}|(z={z0})$")

        # Weyl-Weyl (Weyl enabled)
        if PKWeyl is not None:
            k_ww, p_ww = _pk_raw(PKWeyl, z0, k_full)
            ax.loglog(k_ww, np.abs(p_ww), ls=":", label=fr"$P_{{WW}}(z={z0})$")

    ax.set_xlabel(r"$k\ [\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$|P(z,k)|$")
    ax.set_title(f"Pk interpolators (use_MG={use_MG}, use_Weyl={use_Weyl})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=1)
    fig.tight_layout()
    fig.savefig(PLOT_OUTDIR / "pk_debug.png", dpi=200)
    plt.close(fig)

    print("Saved:", PLOT_OUTDIR / "pk_debug.png")

    wl_photoz_errors = [p0.get(p, 0.0) for p in ["DES_DzS1", "DES_DzS2", "DES_DzS3", "DES_DzS4"]]
    lens_photoz_errors = [p0.get(p, 0.0) for p in ["DES_DzL1", "DES_DzL2", "DES_DzL3", "DES_DzL4", "DES_DzL5"]]
    bin_bias = [p0.get(p, 1.0) for p in ["DES_b1", "DES_b2", "DES_b3", "DES_b4", "DES_b5"]]
    shear_m = [p0.get(p, 0.0) for p in ["DES_m1", "DES_m2", "DES_m3", "DES_m4"]]

    theory = des_like.get_theory(
        PKdelta, PKWeyl, PKdeltaWeyl,
        bin_bias=bin_bias,
        wl_photoz_errors=wl_photoz_errors,
        lens_photoz_errors=lens_photoz_errors,
        shear_calibration_parameters=shear_m,
        intrinsic_alignment_A=p0.get("DES_AIA", 0.0),
        intrinsic_alignment_alpha=p0.get("DES_alphaIA", 0.0),
        intrinsic_alignment_z0=p0.get("DES_z0IA", 0.62),
    )

    corrs_p, corrs_m, corrs_t, corrs_w = theory
    _require(corrs_p, "corrs_p")
    _require(corrs_m, "corrs_m")
    _require(corrs_t, "corrs_t")
    _require(corrs_w, "corrs_w")

    # ----------------------------
    # Plots (optional but nice)
    # ----------------------------
    axs = des_like.plot_lensing(corrs_p=corrs_p, corrs_m=corrs_m, diff=False, errors=True)
    fig = axs[0, 0].figure
    fig.suptitle("DES 2-bin lensing (xip/xim)")
    fig.tight_layout()
    fig.savefig(PLOT_OUTDIR / "des2bin_lensing_xipxim.png", dpi=200)
    plt.close(fig)

    axs = des_like.plot_cross(corrs_t=corrs_t, diff=False, errors=True)
    fig = axs[0, 0].figure
    fig.suptitle("DES 2-bin gammat")
    fig.tight_layout()
    fig.savefig(PLOT_OUTDIR / "des2bin_gammat.png", dpi=200)
    plt.close(fig)

    axs = des_like.plot_w(corrs_w=corrs_w, diff=False, errors=True)
    fig = axs[0].figure
    fig.suptitle("DES 2-bin wtheta")
    fig.tight_layout()
    fig.savefig(PLOT_OUTDIR / "des2bin_wtheta.png", dpi=200)
    plt.close(fig)

    print("Saved plots to:", PLOT_OUTDIR)

    # ----------------------------
    # MATCH PYTEST THETA GRID
    # ----------------------------
    DIRDAT.mkdir(parents=True, exist_ok=True)

    if not THETA_FILE.exists():
        raise RuntimeError(f"Missing {THETA_FILE}. Your pytest uses this file as theta grid.")

    theta_arcmin = np.loadtxt(THETA_FILE)
    ntheta = int(theta_arcmin.size)
    print(f"Using ntheta = {ntheta} from {THETA_FILE.name}")

    # Optional sanity: compare to des_like.theta_bins
    if hasattr(des_like, "theta_bins") and len(des_like.theta_bins) == ntheta:
        tb = np.asarray(des_like.theta_bins)
        if not np.allclose(tb, theta_arcmin, rtol=0, atol=1e-10):
            print("WARNING: des_like.theta_bins != theta_corr_MG.dat (values differ).")
            print("         Writing files on theta_corr_MG.dat ordering anyway (to match pytest).")

    # ----------------------------
    # Write vectors in EXACT order expected by pytest loader
    # ----------------------------

    # wtheta: dd_11 then dd_22
    dd_11 = _take_exact(corrs_w[0, 0], ntheta, "wtheta(0,0)")
    dd_22 = _take_exact(corrs_w[1, 1], ntheta, "wtheta(1,1)")
    wvec = np.concatenate([dd_11, dd_22])
    np.savetxt(FN_W, wvec, fmt="%.18e")
    print("Wrote:", FN_W, "len =", wvec.size)

    # gammat: dl_11 dl_12 dl_21 dl_22  (lens x source)
    dl_11 = _take_exact(corrs_t[0, 0], ntheta, "gammat(0,0)")
    dl_12 = _take_exact(corrs_t[0, 1], ntheta, "gammat(0,1)")
    dl_21 = _take_exact(corrs_t[1, 0], ntheta, "gammat(1,0)")
    dl_22 = _take_exact(corrs_t[1, 1], ntheta, "gammat(1,1)")
    tvec = np.concatenate([dl_11, dl_12, dl_21, dl_22])
    np.savetxt(FN_T, tvec, fmt="%.18e")
    print("Wrote:", FN_T, "len =", tvec.size)

    # xip: ll_11, ll_12, ll_22 (source-source)
    ll_11_p = _take_exact(corrs_p[0, 0], ntheta, "xip(0,0)")
    ll_12_p = _take_exact(corrs_p[0, 1], ntheta, "xip(0,1)")
    ll_22_p = _take_exact(corrs_p[1, 1], ntheta, "xip(1,1)")
    pvec = np.concatenate([ll_11_p, ll_12_p, ll_22_p])
    np.savetxt(FN_P, pvec, fmt="%.18e")
    print("Wrote:", FN_P, "len =", pvec.size)

    # xim: ll_11, ll_12, ll_22
    ll_11_m = _take_exact(corrs_m[0, 0], ntheta, "xim(0,0)")
    ll_12_m = _take_exact(corrs_m[0, 1], ntheta, "xim(0,1)")
    ll_22_m = _take_exact(corrs_m[1, 1], ntheta, "xim(1,1)")
    mvec = np.concatenate([ll_11_m, ll_12_m, ll_22_m])
    np.savetxt(FN_M, mvec, fmt="%.18e")
    print("Wrote:", FN_M, "len =", mvec.size)

    print("\nDone. Your pytest loader should read:")
    print("  wtheta:", FN_W.name)
    print("  gammat:", FN_T.name)
    print("  Xip   :", FN_P.name)
    print("  Xim   :", FN_M.name)


if __name__ == "__main__":
    main()
