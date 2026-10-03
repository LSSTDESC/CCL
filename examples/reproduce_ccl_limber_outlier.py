"""Reproduce a CCL Limber-integration discontinuity with frozen tomography.

This standalone diagnostic uses only pyccl, NumPy, and Matplotlib.

The tomographic n(z) distributions are frozen benchmark inputs stored in
benchmarks/data/limber_outlier_tomography.npz. They correspond to LSST Y1
lens tomography with a photo-z outlier fraction of 0.30.

The script compares CCL's qag_quad, cquad, and spline Limber integrations
while scanning w0 through a known pathological region.

Run with:

    python examples/reproduce_ccl_limber_outlier.py

or, for example:

    python examples/reproduce_ccl_limber_outlier.py --n-workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl


__all__ = ["main"]


OMEGA_C = 0.2664
OMEGA_B = 0.0492
H = 0.6727
N_S = 0.9645
SIGMA8 = 0.831
WA = 0.0

BIAS_PREFACTOR = 1.05

PAIR = (3, 4)
ELL = 355.655882

OUTLIER_FRAC = 0.30

# Broad scan around the pathological region.
W0_VALUES = np.linspace(-1.08, -0.94, 71)

# Dense scan around the known QAG discontinuity.
FINE_W0_VALUES = np.linspace(-1.042, -1.035, 141)

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "benchmarks"
    / "data"
    / "limber_outlier_tomography.npz"
)

OUTPUT_DIR = (
    ROOT
    / "benchmarks"
    / "limber_outlier_diagnostic"
)


def load_tomography() -> tuple[
    np.ndarray,
    tuple[np.ndarray, ...],
    np.ndarray,
]:
    """Load the frozen tomographic redshift distributions."""
    data = np.load(DATA_FILE)

    z = np.asarray(data["z"], dtype=float)
    centers = np.asarray(data["bin_centers"], dtype=float)

    bins = tuple(
        np.asarray(data[f"nz_{i}"], dtype=float)
        for i in range(len(centers))
    )

    return z, bins, centers


def evaluate_w0(
    args: tuple[
        float,
        np.ndarray,
        tuple[np.ndarray, ...],
        np.ndarray,
    ],
) -> tuple[float, float, float]:
    """Evaluate all CCL Limber methods for one cosmology."""
    w0, z, bins, bin_centers = args

    cosmo = ccl.Cosmology(
        Omega_c=OMEGA_C,
        Omega_b=OMEGA_B,
        h=H,
        n_s=N_S,
        sigma8=SIGMA8,
        w0=float(w0),
        wa=WA,
        m_nu=0.0,
        mass_split="equal",
    )

    a = 1.0 / (1.0 + bin_centers)
    bias_values = BIAS_PREFACTOR / np.asarray(
        ccl.growth_factor(cosmo, a),
        dtype=float,
    )

    tracers = [
        ccl.NumberCountsTracer(
            cosmo,
            has_rsd=False,
            dndz=(z, nz),
            bias=(z, np.full_like(z, bias_value)),
            mag_bias=None,
        )
        for nz, bias_value in zip(
            bins,
            bias_values,
            strict=True,
        )
    ]

    tracer1 = tracers[PAIR[0]]
    tracer2 = tracers[PAIR[1]]

    qag = ccl.angular_cl(
        cosmo,
        tracer1,
        tracer2,
        ELL,
        l_limber=-1,
        limber_integration_method="qag_quad",
    )

    cquad = ccl.angular_cl(
        cosmo,
        tracer1,
        tracer2,
        ELL,
        l_limber=-1,
        limber_integration_method="cquad",
    )

    spline = ccl.angular_cl(
        cosmo,
        tracer1,
        tracer2,
        ELL,
        l_limber=-1,
        limber_integration_method="spline",
    )

    return float(qag), float(cquad), float(spline)


def scan_w0(
    w0_values: np.ndarray,
    *,
    n_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scan C_ell over w0 using all CCL Limber integrators."""
    z, bins, centers = load_tomography()

    jobs = [
        (float(w0), z, bins, centers)
        for w0 in w0_values
    ]

    if n_workers == 1:
        values = [evaluate_w0(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            values = list(
                executor.map(
                    evaluate_w0,
                    jobs,
                    chunksize=1,
                )
            )

    values = np.asarray(values, dtype=float)

    return values[:, 0], values[:, 1], values[:, 2]


def midpoint_residual(values: np.ndarray) -> np.ndarray:
    """Return the fractional residual from neighboring point interpolation."""
    residual = np.full_like(values, np.nan)

    midpoint = 0.5 * (
        values[:-2]
        + values[2:]
    )

    residual[1:-1] = (
        values[1:-1] - midpoint
    ) / np.abs(midpoint)

    return residual


def report_scan(
    w0_values: np.ndarray,
    qag: np.ndarray,
    cquad: np.ndarray,
    spline: np.ndarray,
) -> None:
    """Print the strongest anomalies in a scan."""
    qag_res = midpoint_residual(qag)
    cquad_res = midpoint_residual(cquad)
    spline_res = midpoint_residual(spline)

    iq = int(np.nanargmax(np.abs(qag_res)))
    ic = int(np.nanargmax(np.abs(cquad_res)))
    is_ = int(np.nanargmax(np.abs(spline_res)))

    qag_cquad = (qag - cquad) / cquad
    qag_spline = (qag - spline) / spline
    cquad_spline = (cquad - spline) / spline

    iqc = int(np.argmax(np.abs(qag_cquad)))
    iqs = int(np.argmax(np.abs(qag_spline)))
    ics = int(np.argmax(np.abs(cquad_spline)))

    print()
    print(f"f_out = {OUTLIER_FRAC:.3f}")
    print("-" * 78)

    print(
        f"qag_quad     max local residual = "
        f"{abs(qag_res[iq]):.8e} at w0={w0_values[iq]:.7f}"
    )
    print(
        f"cquad        max local residual = "
        f"{abs(cquad_res[ic]):.8e} at w0={w0_values[ic]:.7f}"
    )
    print(
        f"spline       max local residual = "
        f"{abs(spline_res[is_]):.8e} at w0={w0_values[is_]:.7f}"
    )

    print()
    print(
        f"qag-cquad    max relative diff  = "
        f"{abs(qag_cquad[iqc]):.8e} at w0={w0_values[iqc]:.7f}"
    )
    print(
        f"qag-spline   max relative diff  = "
        f"{abs(qag_spline[iqs]):.8e} at w0={w0_values[iqs]:.7f}"
    )
    print(
        f"cquad-spline max relative diff  = "
        f"{abs(cquad_spline[ics]):.8e} at w0={w0_values[ics]:.7f}"
    )


def plot_tomography() -> None:
    """Plot the frozen tomography used in the diagnostic."""
    z, bins, centers = load_tomography()

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for i, (nz, center) in enumerate(
        zip(bins, centers, strict=True)
    ):
        ax.plot(
            z,
            nz,
            label=rf"bin {i}, $z_c={center:.2f}$",
        )

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$n_i(z)$")
    ax.set_title(
        rf"Frozen LSST Y1 lens tomography, "
        rf"$f_{{\rm out}}={OUTLIER_FRAC:.2f}$"
    )
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "tomography.png",
        dpi=200,
    )
    plt.close(fig)


def plot_scan(
    name: str,
    w0_values: np.ndarray,
    qag: np.ndarray,
    cquad: np.ndarray,
    spline: np.ndarray,
) -> None:
    """Plot a CCL integration-method scan."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        w0_values,
        qag,
        label="qag_quad",
    )
    ax.plot(
        w0_values,
        cquad,
        linestyle="--",
        label="cquad",
    )
    ax.plot(
        w0_values,
        spline,
        linestyle=":",
        label="spline",
    )

    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$C_\ell$")
    ax.set_title(
        rf"$f_{{\rm out}}={OUTLIER_FRAC:.2f}$, "
        rf"$\ell={ELL:.3f}$, bins {PAIR}"
    )
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{name}_scan.png",
        dpi=200,
    )
    plt.close(fig)

    qag_cquad = (qag - cquad) / cquad
    qag_spline = (qag - spline) / spline
    cquad_spline = (cquad - spline) / spline

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        w0_values,
        qag_cquad,
        label="qag_quad - cquad",
    )
    ax.plot(
        w0_values,
        qag_spline,
        linestyle="--",
        label="qag_quad - spline",
    )
    ax.plot(
        w0_values,
        cquad_spline,
        linestyle=":",
        label="cquad - spline",
    )

    ax.axhline(0.0, linewidth=1.0)

    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel("relative difference")
    ax.set_title("CCL Limber integration-method differences")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{name}_method_difference.png",
        dpi=200,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        w0_values,
        midpoint_residual(qag),
        label="qag_quad",
    )
    ax.plot(
        w0_values,
        midpoint_residual(cquad),
        linestyle="--",
        label="cquad",
    )
    ax.plot(
        w0_values,
        midpoint_residual(spline),
        linestyle=":",
        label="spline",
    )

    ax.axhline(0.0, linewidth=1.0)

    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel("relative midpoint residual")
    ax.set_title("CCL Limber integration smoothness")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{name}_smoothness.png",
        dpi=200,
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the standalone CCL Limber diagnostic."""
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    z, bins, centers = load_tomography()

    print()
    print("Standalone CCL Limber integration diagnostic")
    print("=" * 78)
    print(f"tomography file   = {DATA_FILE}")
    print(f"tomographic bins  = {len(bins)}")
    print(f"z samples         = {len(z)}")
    print(f"bin centers       = {centers}")
    print(f"outlier fraction  = {OUTLIER_FRAC:.2f}")
    print(f"tomographic pair  = {PAIR}")
    print(f"ell               = {ELL:.9f}")
    print(f"workers           = {args.n_workers}")
    print("integration modes = qag_quad, cquad, spline")
    print(
        f"coarse w0 scan    = "
        f"{W0_VALUES[0]:.4f} -- {W0_VALUES[-1]:.4f} "
        f"({len(W0_VALUES)} points)"
    )
    print(
        f"fine w0 scan      = "
        f"{FINE_W0_VALUES[0]:.6f} -- {FINE_W0_VALUES[-1]:.6f} "
        f"({len(FINE_W0_VALUES)} points)"
    )
    print(f"output directory  = {OUTPUT_DIR}")

    plot_tomography()

    print()
    print("COARSE SCAN")
    print("=" * 78)

    coarse_qag, coarse_cquad, coarse_spline = scan_w0(
        W0_VALUES,
        n_workers=args.n_workers,
    )

    report_scan(
        W0_VALUES,
        coarse_qag,
        coarse_cquad,
        coarse_spline,
    )

    plot_scan(
        "coarse_w0",
        W0_VALUES,
        coarse_qag,
        coarse_cquad,
        coarse_spline,
    )

    print()
    print("FINE SCAN")
    print("=" * 78)

    fine_qag, fine_cquad, fine_spline = scan_w0(
        FINE_W0_VALUES,
        n_workers=args.n_workers,
    )

    report_scan(
        FINE_W0_VALUES,
        fine_qag,
        fine_cquad,
        fine_spline,
    )

    plot_scan(
        "fine_w0",
        FINE_W0_VALUES,
        fine_qag,
        fine_cquad,
        fine_spline,
    )

    print()
    print("FINE-SCAN SUMMARY")
    print("=" * 78)

    qag_res = midpoint_residual(fine_qag)
    cquad_res = midpoint_residual(fine_cquad)
    spline_res = midpoint_residual(fine_spline)

    iq = int(np.nanargmax(np.abs(qag_res)))
    ic = int(np.nanargmax(np.abs(cquad_res)))
    is_ = int(np.nanargmax(np.abs(spline_res)))

    print(
        f"qag_quad  : {abs(qag_res[iq]):.8e} "
        f"at w0={FINE_W0_VALUES[iq]:.7f}"
    )
    print(
        f"cquad     : {abs(cquad_res[ic]):.8e} "
        f"at w0={FINE_W0_VALUES[ic]:.7f}"
    )
    print(
        f"spline    : {abs(spline_res[is_]):.8e} "
        f"at w0={FINE_W0_VALUES[is_]:.7f}"
    )

    print()
    print("Plots written to:")
    print(f"  {OUTPUT_DIR}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
