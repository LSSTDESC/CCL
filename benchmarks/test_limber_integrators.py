"""Stress-test CCL Limber integration methods across tomography and scale.

This diagnostic compares qag_quad, cquad, and spline for frozen LSST Y1
lens tomography.

The test first performs a broad scan over tomographic bin pairs, multipoles,
and w0 values. It then performs dense local w0 scans around the strongest
method disagreements and smoothness anomalies found in the broad scan.

A known narrow QAG failure near w0=-1.0389 is included explicitly as a
regression target.

Run with:

    python benchmarks/test_limber_integrators.py

or:

    python benchmarks/test_limber_integrators.py --n-workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations_with_replacement
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
OUTLIER_FRAC = 0.30

W0_VALUES = np.linspace(-1.08, -0.94, 71)
ELL_VALUES = np.geomspace(20.0, 2000.0, 25)

METHODS = ("qag_quad", "cquad", "spline")

N_REFINE_CANDIDATES = 5  # n
REFINE_HALF_WIDTH = 0.002
REFINE_N_W0 = 81

KNOWN_FAILURE_W0 = -1.0389
KNOWN_FAILURE_PAIR = (3, 4)
KNOWN_FAILURE_ELL = 355.655882

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "benchmarks" / "data" / "limber_outlier_tomography.npz"
OUTPUT_DIR = ROOT / "benchmarks" / "limber_integrator_stress_test"


def load_tomography() -> tuple[np.ndarray, tuple[np.ndarray, ...], np.ndarray]:
    """Load the frozen tomographic redshift distributions."""
    data = np.load(DATA_FILE)

    z = np.asarray(data["z"], dtype=float)
    centers = np.asarray(data["bin_centers"], dtype=float)
    bins = tuple(
        np.asarray(data[f"nz_{i}"], dtype=float)
        for i in range(len(centers))
    )

    return z, bins, centers


def build_cosmology(w0: float) -> ccl.Cosmology:
    """Build the cosmology for one value of w0."""
    return ccl.Cosmology(
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


def build_tracers(
    cosmo: ccl.Cosmology,
    z: np.ndarray,
    bins: tuple[np.ndarray, ...],
    centers: np.ndarray,
) -> list[ccl.NumberCountsTracer]:
    """Build the number-count tracers for one cosmology."""
    a = 1.0 / (1.0 + centers)
    bias_values = BIAS_PREFACTOR / np.asarray(
        ccl.growth_factor(cosmo, a), dtype=float
    )

    return [
        ccl.NumberCountsTracer(
            cosmo,
            has_rsd=False,
            dndz=(z, nz),
            bias=(z, np.full_like(z, bias_value)),
            mag_bias=None,
        )
        for nz, bias_value in zip(bins, bias_values, strict=True)
    ]


def evaluate_w0(
    args: tuple[
        float,
        np.ndarray,
        tuple[np.ndarray, ...],
        np.ndarray,
        tuple[tuple[int, int], ...],
        np.ndarray,
    ],
) -> tuple[float, np.ndarray]:
    """Evaluate all Limber methods for one cosmology."""
    w0, z, bins, centers, pairs, ell_values = args

    cosmo = build_cosmology(w0)
    tracers = build_tracers(cosmo, z, bins, centers)

    values = np.empty(
        (len(METHODS), len(pairs), len(ell_values)),
        dtype=float,
    )

    for ipair, (i, j) in enumerate(pairs):
        for imethod, method in enumerate(METHODS):
            values[imethod, ipair] = ccl.angular_cl(
                cosmo,
                tracers[i],
                tracers[j],
                ell_values,
                l_limber=-1,
                limber_integration_method=method,
            )

    return float(w0), values


def run_scan(
    w0_values: np.ndarray,
    ell_values: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    z: np.ndarray,
    bins: tuple[np.ndarray, ...],
    centers: np.ndarray,
    *,
    n_workers: int,
) -> np.ndarray:
    """Evaluate a w0, pair, ell, and integration-method grid."""
    jobs = [
        (float(w0), z, bins, centers, pairs, ell_values)
        for w0 in w0_values
    ]

    if n_workers == 1:
        results = [evaluate_w0(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(evaluate_w0, jobs, chunksize=1))

    results.sort(key=lambda item: item[0])

    return np.asarray([values for _, values in results], dtype=float)


def midpoint_residual(values: np.ndarray) -> np.ndarray:
    """Return fractional midpoint residuals along the w0 axis."""
    residual = np.full_like(values, np.nan)
    midpoint = 0.5 * (values[:-2] + values[2:])
    residual[1:-1] = (values[1:-1] - midpoint) / np.abs(midpoint)

    return residual


def relative_difference(
    values: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Return the absolute relative difference from a reference."""
    return np.abs((values - reference) / reference)


def find_worst(
    values: np.ndarray,
) -> tuple[float, int, int, int]:
    """Return the largest finite value and its scan indices."""
    index = np.nanargmax(values)
    iw0, ipair, iell = np.unravel_index(index, values.shape)

    return (
        float(values[iw0, ipair, iell]),
        int(iw0),
        int(ipair),
        int(iell),
    )


def find_top_candidates(
    values: np.ndarray,
    *,
    n_candidates: int,
) -> list[tuple[float, int, int, int]]:
    """Return the largest anomalies in a scan."""
    flat = np.asarray(values, dtype=float).ravel()
    finite = np.flatnonzero(np.isfinite(flat))

    if finite.size == 0:
        return []

    order = finite[np.argsort(flat[finite])[::-1]]
    candidates = []

    for index in order[:n_candidates]:
        iw0, ipair, iell = np.unravel_index(int(index), values.shape)
        candidates.append(
            (
                float(values[iw0, ipair, iell]),
                int(iw0),
                int(ipair),
                int(iell),
            )
        )

    return candidates


def print_worst(
    label: str,
    values: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    w0_values: np.ndarray,
    ell_values: np.ndarray,
) -> None:
    """Print the worst point in a three-dimensional scan."""
    value, iw0, ipair, iell = find_worst(values)

    print(
        f"{label:<24} "
        f"{value:.8e}  "
        f"w0={w0_values[iw0]:.7f}  "
        f"ell={ell_values[iell]:.3f}  "
        f"pair={pairs[ipair]}"
    )


def get_diagnostics(values: np.ndarray) -> dict[str, np.ndarray]:
    """Construct disagreement and smoothness diagnostics."""
    qag = values[:, 0]
    cquad = values[:, 1]
    spline = values[:, 2]

    return {
        "qag_cquad": relative_difference(qag, cquad),
        "spline_cquad": relative_difference(spline, cquad),
        "qag_spline": relative_difference(qag, spline),
        "qag_smoothness": np.abs(midpoint_residual(qag)),
        "cquad_smoothness": np.abs(midpoint_residual(cquad)),
        "spline_smoothness": np.abs(midpoint_residual(spline)),
    }


def print_summary(
    values: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    w0_values: np.ndarray,
    ell_values: np.ndarray,
) -> None:
    """Print method-disagreement and smoothness diagnostics."""
    diagnostics = get_diagnostics(values)

    print()
    print("METHOD DISAGREEMENT")
    print("=" * 100)

    print_worst(
        "qag_quad vs cquad",
        diagnostics["qag_cquad"],
        pairs,
        w0_values,
        ell_values,
    )
    print_worst(
        "spline vs cquad",
        diagnostics["spline_cquad"],
        pairs,
        w0_values,
        ell_values,
    )
    print_worst(
        "qag_quad vs spline",
        diagnostics["qag_spline"],
        pairs,
        w0_values,
        ell_values,
    )

    print()
    print("SMOOTHNESS ALONG w0")
    print("=" * 100)

    print_worst(
        "qag_quad",
        diagnostics["qag_smoothness"],
        pairs,
        w0_values,
        ell_values,
    )
    print_worst(
        "cquad",
        diagnostics["cquad_smoothness"],
        pairs,
        w0_values,
        ell_values,
    )
    print_worst(
        "spline",
        diagnostics["spline_smoothness"],
        pairs,
        w0_values,
        ell_values,
    )


def collect_refinement_candidates(
    values: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> list[tuple[str, float, tuple[int, int], float, float]]:
    """Collect and deduplicate coarse-scan refinement candidates."""
    diagnostics = get_diagnostics(values)

    candidate_diagnostics = (
        ("qag-cquad disagreement", diagnostics["qag_cquad"]),
        ("spline-cquad disagreement", diagnostics["spline_cquad"]),
        ("qag smoothness", diagnostics["qag_smoothness"]),
        ("cquad smoothness", diagnostics["cquad_smoothness"]),
        ("spline smoothness", diagnostics["spline_smoothness"]),
    )

    candidates = []

    for label, diagnostic in candidate_diagnostics:
        top = find_top_candidates(
            diagnostic,
            n_candidates=N_REFINE_CANDIDATES,
        )

        for value, iw0, ipair, iell in top:
            candidates.append(
                (
                    label,
                    value,
                    pairs[ipair],
                    float(ELL_VALUES[iell]),
                    float(W0_VALUES[iw0]),
                )
            )

    candidates.append(
        (
            "known QAG regression",
            np.nan,
            KNOWN_FAILURE_PAIR,
            KNOWN_FAILURE_ELL,
            KNOWN_FAILURE_W0,
        )
    )

    deduplicated = []
    seen = set()

    for candidate in candidates:
        label, value, pair, ell, w0 = candidate
        key = (pair, round(ell, 8), round(w0, 6))

        if key in seen:
            continue

        seen.add(key)
        deduplicated.append((label, value, pair, ell, w0))

    return deduplicated


def evaluate_refinement_point(
    args: tuple[
        float,
        float,
        tuple[int, int],
        np.ndarray,
        tuple[np.ndarray, ...],
        np.ndarray,
    ],
) -> tuple[float, np.ndarray]:
    """Evaluate one refined w0 point for one pair and ell."""
    w0, ell, pair, z, bins, centers = args

    cosmo = build_cosmology(w0)
    tracers = build_tracers(cosmo, z, bins, centers)

    i, j = pair
    values = np.empty(len(METHODS), dtype=float)
    ell_array = np.asarray([ell], dtype=float)

    for imethod, method in enumerate(METHODS):
        cl = ccl.angular_cl(
            cosmo,
            tracers[i],
            tracers[j],
            ell_array,
            l_limber=-1,
            limber_integration_method=method,
        )
        values[imethod] = np.asarray(cl, dtype=float)[0]

    return float(w0), values


def run_refined_scan(
    center_w0: float,
    ell: float,
    pair: tuple[int, int],
    z: np.ndarray,
    bins: tuple[np.ndarray, ...],
    centers: np.ndarray,
    *,
    n_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a dense local w0 scan around one anomaly."""
    w0_values = np.linspace(
        center_w0 - REFINE_HALF_WIDTH,
        center_w0 + REFINE_HALF_WIDTH,
        REFINE_N_W0,
    )

    jobs = [
        (float(w0), float(ell), pair, z, bins, centers)
        for w0 in w0_values
    ]

    if n_workers == 1:
        results = [evaluate_refinement_point(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                executor.map(evaluate_refinement_point, jobs, chunksize=1)
            )

    results.sort(key=lambda item: item[0])
    values = np.asarray([value for _, value in results], dtype=float)

    return w0_values, values


def refined_midpoint_residual(values: np.ndarray) -> np.ndarray:
    """Return midpoint residuals for a one-dimensional refined scan."""
    residual = np.full(values.shape, np.nan, dtype=float)
    midpoint = 0.5 * (values[:-2] + values[2:])
    residual[1:-1] = np.abs((values[1:-1] - midpoint) / midpoint)

    return residual


def refined_diagnostics(values: np.ndarray) -> dict[str, np.ndarray]:
    """Construct diagnostics for one local refinement."""
    qag = values[:, 0]
    cquad = values[:, 1]
    spline = values[:, 2]

    return {
        "qag_cquad": relative_difference(qag, cquad),
        "spline_cquad": relative_difference(spline, cquad),
        "qag_spline": relative_difference(qag, spline),
        "qag_smoothness": refined_midpoint_residual(qag),
        "cquad_smoothness": refined_midpoint_residual(cquad),
        "spline_smoothness": refined_midpoint_residual(spline),
    }


def maximum_1d(
    values: np.ndarray,
    w0_values: np.ndarray,
) -> tuple[float, float]:
    """Return the maximum finite value and corresponding w0."""
    index = int(np.nanargmax(values))

    return float(values[index]), float(w0_values[index])


def print_refined_result(
    index: int,
    label: str,
    coarse_value: float,
    pair: tuple[int, int],
    ell: float,
    center_w0: float,
    w0_values: np.ndarray,
    values: np.ndarray,
) -> dict[str, float]:
    """Print diagnostics for one refined anomaly."""
    diagnostics = refined_diagnostics(values)

    qag_cquad, qag_cquad_w0 = maximum_1d(
        diagnostics["qag_cquad"], w0_values
    )
    spline_cquad, spline_cquad_w0 = maximum_1d(
        diagnostics["spline_cquad"], w0_values
    )
    qag_smooth, qag_smooth_w0 = maximum_1d(
        diagnostics["qag_smoothness"], w0_values
    )
    cquad_smooth, cquad_smooth_w0 = maximum_1d(
        diagnostics["cquad_smoothness"], w0_values
    )
    spline_smooth, spline_smooth_w0 = maximum_1d(
        diagnostics["spline_smoothness"], w0_values
    )

    print()
    print(f"[{index:02d}] {label}")
    print("-" * 100)
    print(f"pair              = {pair}")
    print(f"ell               = {ell:.9f}")
    print(f"coarse center w0  = {center_w0:.7f}")

    if np.isfinite(coarse_value):
        print(f"coarse diagnostic = {coarse_value:.8e}")

    print(
        f"refined range     = "
        f"{w0_values[0]:.7f} -- {w0_values[-1]:.7f}"
    )
    print(
        f"refined spacing   = "
        f"{w0_values[1] - w0_values[0]:.8f}"
    )
    print()
    print(
        f"qag-cquad         = {qag_cquad:.8e}  "
        f"at w0={qag_cquad_w0:.7f}"
    )
    print(
        f"spline-cquad      = {spline_cquad:.8e}  "
        f"at w0={spline_cquad_w0:.7f}"
    )
    print(
        f"qag smoothness    = {qag_smooth:.8e}  "
        f"at w0={qag_smooth_w0:.7f}"
    )
    print(
        f"cquad smoothness  = {cquad_smooth:.8e}  "
        f"at w0={cquad_smooth_w0:.7f}"
    )
    print(
        f"spline smoothness = {spline_smooth:.8e}  "
        f"at w0={spline_smooth_w0:.7f}"
    )

    return {
        "qag_cquad": qag_cquad,
        "qag_cquad_w0": qag_cquad_w0,
        "spline_cquad": spline_cquad,
        "spline_cquad_w0": spline_cquad_w0,
        "qag_smoothness": qag_smooth,
        "qag_smoothness_w0": qag_smooth_w0,
        "cquad_smoothness": cquad_smooth,
        "cquad_smoothness_w0": cquad_smooth_w0,
        "spline_smoothness": spline_smooth,
        "spline_smoothness_w0": spline_smooth_w0,
    }


def plot_refined_case(
    index: int,
    label: str,
    pair: tuple[int, int],
    ell: float,
    w0_values: np.ndarray,
    values: np.ndarray,
) -> None:
    """Plot one dense local refinement."""
    qag = values[:, 0]
    cquad = values[:, 1]
    spline = values[:, 2]

    stem = (
        f"refined_{index:02d}_pair_{pair[0]}_{pair[1]}_ell_{ell:.3f}"
    ).replace(".", "p")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(w0_values, qag, label="qag_quad")
    ax.plot(w0_values, cquad, linestyle="--", label="cquad")
    ax.plot(w0_values, spline, linestyle=":", label="spline")

    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$C_\ell$")
    ax.set_title(rf"{label}: $\ell={ell:.3f}$, bins {pair}")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_cl.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        w0_values,
        relative_difference(qag, cquad),
        label="qag_quad vs cquad",
    )
    ax.plot(
        w0_values,
        relative_difference(spline, cquad),
        linestyle="--",
        label="spline vs cquad",
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel("absolute relative difference")
    ax.set_title(rf"{label}: $\ell={ell:.3f}$, bins {pair}")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{stem}_difference.png", dpi=200)
    plt.close(fig)


def run_refinements(
    values: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    z: np.ndarray,
    bins: tuple[np.ndarray, ...],
    centers: np.ndarray,
    *,
    n_workers: int,
) -> None:
    """Refine the worst coarse-scan anomalies."""
    candidates = collect_refinement_candidates(values, pairs)

    print()
    print("REFINEMENT CANDIDATES")
    print("=" * 100)
    print(
        f"candidate diagnostics per category = "
        f"{N_REFINE_CANDIDATES}"
    )
    print(f"unique refinement locations         = {len(candidates)}")
    print(f"local half-width                    = {REFINE_HALF_WIDTH:.6f}")
    print(f"local w0 samples                    = {REFINE_N_W0}")
    print(
        f"local w0 spacing                    = "
        f"{2.0 * REFINE_HALF_WIDTH / (REFINE_N_W0 - 1):.8f}"
    )

    summary_rows = []

    for index, candidate in enumerate(candidates, start=1):
        label, coarse_value, pair, ell, center_w0 = candidate

        w0_values, refined_values = run_refined_scan(
            center_w0,
            ell,
            pair,
            z,
            bins,
            centers,
            n_workers=n_workers,
        )

        result = print_refined_result(
            index,
            label,
            coarse_value,
            pair,
            ell,
            center_w0,
            w0_values,
            refined_values,
        )

        plot_refined_case(
            index,
            label,
            pair,
            ell,
            w0_values,
            refined_values,
        )

        summary_rows.append(
            (index, label, pair, ell, center_w0, result)
        )

        np.savez(
            OUTPUT_DIR / f"refined_{index:02d}.npz",
            w0=w0_values,
            pair=np.asarray(pair, dtype=int),
            ell=float(ell),
            methods=np.asarray(METHODS),
            values=refined_values,
        )

    print()
    print("REFINED ANOMALY SUMMARY")
    print("=" * 140)

    print(
        f"{'#':>3}  "
        f"{'source':<27} "
        f"{'pair':<8} "
        f"{'ell':>10} "
        f"{'center w0':>11} "
        f"{'max qag-cquad':>16} "
        f"{'w0':>11} "
        f"{'max spline-cquad':>18} "
        f"{'w0':>11}"
    )

    print("-" * 140)

    for index, label, pair, ell, center_w0, result in summary_rows:
        print(
            f"{index:>3d}  "
            f"{label:<27} "
            f"{str(pair):<8} "
            f"{ell:>10.3f} "
            f"{center_w0:>11.6f} "
            f"{result['qag_cquad']:>16.8e} "
            f"{result['qag_cquad_w0']:>11.6f} "
            f"{result['spline_cquad']:>18.8e} "
            f"{result['spline_cquad_w0']:>11.6f}"
        )


def plot_accuracy_heatmap(
    name: str,
    difference: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    title: str,
) -> None:
    """Plot maximum method disagreement over w0."""
    maximum = np.nanmax(difference, axis=0)
    maximum = np.maximum(maximum, np.finfo(float).tiny)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    image = ax.imshow(
        np.log10(maximum),
        aspect="auto",
        origin="lower",
        extent=(
            np.log10(ELL_VALUES[0]),
            np.log10(ELL_VALUES[-1]),
            -0.5,
            len(pairs) - 0.5,
        ),
    )

    ax.set_xlabel(r"$\log_{10}\ell$")
    ax.set_ylabel("tomographic pair")
    ax.set_yticks(np.arange(len(pairs)))
    ax.set_yticklabels([f"{i} × {j}" for i, j in pairs])
    ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\log_{10}$ maximum relative difference")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=200)
    plt.close(fig)


def plot_smoothness_heatmap(
    name: str,
    values: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
    title: str,
) -> None:
    """Plot maximum w0 smoothness residual for each pair and ell."""
    residual = np.abs(midpoint_residual(values))
    maximum = np.nanmax(residual, axis=0)
    maximum = np.maximum(maximum, np.finfo(float).tiny)

    fig, ax = plt.subplots(figsize=(9.0, 6.0))

    image = ax.imshow(
        np.log10(maximum),
        aspect="auto",
        origin="lower",
        extent=(
            np.log10(ELL_VALUES[0]),
            np.log10(ELL_VALUES[-1]),
            -0.5,
            len(pairs) - 0.5,
        ),
    )

    ax.set_xlabel(r"$\log_{10}\ell$")
    ax.set_ylabel("tomographic pair")
    ax.set_yticks(np.arange(len(pairs)))
    ax.set_yticklabels([f"{i} × {j}" for i, j in pairs])
    ax.set_title(title)

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\log_{10}$ maximum midpoint residual")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=200)
    plt.close(fig)


def plot_worst_case(
    values: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> None:
    """Plot the coarse scan containing the largest QAG-CQUAD disagreement."""
    qag = values[:, 0]
    cquad = values[:, 1]
    spline = values[:, 2]

    difference = relative_difference(qag, cquad)
    _, _, ipair, iell = find_worst(difference)

    pair = pairs[ipair]
    ell = ELL_VALUES[iell]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(W0_VALUES, qag[:, ipair, iell], label="qag_quad")
    ax.plot(
        W0_VALUES,
        cquad[:, ipair, iell],
        linestyle="--",
        label="cquad",
    )
    ax.plot(
        W0_VALUES,
        spline[:, ipair, iell],
        linestyle=":",
        label="spline",
    )

    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$C_\ell$")
    ax.set_title(
        rf"Worst coarse QAG-CQUAD disagreement: "
        rf"$\ell={ell:.1f}$, bins {pair}"
    )
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "coarse_worst_case_scan.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        W0_VALUES,
        relative_difference(qag[:, ipair, iell], cquad[:, ipair, iell]),
        label="qag_quad vs cquad",
    )
    ax.plot(
        W0_VALUES,
        relative_difference(spline[:, ipair, iell], cquad[:, ipair, iell]),
        linestyle="--",
        label="spline vs cquad",
    )

    ax.set_yscale("log")
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel("absolute relative difference")
    ax.set_title(
        rf"Worst coarse method disagreement: "
        rf"$\ell={ell:.1f}$, bins {pair}"
    )
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "coarse_worst_case_difference.png", dpi=200)
    plt.close(fig)


def save_results(
    values: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> None:
    """Save the complete coarse stress-test result grid."""
    np.savez(
        OUTPUT_DIR / "limber_integrator_stress_test.npz",
        w0=W0_VALUES,
        ell=ELL_VALUES,
        pairs=np.asarray(pairs, dtype=int),
        methods=np.asarray(METHODS),
        values=values,
    )


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
    """Run the CCL Limber integration stress test."""
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    z, bins, centers = load_tomography()
    pairs = tuple(combinations_with_replacement(range(len(bins)), 2))

    print()
    print("CCL Limber integration stress test")
    print("=" * 100)
    print(f"tomography file   = {DATA_FILE}")
    print(f"tomographic bins  = {len(bins)}")
    print(f"z samples         = {len(z)}")
    print(f"bin centers       = {centers}")
    print(f"outlier fraction  = {OUTLIER_FRAC:.2f}")
    print(f"tomographic pairs = {len(pairs)}")
    print(f"ell range         = {ELL_VALUES[0]:.1f} -- {ELL_VALUES[-1]:.1f}")
    print(f"ell samples       = {len(ELL_VALUES)}")
    print(f"w0 range          = {W0_VALUES[0]:.4f} -- {W0_VALUES[-1]:.4f}")
    print(f"w0 samples        = {len(W0_VALUES)}")
    print(f"coarse w0 spacing = {W0_VALUES[1] - W0_VALUES[0]:.6f}")
    print(f"workers           = {args.n_workers}")
    print("integration modes = qag_quad, cquad, spline")
    print(f"output directory  = {OUTPUT_DIR}")

    print()
    print("COARSE DISCOVERY SCAN")
    print("=" * 100)
    print(
        f"Evaluating {len(W0_VALUES)} cosmologies, "
        f"{len(pairs)} pairs, {len(ELL_VALUES)} ell values..."
    )

    values = run_scan(
        W0_VALUES,
        ELL_VALUES,
        pairs,
        z,
        bins,
        centers,
        n_workers=args.n_workers,
    )

    print_summary(values, pairs, W0_VALUES, ELL_VALUES)
    save_results(values, pairs)

    qag = values[:, 0]
    cquad = values[:, 1]
    spline = values[:, 2]

    plot_accuracy_heatmap(
        "qag_vs_cquad.png",
        relative_difference(qag, cquad),
        pairs,
        "QAG vs CQUAD",
    )
    plot_accuracy_heatmap(
        "spline_vs_cquad.png",
        relative_difference(spline, cquad),
        pairs,
        "Spline vs CQUAD",
    )
    plot_smoothness_heatmap(
        "qag_smoothness.png",
        qag,
        pairs,
        "QAG smoothness along $w_0$",
    )
    plot_smoothness_heatmap(
        "cquad_smoothness.png",
        cquad,
        pairs,
        "CQUAD smoothness along $w_0$",
    )
    plot_smoothness_heatmap(
        "spline_smoothness.png",
        spline,
        pairs,
        "Spline smoothness along $w_0$",
    )

    plot_worst_case(values, pairs)

    print()
    print("ADAPTIVE LOCAL REFINEMENT")
    print("=" * 100)

    run_refinements(
        values,
        pairs,
        z,
        bins,
        centers,
        n_workers=args.n_workers,
    )

    print()
    print("OUTPUT")
    print("=" * 100)
    print(OUTPUT_DIR / "limber_integrator_stress_test.npz")
    print()
    print("Plots and refined scans written to:")
    print(f"  {OUTPUT_DIR}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
