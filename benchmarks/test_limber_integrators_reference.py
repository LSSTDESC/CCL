"""Compare CCL Limber integration methods against an independent reference.

Run with:
    python benchmarks/test_limber_integrators_reference.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from scipy.integrate import quad


__all__ = ["main"]


OMEGA_C = 0.2664
OMEGA_B = 0.0492
H = 0.6727
N_S = 0.9645
SIGMA8 = 0.831
WA = 0.0

BIAS_PREFACTOR = 1.05

METHODS = ("qag_quad", "cquad", "spline")

REFERENCE_EPSABS = 0.0
REFERENCE_EPSREL = 1.0e-10
REFERENCE_LIMIT = 1000
REFERENCE_SUBINTERVALS = 64

REFINE_HALF_WIDTH = 0.002
REFINE_N_W0 = 81

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "benchmarks" / "data" / "limber_outlier_tomography.npz"
OUTPUT_DIR = ROOT / "benchmarks" / "limber_integrator_reference_test"

TEST_CASES = (
    ("known_qag_regression", (3, 4), 355.655882000, -1.0389000),
    ("qag_44_355", (4, 4), 355.655882008, -0.9639500),
    ("qag_44_242", (4, 4), 242.305531726, -0.9839500),
    ("qag_13_633", (1, 3), 632.455532034, -0.9385500),
    ("qag_13_93", (1, 3), 92.831776672, -1.0387000),
    ("cquad_22_77", (2, 2), 76.623736991, -1.0124000),
    ("cquad_22_242", (2, 2), 242.305531726, -1.0138500),
    ("cquad_12_93", (1, 2), 92.831776672, -1.0390000),
    ("spline_12_20", (1, 2), 20.000000000, -0.9861000),
    ("spline_12_24", (1, 2), 24.230553173, -0.9861500),
    ("spline_12_29", (1, 2), 29.355985352, -0.9861000),
    ("spline_12_1651", (1, 2), 1650.808370536, -0.9861000),
)


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
    """Build a cosmology for one value of w0."""
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
    """Build number-count tracers for one cosmology."""
    a = 1.0 / (1.0 + centers)
    growth = np.asarray(ccl.growth_factor(cosmo, a), dtype=float)
    bias_values = BIAS_PREFACTOR / growth

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


def evaluate_ccl_methods(
    cosmo: ccl.Cosmology,
    tracers: list[ccl.NumberCountsTracer],
    pair: tuple[int, int],
    ell: float,
) -> np.ndarray:
    """Evaluate all CCL Limber integration methods."""
    i, j = pair
    ell_array = np.asarray([ell], dtype=float)
    values = np.empty(len(METHODS), dtype=float)

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

    return values


def tracer_chi_range(
    cosmo: ccl.Cosmology,
    z: np.ndarray,
    nz: np.ndarray,
) -> tuple[float, float]:
    """Return the radial support of a tracer."""
    mask = nz > 0.0

    if not np.any(mask):
        raise RuntimeError("Tracer redshift distribution has no support.")

    z_support = z[mask]
    a_support = 1.0 / (1.0 + z_support)
    chi = np.asarray(ccl.comoving_radial_distance(cosmo, a_support), dtype=float)

    return float(np.min(chi)), float(np.max(chi))


def get_lk_interval(
    cosmo: ccl.Cosmology,
    z: np.ndarray,
    nz1: np.ndarray,
    nz2: np.ndarray,
    ell: float,
) -> tuple[float, float]:
    """Return the Limber integration interval in log(k)."""
    chi_min1, chi_max1 = tracer_chi_range(cosmo, z, nz1)
    chi_min2, chi_max2 = tracer_chi_range(cosmo, z, nz2)

    chi_min = max(chi_min1, chi_min2)
    chi_max = min(chi_max1, chi_max2)

    if chi_max <= chi_min:
        raise RuntimeError("The two tracer redshift distributions do not overlap.")

    k_min = ccl.spline_params.K_MIN
    k_max = ccl.spline_params.K_MAX

    if chi_min <= 0.0:
        chi_min = 0.5 * (ell + 0.5) / k_max

    lk_max = np.log(min(k_max, 2.0 * (ell + 0.5) / chi_min))
    lk_min = np.log(max(k_min, (ell + 0.5) / chi_max))

    return float(lk_min), float(lk_max)


def evaluate_kernel(
    tracer: ccl.NumberCountsTracer,
    chi: float,
) -> np.ndarray:
    """Evaluate all tracer kernels at one comoving distance."""
    values = tracer.get_kernel(np.asarray([chi], dtype=float))
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        return values

    return values[..., 0]


def evaluate_transfer(
    tracer: ccl.NumberCountsTracer,
    lk: float,
    a: float,
) -> np.ndarray:
    """Evaluate all tracer transfer functions at one point."""
    values = tracer.get_transfer(
        lk=np.asarray([lk], dtype=float),
        a=np.asarray([a], dtype=float),
    )
    values = np.asarray(values, dtype=float)

    return values.reshape(values.shape[0], -1)[:, 0]


def evaluate_f_ell(
    tracer: ccl.NumberCountsTracer,
    ell: float,
) -> np.ndarray:
    """Evaluate all tracer ell-prefactors."""
    values = tracer.get_f_ell(np.asarray([ell], dtype=float))
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        return values

    return values[..., 0]


def transfer_limber(
    tracer: ccl.NumberCountsTracer,
    ell: float,
    lk: float,
    chi: float,
    a: float,
) -> float:
    """Evaluate the total Limber transfer factor."""
    kernel = evaluate_kernel(tracer, chi)
    transfer = evaluate_transfer(tracer, lk, a)
    f_ell = evaluate_f_ell(tracer, ell)

    if not (kernel.size == transfer.size == f_ell.size):
        raise RuntimeError(
            "Tracer component sizes disagree: "
            f"kernel={kernel.shape}, transfer={transfer.shape}, "
            f"f_ell={f_ell.shape}"
        )

    return float(np.sum(kernel * transfer * f_ell))


def reference_integrand(
    lk: float,
    cosmo: ccl.Cosmology,
    tracer1: ccl.NumberCountsTracer,
    tracer2: ccl.NumberCountsTracer,
    ell: float,
) -> float:
    """Evaluate the CCL Limber integrand in log(k)."""
    k = np.exp(lk)
    chi = (ell + 0.5) / k

    if not np.isfinite(chi) or chi <= 0.0:
        return 0.0

    a = float(ccl.scale_factor_of_chi(cosmo, chi))

    d1 = transfer_limber(tracer1, ell, lk, chi, a)

    if d1 == 0.0:
        return 0.0

    d2 = transfer_limber(tracer2, ell, lk, chi, a)

    if d2 == 0.0:
        return 0.0

    pk = float(ccl.nonlin_matter_power(cosmo, k, a))

    return k * pk * d1 * d2


def reference_angular_cl(
    cosmo: ccl.Cosmology,
    tracer1: ccl.NumberCountsTracer,
    tracer2: ccl.NumberCountsTracer,
    z: np.ndarray,
    nz1: np.ndarray,
    nz2: np.ndarray,
    ell: float,
) -> tuple[float, float]:
    """Compute the Limber integral using independent SciPy quadrature."""
    lk_min, lk_max = get_lk_interval(cosmo, z, nz1, nz2, ell)
    edges = np.linspace(lk_min, lk_max, REFERENCE_SUBINTERVALS + 1)

    total = 0.0
    error2 = 0.0

    for left, right in zip(edges[:-1], edges[1:], strict=True):
        value, error = quad(
            reference_integrand,
            left,
            right,
            args=(cosmo, tracer1, tracer2, ell),
            epsabs=REFERENCE_EPSABS,
            epsrel=REFERENCE_EPSREL,
            limit=REFERENCE_LIMIT,
        )
        total += value
        error2 += error**2

    normalization = ell + 0.5

    return (
        float(total / normalization),
        float(np.sqrt(error2) / normalization),
    )


def evaluate_point(
    args: tuple[
        float,
        float,
        tuple[int, int],
        np.ndarray,
        tuple[np.ndarray, ...],
        np.ndarray,
    ],
) -> tuple[float, np.ndarray, float, float]:
    """Evaluate CCL methods and the independent reference at one point."""
    w0, ell, pair, z, bins, centers = args

    cosmo = build_cosmology(w0)
    tracers = build_tracers(cosmo, z, bins, centers)

    ccl_values = evaluate_ccl_methods(cosmo, tracers, pair, ell)

    i, j = pair
    reference, reference_error = reference_angular_cl(
        cosmo, tracers[i], tracers[j], z, bins[i], bins[j], ell
    )

    return float(w0), ccl_values, reference, reference_error


def run_case(
    name: str,
    pair: tuple[int, int],
    ell: float,
    center_w0: float,
    z: np.ndarray,
    bins: tuple[np.ndarray, ...],
    centers: np.ndarray,
    *,
    n_workers: int,
) -> dict[str, np.ndarray]:
    """Run a dense reference comparison around one pathological point."""
    w0_values = np.linspace(
        center_w0 - REFINE_HALF_WIDTH,
        center_w0 + REFINE_HALF_WIDTH,
        REFINE_N_W0,
    )

    jobs = [
        (float(w0), ell, pair, z, bins, centers)
        for w0 in w0_values
    ]

    if n_workers == 1:
        results = [evaluate_point(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(evaluate_point, jobs, chunksize=1))

    results.sort(key=lambda item: item[0])

    ccl_values = np.asarray([result[1] for result in results], dtype=float)
    reference = np.asarray([result[2] for result in results], dtype=float)
    reference_error = np.asarray([result[3] for result in results], dtype=float)

    errors = np.empty_like(ccl_values)

    for imethod in range(len(METHODS)):
        errors[:, imethod] = np.abs(
            (ccl_values[:, imethod] - reference) / reference
        )

    result = {
        "w0": w0_values,
        "values": ccl_values,
        "reference": reference,
        "reference_error": reference_error,
        "relative_errors": errors,
    }

    np.savez(
        OUTPUT_DIR / f"{name}.npz",
        w0=w0_values,
        pair=np.asarray(pair, dtype=int),
        ell=float(ell),
        methods=np.asarray(METHODS),
        values=ccl_values,
        reference=reference,
        reference_error=reference_error,
        relative_errors=errors,
    )

    return result


def print_case(
    index: int,
    name: str,
    pair: tuple[int, int],
    ell: float,
    center_w0: float,
    result: dict[str, np.ndarray],
) -> None:
    """Print reference-comparison results for one case."""
    w0 = result["w0"]
    errors = result["relative_errors"]
    reference = result["reference"]
    reference_error = result["reference_error"]

    print()
    print(f"[{index:02d}] {name}")
    print("-" * 100)
    print(f"pair              = {pair}")
    print(f"ell               = {ell:.9f}")
    print(f"center w0         = {center_w0:.7f}")
    print(f"w0 range          = {w0[0]:.7f} -- {w0[-1]:.7f}")
    print(f"w0 spacing        = {w0[1] - w0[0]:.8f}")
    print()

    for imethod, method in enumerate(METHODS):
        i = int(np.nanargmax(errors[:, imethod]))
        print(
            f"{method:<10} max relative error = "
            f"{errors[i, imethod]:.8e} at w0={w0[i]:.7f}"
        )

    relative_reference_error = np.abs(reference_error / reference)
    i = int(np.nanargmax(relative_reference_error))

    print()
    print(
        f"reference estimated error          = "
        f"{relative_reference_error[i]:.8e} at w0={w0[i]:.7f}"
    )


def plot_case(
    name: str,
    pair: tuple[int, int],
    ell: float,
    result: dict[str, np.ndarray],
) -> None:
    """Plot one reference-comparison case."""
    w0 = result["w0"]
    values = result["values"]
    reference = result["reference"]
    errors = result["relative_errors"]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(w0, reference, label="reference")
    ax.plot(w0, values[:, 0], linestyle="--", label="qag_quad")
    ax.plot(w0, values[:, 1], linestyle="-.", label="cquad")
    ax.plot(w0, values[:, 2], linestyle=":", label="spline")

    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$C_\ell$")
    ax.set_title(rf"$\ell={ell:.3f}$, bins {pair}")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}_cl.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for imethod, method in enumerate(METHODS):
        ax.plot(w0, errors[:, imethod], label=method)

    ax.set_yscale("log")
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel("absolute relative error")
    ax.set_title(
        rf"Error relative to reference: $\ell={ell:.3f}$, bins {pair}"
    )
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{name}_relative_error.png",
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
    """Run the independent Limber reference comparison."""
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    z, bins, centers = load_tomography()

    print()
    print("CCL Limber independent reference test")
    print("=" * 100)
    print(f"tomography file       = {DATA_FILE}")
    print(f"tomographic bins      = {len(bins)}")
    print(f"z samples             = {len(z)}")
    print(f"test cases            = {len(TEST_CASES)}")
    print(f"workers               = {args.n_workers}")
    print("reference integrator  = scipy.integrate.quad")
    print(f"reference epsrel      = {REFERENCE_EPSREL:.1e}")
    print(f"reference subintervals= {REFERENCE_SUBINTERVALS}")
    print(f"local w0 samples      = {REFINE_N_W0}")
    print(f"output directory      = {OUTPUT_DIR}")

    summary = []

    for index, (name, pair, ell, center_w0) in enumerate(
        TEST_CASES,
        start=1,
    ):
        result = run_case(
            name,
            pair,
            ell,
            center_w0,
            z,
            bins,
            centers,
            n_workers=args.n_workers,
        )

        print_case(index, name, pair, ell, center_w0, result)
        plot_case(name, pair, ell, result)

        maxima = np.nanmax(result["relative_errors"], axis=0)
        summary.append((name, pair, ell, center_w0, *maxima))

    print()
    print("SUMMARY")
    print("=" * 130)
    print(
        f"{'case':<24} {'pair':<8} {'ell':>10} {'w0':>11} "
        f"{'qag error':>15} {'cquad error':>15} {'spline error':>15}"
    )
    print("-" * 130)

    for name, pair, ell, w0, qag_error, cquad_error, spline_error in summary:
        print(
            f"{name:<24} {str(pair):<8} {ell:>10.3f} {w0:>11.6f} "
            f"{qag_error:>15.8e} {cquad_error:>15.8e} "
            f"{spline_error:>15.8e}"
        )

    np.savez(
        OUTPUT_DIR / "summary.npz",
        names=np.asarray([row[0] for row in summary]),
        pairs=np.asarray([row[1] for row in summary], dtype=int),
        ell=np.asarray([row[2] for row in summary], dtype=float),
        center_w0=np.asarray([row[3] for row in summary], dtype=float),
        qag_error=np.asarray([row[4] for row in summary], dtype=float),
        cquad_error=np.asarray([row[5] for row in summary], dtype=float),
        spline_error=np.asarray([row[6] for row in summary], dtype=float),
    )

    print()
    print("OUTPUT")
    print("=" * 100)
    print(OUTPUT_DIR / "summary.npz")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
