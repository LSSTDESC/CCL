"""Validate the CCL Limber QAG regression against independent integration.

This script compares the CCL Limber integration methods against dense
fixed-grid Simpson integration and independently tests adaptive
Gauss-Kronrod integration on progressively partitioned domains.

Run with:
    python benchmarks/validate_limber_bad_point.py

"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from scipy.integrate import (
    IntegrationWarning,
    cumulative_simpson,
    quad,
    simpson,
)


__all__ = ["main"]


ROOT = Path(__file__).resolve().parents[1]
TOMOGRAPHY_FILE = ROOT / "benchmarks/data/limber_outlier_tomography.npz"
OUTPUT_DIR = ROOT / "benchmarks/limber_bad_point_validation"

METHODS = ("qag_quad", "cquad", "spline")

OMEGA_C = 0.2664
OMEGA_B = 0.0492
H = 0.6727
N_S = 0.9645
SIGMA8 = 0.831
WA = 0.0

BIAS_PREFACTOR = 1.05

PAIR = (3, 4)
ELL = 355.655882000

W0_VALUES = np.array([-1.03895, -1.03890, -1.03885])

GRID_SIZES = (100_001, 200_001)
PLOT_GRID_SIZE = 20_001

QUAD_EPSREL = 1.0e-10
QUAD_LIMIT = 500

SPLIT_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128)


def load_tomography() -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Load the frozen tomography used by the Limber reproducer."""
    frozen = np.load(TOMOGRAPHY_FILE)

    z = np.asarray(frozen["z"], dtype=float)
    centers = np.asarray(frozen["bin_centers"], dtype=float)

    bins = [
        np.asarray(frozen[f"nz_{i}"], dtype=float)
        for i in range(len(centers))
    ]

    return z, bins, centers


def make_cosmology(w0: float) -> ccl.Cosmology:
    """Construct the cosmology used by the Limber reproducer."""
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


def make_tracers(
    cosmo: ccl.Cosmology,
    z: np.ndarray,
    bins: list[np.ndarray],
    bin_centers: np.ndarray,
) -> list[ccl.NumberCountsTracer]:
    """Construct the number-counts tracers used by the reproducer."""
    a = 1.0 / (1.0 + bin_centers)
    bias_values = BIAS_PREFACTOR / np.asarray(
        ccl.growth_factor(cosmo, a),
        dtype=float,
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


def tracer_support(
    cosmo: ccl.Cosmology,
    z: np.ndarray,
    nz: np.ndarray,
) -> tuple[float, float]:
    """Return the comoving-distance support of a redshift distribution."""
    mask = nz > 0.0

    if not np.any(mask):
        raise ValueError("Tracer has no non-zero support.")

    a = 1.0 / (1.0 + z[mask])
    chi = np.asarray(
        ccl.comoving_radial_distance(cosmo, a),
        dtype=float,
    )

    return float(np.min(chi)), float(np.max(chi))


def get_lk_interval(
    cosmo: ccl.Cosmology,
    z: np.ndarray,
    nz1: np.ndarray,
    nz2: np.ndarray,
    ell: float,
) -> tuple[float, float]:
    """Reproduce the Limber integration bounds used by CCL."""
    chi_min1, chi_max1 = tracer_support(cosmo, z, nz1)
    chi_min2, chi_max2 = tracer_support(cosmo, z, nz2)

    chi_min = max(chi_min1, chi_min2)
    chi_max = min(chi_max1, chi_max2)

    k_min = float(ccl.spline_params.K_MIN)
    k_max = float(ccl.spline_params.K_MAX)

    if chi_min <= 0.0:
        chi_min = 0.5 * (ell + 0.5) / k_max

    lk_max = np.log(
        min(k_max, 2.0 * (ell + 0.5) / chi_min)
    )
    lk_min = np.log(
        max(k_min, (ell + 0.5) / chi_max)
    )

    return float(lk_min), float(lk_max)


def tracer_transfer(
    tracer: ccl.Tracer,
    ell: float,
    lk: float,
    chi: float,
    a: float,
    cosmo: ccl.Cosmology,
    psp: ccl.Pk2D,
) -> float:
    """Evaluate the CCL Limber tracer transfer at one point."""
    k = np.exp(lk)
    total = 0.0

    for trc in tracer._trc:
        status = 0
        kernel, status = ccl.lib.cl_tracer_t_get_kernel(
            trc,
            chi,
            status,
        )
        transfer, status = ccl.lib.cl_tracer_t_get_transfer(
            trc,
            lk,
            a,
            status,
        )
        f_ell, status = ccl.lib.cl_tracer_t_get_f_ell(
            trc,
            ell,
            status,
        )
        ccl.check(status, cosmo=cosmo)

        der_bessel = trc.der_bessel

        if der_bessel < 1:
            value = kernel * transfer

            if der_bessel == -1:
                value /= (ell + 0.5) ** 2

        else:
            chi_p = (ell + 1.5) / k
            a_p = float(ccl.scale_factor_of_chi(cosmo, chi_p))

            pk_here = float(psp(k, a, cosmo))
            pk_next = float(psp(k, a_p, cosmo))
            pk_ratio = abs(pk_next / pk_here)

            status = 0
            kernel_p, status = ccl.lib.cl_tracer_t_get_kernel(
                trc,
                chi_p,
                status,
            )
            transfer_p, status = ccl.lib.cl.cl_tracer_t_get_transfer(
                trc,
                lk,
                a_p,
                status,
            )
            ccl.check(status, cosmo=cosmo)

            sqell = np.sqrt(
                (ell + 0.5) * pk_ratio / (ell + 1.5)
            )

            if der_bessel == 1:
                value = (
                    ell
                    * kernel
                    * transfer
                    / (ell + 0.5)
                )
                value -= sqell * kernel_p * transfer_p

            else:
                value = (
                    2.0
                    * sqell
                    * kernel_p
                    * transfer_p
                    / (ell + 1.5)
                )
                value -= (
                    (0.25 + 2.0 * ell)
                    * kernel
                    * transfer
                    / (ell + 0.5) ** 2
                )

        total += value * f_ell

    return float(total)


def limber_integrand(
    lk: float,
    ell: float,
    cosmo: ccl.Cosmology,
    tracer1: ccl.Tracer,
    tracer2: ccl.Tracer,
    psp: ccl.Pk2D,
) -> float:
    """Evaluate the CCL Limber integrand directly."""
    k = np.exp(lk)
    chi = (ell + 0.5) / k
    a = float(ccl.scale_factor_of_chi(cosmo, chi))

    d1 = tracer_transfer(
        tracer1,
        ell,
        lk,
        chi,
        a,
        cosmo,
        psp,
    )

    if d1 == 0.0:
        return 0.0

    d2 = tracer_transfer(
        tracer2,
        ell,
        lk,
        chi,
        a,
        cosmo,
        psp,
    )

    if d2 == 0.0:
        return 0.0

    pk = float(psp(k, a, cosmo))

    return float(k * pk * d1 * d2)


def evaluate_integrand_grid(
    cosmo: ccl.Cosmology,
    tracer1: ccl.Tracer,
    tracer2: ccl.Tracer,
    lk_min: float,
    lk_max: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the Limber integrand on a fixed logarithmic k grid."""
    psp = cosmo.get_nonlin_power()
    lk = np.linspace(lk_min, lk_max, n_grid)

    values = np.asarray(
        [
            limber_integrand(
                x,
                ELL,
                cosmo,
                tracer1,
                tracer2,
                psp,
            )
            for x in lk
        ],
        dtype=float,
    )

    return lk, values


def fixed_grid_cl(
    lk: np.ndarray,
    integrand: np.ndarray,
) -> float:
    """Integrate the Limber integrand using fixed-grid Simpson integration."""
    value = simpson(integrand, x=lk)
    return float(value / (ELL + 0.5))


def adaptive_quad_cl(
    cosmo: ccl.Cosmology,
    tracer1: ccl.Tracer,
    tracer2: ccl.Tracer,
    lk_min: float,
    lk_max: float,
    n_split: int,
) -> tuple[float, float]:
    """Integrate the Limber integrand using split adaptive quadrature.

    The integration domain is divided into equal subintervals in log-k.
    SciPy's adaptive Gauss-Kronrod quadrature is then applied independently
    to every subinterval.

    This is intended to diagnose whether forced domain subdivision resolves
    a localized feature that a single adaptive integration may miss.

    Parameters
    ----------
    cosmo:
        CCL cosmology.
    tracer1:
        First CCL tracer.
    tracer2:
        Second CCL tracer.
    lk_min:
        Lower log-k integration limit.
    lk_max:
        Upper log-k integration limit.
    n_split:
        Number of fixed subintervals.

    Returns
    -------
    tuple[float, float]
        Integrated angular power spectrum and summed absolute quadrature
        error estimate.
    """
    psp = cosmo.get_nonlin_power()

    edges = np.linspace(
        lk_min,
        lk_max,
        n_split + 1,
    )

    total = 0.0
    total_error = 0.0

    def integrand(lk: float) -> float:
        return limber_integrand(
            lk,
            ELL,
            cosmo,
            tracer1,
            tracer2,
            psp,
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            IntegrationWarning,
        )

        for left, right in zip(
            edges[:-1],
            edges[1:],
            strict=True,
        ):
            value, error = quad(
                integrand,
                float(left),
                float(right),
                epsabs=0.0,
                epsrel=QUAD_EPSREL,
                limit=QUAD_LIMIT,
            )

            total += value
            total_error += error

    norm = ELL + 0.5

    return (
        float(total / norm),
        float(total_error / norm),
    )


def ccl_values(
    cosmo: ccl.Cosmology,
    tracer1: ccl.Tracer,
    tracer2: ccl.Tracer,
) -> dict[str, float]:
    """Evaluate all CCL Limber integration methods."""
    return {
        method: float(
            ccl.angular_cl(
                cosmo,
                tracer1,
                tracer2,
                ELL,
                l_limber=-1,
                limber_integration_method=method,
            )
        )
        for method in METHODS
    }


def relative_error(
    value: float,
    reference: float,
) -> float:
    """Return the absolute relative error against a reference value."""
    return abs(value - reference) / abs(reference)


def evaluate_w0(
    w0: float,
    z: np.ndarray,
    bins: list[np.ndarray],
    centers: np.ndarray,
) -> dict:
    """Evaluate all integration diagnostics for one w0 value."""
    cosmo = make_cosmology(w0)
    tracers = make_tracers(
        cosmo,
        z,
        bins,
        centers,
    )

    tracer1 = tracers[PAIR[0]]
    tracer2 = tracers[PAIR[1]]

    nz1 = bins[PAIR[0]]
    nz2 = bins[PAIR[1]]

    lk_min, lk_max = get_lk_interval(
        cosmo,
        z,
        nz1,
        nz2,
        ELL,
    )

    values = ccl_values(
        cosmo,
        tracer1,
        tracer2,
    )

    simpson_values = {}

    for n_grid in GRID_SIZES:
        lk, integrand = evaluate_integrand_grid(
            cosmo,
            tracer1,
            tracer2,
            lk_min,
            lk_max,
            n_grid,
        )

        simpson_values[n_grid] = fixed_grid_cl(
            lk,
            integrand,
        )

    split_quad = {}

    for n_split in SPLIT_COUNTS:
        value, error = adaptive_quad_cl(
            cosmo,
            tracer1,
            tracer2,
            lk_min,
            lk_max,
            n_split,
        )

        split_quad[n_split] = {
            "value": value,
            "error": error,
        }

    return {
        "w0": w0,
        "cosmo": cosmo,
        "tracer1": tracer1,
        "tracer2": tracer2,
        "lk_min": lk_min,
        "lk_max": lk_max,
        "ccl": values,
        "simpson": simpson_values,
        "split_quad": split_quad,
    }


def plot_integrands(
    results: list[dict],
) -> None:
    """Plot the Limber integrand around the QAG regression."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for result in results:
        lk, integrand = evaluate_integrand_grid(
            result["cosmo"],
            result["tracer1"],
            result["tracer2"],
            result["lk_min"],
            result["lk_max"],
            PLOT_GRID_SIZE,
        )

        ax.plot(
            lk,
            integrand,
            label=rf"$w_0={result['w0']:.5f}$",
        )

    ax.set_xlabel(r"$\ln k$")
    ax.set_ylabel(r"Limber integrand")
    ax.set_title(
        r"CCL Limber integrand near the QAG regression"
    )
    ax.legend()
    ax.tick_params(direction="in")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "limber_integrand.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_cumulative_integrals(
    results: list[dict],
) -> None:
    """Plot cumulative fixed-grid Limber integrals."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for result in results:
        lk, integrand = evaluate_integrand_grid(
            result["cosmo"],
            result["tracer1"],
            result["tracer2"],
            result["lk_min"],
            result["lk_max"],
            PLOT_GRID_SIZE,
        )

        cumulative = cumulative_simpson(
            integrand,
            x=lk,
            initial=0.0,
        )
        cumulative /= ELL + 0.5

        ax.plot(
            lk,
            cumulative,
            label=rf"$w_0={result['w0']:.5f}$",
        )

    ax.set_xlabel(r"$\ln k$")
    ax.set_ylabel(r"Cumulative $C_\ell$")
    ax.set_title(r"Cumulative CCL Limber integral")
    ax.legend()
    ax.tick_params(direction="in")

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "limber_cumulative_integral.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def print_integrals(
    results: list[dict],
) -> None:
    """Print CCL and fixed-grid integral values."""
    header = (
        f"{'w0':>12}"
        f"{'qag_quad':>18}"
        f"{'cquad':>18}"
        f"{'spline':>18}"
        f"{'simpson 100k':>18}"
        f"{'simpson 200k':>18}"
    )

    print("INTEGRALS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for result in results:
        values = result["ccl"]
        simpson_values = result["simpson"]

        print(
            f"{result['w0']:>12.7f}"
            f"{values['qag_quad']:>18.10e}"
            f"{values['cquad']:>18.10e}"
            f"{values['spline']:>18.10e}"
            f"{simpson_values[GRID_SIZES[0]]:>18.10e}"
            f"{simpson_values[GRID_SIZES[1]]:>18.10e}"
        )


def print_relative_errors(
    results: list[dict],
) -> None:
    """Print CCL errors against the finest Simpson result."""
    reference_grid = GRID_SIZES[-1]

    print()
    print("RELATIVE ERROR AGAINST SIMPSON 200k")
    print("=" * 92)

    header = (
        f"{'w0':>12}"
        f"{'qag error':>18}"
        f"{'cquad error':>18}"
        f"{'spline error':>18}"
        f"{'100k/200k':>18}"
    )

    print(header)
    print("-" * len(header))

    for result in results:
        reference = result["simpson"][reference_grid]
        values = result["ccl"]

        qag_error = relative_error(
            values["qag_quad"],
            reference,
        )
        cquad_error = relative_error(
            values["cquad"],
            reference,
        )
        spline_error = relative_error(
            values["spline"],
            reference,
        )
        grid_error = relative_error(
            result["simpson"][GRID_SIZES[0]],
            reference,
        )

        print(
            f"{result['w0']:>12.7f}"
            f"{qag_error:>18.8e}"
            f"{cquad_error:>18.8e}"
            f"{spline_error:>18.8e}"
            f"{grid_error:>18.8e}"
        )


def print_simpson_convergence(
    results: list[dict],
) -> None:
    """Print fixed-grid Simpson convergence."""
    print()
    print("SIMPSON CONVERGENCE")
    print("=" * 70)

    for result in results:
        coarse = result["simpson"][GRID_SIZES[0]]
        fine = result["simpson"][GRID_SIZES[1]]

        error = relative_error(
            coarse,
            fine,
        )

        print(
            f"w0={result['w0']:.7f}  "
            f"N={GRID_SIZES[0]} vs N={GRID_SIZES[1]}  "
            f"relative difference={error:.8e}"
        )


def print_split_quad_convergence(
    results: list[dict],
) -> None:
    """Print split adaptive-quadrature convergence."""
    print()
    print("SPLIT ADAPTIVE GAUSS-KRONROD CONVERGENCE")
    print("=" * 118)
    print(
        "SciPy quad is applied independently to equal log-k "
        "subintervals."
    )
    print(
        "Errors below are relative to the 200k-point Simpson reference."
    )
    print()

    header = (
        f"{'splits':>8}"
        + "".join(
            f"{'w0=' + format(result['w0'], '.5f'):>24}"
            for result in results
        )
    )

    print(header)
    print("-" * len(header))

    for n_split in SPLIT_COUNTS:
        row = f"{n_split:>8}"

        for result in results:
            reference = result["simpson"][GRID_SIZES[-1]]
            value = result["split_quad"][n_split]["value"]
            error = relative_error(
                value,
                reference,
            )

            row += f"{error:>24.8e}"

        print(row)

    print()
    print("SPLIT ADAPTIVE INTEGRAL VALUES")
    print("=" * 118)

    header = (
        f"{'splits':>8}"
        + "".join(
            f"{'w0=' + format(result['w0'], '.5f'):>24}"
            for result in results
        )
    )

    print(header)
    print("-" * len(header))

    for n_split in SPLIT_COUNTS:
        row = f"{n_split:>8}"

        for result in results:
            value = result["split_quad"][n_split]["value"]
            row += f"{value:>24.15e}"

        print(row)

    print()
    print("SPLIT ADAPTIVE REPORTED ERROR ESTIMATES")
    print("=" * 118)

    print(header)
    print("-" * len(header))

    for n_split in SPLIT_COUNTS:
        row = f"{n_split:>8}"

        for result in results:
            error = result["split_quad"][n_split]["error"]
            row += f"{error:>24.8e}"

        print(row)


def main() -> None:
    """Validate the pathological Limber point against independent methods."""
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    z, bins, centers = load_tomography()

    print()
    print("CCL Limber bad-point validation")
    print("=" * 120)
    print(f"tomography file = {TOMOGRAPHY_FILE}")
    print(f"pair            = {PAIR}")
    print(f"ell             = {ELL:.12f}")
    print(f"w0 values       = {W0_VALUES}")
    print(f"grid sizes      = {GRID_SIZES}")
    print(f"split counts    = {SPLIT_COUNTS}")
    print(f"quad epsrel     = {QUAD_EPSREL:.1e}")
    print(f"output          = {OUTPUT_DIR}")
    print()

    results = [
        evaluate_w0(
            float(w0),
            z,
            bins,
            centers,
        )
        for w0 in W0_VALUES
    ]

    print_integrals(results)
    print_relative_errors(results)
    print_simpson_convergence(results)
    print_split_quad_convergence(results)

    plot_integrands(results)
    plot_cumulative_integrals(results)

    print()
    print("Plots written to:")
    print(
        f"  {OUTPUT_DIR / 'limber_integrand.png'}"
    )
    print(
        f"  {OUTPUT_DIR / 'limber_cumulative_integral.png'}"
    )
    print()
    print("DONE")


if __name__ == "__main__":
    main()
