"""Benchmark the runtime of CCL Limber integration methods.

This microbenchmark compares the runtime of the qag_quad, cquad, and spline
Limber integration methods for repeated angular power-spectrum evaluations.

The cosmology and tracers are constructed before timing, and one angular_cl
evaluation is performed as a warm-up. The reported timings therefore measure
the integration cost for repeated evaluations rather than end-to-end
cosmology and tracer construction.

Run with:

    python benchmarks/benchmark_limber_performance.py \
        --n-ell 200 \
        --repeats 10

For a more stable comparison:

    python benchmarks/benchmark_limber_performance.py \
        --n-ell 1000 \
        --repeats 50
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pyccl as ccl


__all__ = ["main"]


def build_cosmology() -> ccl.Cosmology:
    """Build the benchmark cosmology."""
    return ccl.Cosmology(
        Omega_c=0.2664,
        Omega_b=0.0492,
        h=0.6727,
        n_s=0.9645,
        sigma8=0.831,
        w0=-1.0,
        wa=0.0,
        transfer_function="eisenstein_hu",
        matter_power_spectrum="halofit",
    )


def build_tracers(
    cosmo: ccl.Cosmology,
) -> tuple[ccl.NumberCountsTracer, ccl.NumberCountsTracer]:
    """Build two representative galaxy-clustering tracers."""
    z = np.linspace(0.01, 2.0, 500)

    nz1 = np.exp(-0.5 * ((z - 0.8) / 0.20) ** 2)
    nz2 = np.exp(-0.5 * ((z - 1.0) / 0.20) ** 2)

    bias1 = np.ones_like(z)
    bias2 = np.ones_like(z)

    tracer1 = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=(z, nz1),
        bias=(z, bias1),
    )
    tracer2 = ccl.NumberCountsTracer(
        cosmo,
        has_rsd=False,
        dndz=(z, nz2),
        bias=(z, bias2),
    )

    return tracer1, tracer2


def benchmark(
    method: str,
    ell: np.ndarray,
    *,
    repeats: int,
) -> np.ndarray:
    """Benchmark repeated angular power-spectrum evaluations."""
    cosmo = build_cosmology()
    tracer1, tracer2 = build_tracers(cosmo)

    # Warm up the cosmology and all required internal quantities.
    ccl.angular_cl(
        cosmo,
        tracer1,
        tracer2,
        ell,
        limber_integration_method=method,
    )

    times = []

    for _ in range(repeats):
        start = time.perf_counter()

        ccl.angular_cl(
            cosmo,
            tracer1,
            tracer2,
            ell,
            limber_integration_method=method,
        )

        times.append(time.perf_counter() - start)

    return np.asarray(times)


def summarize(method: str, times: np.ndarray) -> None:
    """Print timing statistics."""
    print(f"{method}")
    print(f"  runs   = {len(times)}")
    print(f"  mean   = {np.mean(times):.6f} s")
    print(f"  median = {np.median(times):.6f} s")
    print(f"  std    = {np.std(times):.6f} s")
    print(f"  min    = {np.min(times):.6f} s")
    print(f"  max    = {np.max(times):.6f} s")


def main() -> None:
    """Run the Limber integration performance benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-ell", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--ell-min", type=float, default=20.0)
    parser.add_argument("--ell-max", type=float, default=2000.0)
    args = parser.parse_args()

    ell = np.geomspace(
        args.ell_min,
        args.ell_max,
        args.n_ell,
    )

    print()
    print("CCL Limber performance benchmark")
    print("=" * 72)
    print(f"ell range = {args.ell_min:g} -- {args.ell_max:g}")
    print(f"N_ell     = {args.n_ell}")
    print(f"repeats   = {args.repeats}")
    print()

    methods = ("qag_quad", "cquad", "spline")
    results = {}

    for method in methods:
        times = benchmark(
            method,
            ell,
            repeats=args.repeats,
        )
        results[method] = times

        summarize(method, times)
        print()

    qag = np.median(results["qag_quad"])

    print("RELATIVE TIMING")
    print("=" * 72)

    for method in methods:
        median = np.median(results[method])
        print(
            f"{method:10s}: "
            f"{median:.6f} s  "
            f"({median / qag:.3f} x qag_quad)"
        )


if __name__ == "__main__":
    main()
