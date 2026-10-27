"""Generate frozen tomography data for the CCL Limber regression test.

Run example:
    python examples/generate_limber_outlier_tomography.py

"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
from binny import NZTomography


__all__ = ["main"]


YEAR = "1"

MEAN_OFFSET = 0.0
SCATTER_SCALE = 0.03
OUTLIER_FRAC = 0.30
OUTLIER_SCATTER_SCALE = 0.30

OUTPUT = Path("benchmarks/data/limber_outlier_tomography.npz")
PROVENANCE = Path("benchmarks/data/limber_outlier_tomography.txt")


def get_bin_centers(stats: dict) -> np.ndarray:
    """Extract bin centers reported by Binny."""
    for key in ("center", "centers", "mean", "means"):
        if key not in stats:
            continue

        values = stats[key]

        if isinstance(values, dict):
            return np.asarray(
                [values[key] for key in sorted(values)],
                dtype=float,
            )

        return np.asarray(values, dtype=float)

    raise KeyError(
        "Could not find bin centers in Binny shape_stats(). "
        f"Available keys: {list(stats)}"
    )


def main() -> None:
    """Generate and save the tomography used by the CCL regression."""
    tomo = NZTomography()

    tomography = tomo.build_survey_bins(
        "lsst",
        role="lens",
        year=YEAR,
        overrides={
            "uncertainties": {
                "mean_offset": MEAN_OFFSET,
                "scatter_scale": SCATTER_SCALE,
                "outlier_frac": OUTLIER_FRAC,
                "outlier_scatter_scale": OUTLIER_SCATTER_SCALE,
            },
        },
    )

    centers = get_bin_centers(tomo.shape_stats())

    z = np.asarray(tomography.z, dtype=float)
    bin_keys = sorted(tomography.bins)

    data = {
        "z": z,
        "bin_centers": centers,
    }

    for key in bin_keys:
        data[f"nz_{key}"] = np.asarray(
            tomography.bins[key],
            dtype=float,
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT,
        **data,
    )

    binny_version = version("pybinny")

    provenance = f"""CCL Limber integration regression tomography

Source
------
Generated with Binny using the LSST Y1 lens preset.
Binny version: {binny_version}

Binny configuration
-------------------
survey: lsst
role: lens
year: {YEAR}
mean_offset: {MEAN_OFFSET}
scatter_scale: {SCATTER_SCALE}
outlier_frac: {OUTLIER_FRAC}
outlier_scatter_scale: {OUTLIER_SCATTER_SCALE}

Frozen data
-----------
z samples: {z.size}
tomographic bins: {len(bin_keys)}
bin centers: {", ".join(f"{center:.8f}" for center in centers)}

Contents
--------
z: common true-redshift grid
bin_centers: Binny-reported tomographic bin centers
nz_0 ... nz_{len(bin_keys) - 1}: normalized tomographic redshift distributions

Purpose
-------
Frozen input for reproducing the CCL Limber integration discontinuity without
requiring Binny as a runtime dependency.
"""

    PROVENANCE.write_text(provenance)

    print()
    print("Frozen CCL Limber tomography")
    print("=" * 78)
    print("Binny preset           = LSST Y1 lens")
    print(f"Binny version          = {binny_version}")
    print(f"mean offset            = {MEAN_OFFSET}")
    print(f"scatter scale          = {SCATTER_SCALE}")
    print(f"outlier fraction       = {OUTLIER_FRAC}")
    print(f"outlier scatter scale  = {OUTLIER_SCATTER_SCALE}")
    print(f"z samples              = {z.size}")
    print(f"tomographic bins       = {len(bin_keys)}")
    print()

    print("TOMOGRAPHY")
    print("=" * 78)

    for i, key in enumerate(bin_keys):
        nz = data[f"nz_{key}"]

        print(
            f"bin {key}: "
            f"center={centers[i]:.8f}  "
            f"integral={np.trapezoid(nz, z):.10e}  "
            f"min={np.min(nz):.10e}  "
            f"max={np.max(nz):.10e}"
        )

    print()
    print("OUTPUT")
    print("=" * 78)
    print(OUTPUT)
    print(PROVENANCE)
    print()

    # Read the file back immediately so that we know the frozen artifact
    # contains exactly what the CCL reproducer will later consume.
    frozen = np.load(OUTPUT)

    print("CONTENTS")
    print("=" * 78)

    for key in frozen.files:
        value = frozen[key]

        print(
            f"{key:<15} "
            f"shape={str(value.shape):<12} "
            f"dtype={value.dtype}"
        )

    print()
    print("bin centers =", frozen["bin_centers"])
    print()
    print("DONE")


if __name__ == "__main__":
    main()
