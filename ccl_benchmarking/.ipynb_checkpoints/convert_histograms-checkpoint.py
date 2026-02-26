#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def mid_to_edges(zmid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given bin centers zmid, infer (zlow, zhigh) assuming midpoints between centers."""
    zmid = np.asarray(zmid, dtype=float)
    if zmid.ndim != 1 or zmid.size < 2:
        raise ValueError("z_mid must be a 1D array with at least 2 points.")
    dz = np.diff(zmid)
    if not np.all(dz > 0):
        raise ValueError("z_mid must be strictly increasing.")
    edges = np.empty(zmid.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (zmid[1:] + zmid[:-1])
    edges[0] = zmid[0] - 0.5 * dz[0]
    edges[-1] = zmid[-1] + 0.5 * dz[-1]
    return edges[:-1], edges[1:]


def build_uniform_grid(zlow_min: float, dz: float, zmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build uniform (zlow, zmid, zhigh) grid with bin width dz, reaching at least zmax in zhigh."""
    if dz <= 0:
        raise ValueError("--dz must be > 0.")
    if zmax <= zlow_min:
        raise ValueError("--zmax must be > --zlow_min.")
    # ensure last edge >= zmax
    n = int(np.ceil((zmax - zlow_min) / dz))
    edges = zlow_min + dz * np.arange(n + 1, dtype=float)
    zlow = edges[:-1]
    zhigh = edges[1:]
    zmid = 0.5 * (zlow + zhigh)
    return zlow, zmid, zhigh


def trapz_norm(z: np.ndarray, nz: np.ndarray) -> float:
    return float(np.trapz(nz, z))


def load_histo_2col(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError(f"Expected >=2 columns in {path}, got shape {arr.shape}")
    z = arr[:, 0].astype(float)
    nz = arr[:, 1].astype(float)
    return z, nz


def load_des_zgrid_3col(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a DES-style nz file and return (zlow, zmid, zhigh) from the first 3 cols.
    Any extra BIN columns are ignored.
    """
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RuntimeError(f"Expected >=3 columns in {path}, got shape {arr.shape}")
    zlow = arr[:, 0].astype(float)
    zmid = arr[:, 1].astype(float)
    zhigh = arr[:, 2].astype(float)
    if not np.all(np.diff(zmid) > 0):
        raise ValueError(f"Reference z_mid in {path} must be strictly increasing.")
    return zlow, zmid, zhigh


def interp_to_grid(z_in: np.ndarray, nz_in: np.ndarray, z_out: np.ndarray) -> np.ndarray:
    """Linear interpolation with n(z)=0 outside input z-range (typical for histograms)."""
    z_in = np.asarray(z_in, dtype=float)
    nz_in = np.asarray(nz_in, dtype=float)
    z_out = np.asarray(z_out, dtype=float)

    if z_in.ndim != 1 or z_out.ndim != 1:
        raise ValueError("z_in and z_out must be 1D arrays.")
    if z_in.size < 2:
        raise ValueError("z_in must have at least 2 points for interpolation.")
    if not np.all(np.diff(z_in) > 0):
        raise ValueError("z_in must be strictly increasing.")
    return np.interp(z_out, z_in, nz_in, left=0.0, right=0.0)


def parse_int_list(s: str) -> list[int]:
    if s is None:
        return []
    out: list[int] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    # Old mode (single file)
    ap.add_argument("--input_file", type=str, default=None, help="2-col file: z_mid  n(z)")

    # New mode (multi file)
    ap.add_argument(
        "--bin_files",
        type=str,
        default=None,
        help='Comma-separated list of 2-col histo files, e.g. "bin1_histo.txt,bin2_histo.txt"',
    )
    ap.add_argument(
        "--bin_cols",
        type=str,
        default=None,
        help='Comma-separated list of BIN columns (1-based) to fill, e.g. "1,2"',
    )

    ap.add_argument("--output_file", required=True, type=str, help="DES-style nz output")
    ap.add_argument("--nbins", type=int, default=4, help="Number of BIN columns to write")

    ap.add_argument(
        "--zgrid_file",
        type=str,
        default=None,
        help=(
            "Optional DES-style reference file whose (Z_LOW,Z_MID,Z_HIGH) grid will be used. "
            "We ignore any BIN columns in that file and only use its first 3 columns."
        ),
    )

    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize each input n(z) so that integral dz = 1 (done AFTER interpolation if output grid differs).",
    )
    ap.add_argument("--plot", action="store_true", help="Save a quick diagnostic plot PNG.")

    # Kept for backwards compatibility (single-file fill)
    ap.add_argument(
        "--fill_bins",
        type=str,
        default="1",
        help='(single-file mode) BIN indices to fill with SAME n(z). Example: "1,2". Default: "1"',
    )

    ap.add_argument(
        "--zlow_min",
        type=float,
        default=None,
        help="If set (and no --zgrid_file), build a uniform output grid starting at this Z_LOW (e.g. 1e-4).",
    )
    ap.add_argument(
        "--dz",
        type=float,
        default=None,
        help="If set with --zlow_min (and no --zgrid_file), use uniform bin width dz (e.g. 0.01).",
    )
    ap.add_argument(
        "--zmax",
        type=float,
        default=None,
        help="If set with --zlow_min/--dz (and no --zgrid_file), extend grid up to at least this z (in Z_HIGH).",
    )

    args = ap.parse_args()

    outpath = Path(args.output_file)
    nbins = int(args.nbins)
    if nbins < 1:
        raise ValueError("--nbins must be >= 1")

    # Decide output z-grid
    use_ref_grid = args.zgrid_file is not None
    use_uniform_grid = (args.zlow_min is not None) or (args.dz is not None) or (args.zmax is not None)

    if use_ref_grid and use_uniform_grid:
        raise ValueError("Use either --zgrid_file OR (--zlow_min/--dz/--zmax), not both.")

    if use_ref_grid:
        zlow_out, zmid_out, zhigh_out = load_des_zgrid_3col(Path(args.zgrid_file))
    elif use_uniform_grid:
        if args.zlow_min is None or args.dz is None or args.zmax is None:
            raise ValueError("Uniform grid mode requires all of --zlow_min, --dz, --zmax.")
        zlow_out, zmid_out, zhigh_out = build_uniform_grid(float(args.zlow_min), float(args.dz), float(args.zmax))
    else:
        # We will define from the first input histogram we see (single or multi mode)
        zlow_out = zmid_out = zhigh_out = None  # type: ignore[assignment]

    # For plotting: store original inputs used for each BIN column (0-based index)
    orig_for_bin: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # Decide mode
    use_multi = (args.bin_files is not None) or (args.bin_cols is not None)

    if use_multi:
        if args.bin_files is None or args.bin_cols is None:
            raise ValueError("Multi-file mode requires BOTH --bin_files and --bin_cols.")
        files = [Path(x.strip()) for x in args.bin_files.split(",") if x.strip()]
        cols = parse_int_list(args.bin_cols)
        if len(files) != len(cols):
            raise ValueError(f"--bin_files has {len(files)} items but --bin_cols has {len(cols)} items.")
        for c in cols:
            if not (1 <= c <= nbins):
                raise ValueError(f"--bin_cols contains {c} but --nbins={nbins}")

        # If we still don't have an output grid, define it from the first file
        if zmid_out is None:
            zmid0, _ = load_histo_2col(files[0])
            if not np.all(np.diff(zmid0) > 0):
                raise ValueError(f"z_mid in {files[0]} must be strictly increasing.")
            zlow_out, zhigh_out = mid_to_edges(zmid0)
            zmid_out = zmid0

        nz_cols = np.zeros((zmid_out.size, nbins), dtype=float)

        for f, c in zip(files, cols):
            zmid_in, nz_in = load_histo_2col(f)
            if not np.all(np.diff(zmid_in) > 0):
                raise ValueError(f"z_mid in {f} must be strictly increasing.")

            # --- store ORIGINAL for plotting (optionally normalized on its OWN grid) ---
            nz_orig_plot = nz_in.copy()
            if args.normalize:
                integ0 = trapz_norm(zmid_in, nz_orig_plot)
                if integ0 <= 0:
                    raise RuntimeError(f"Cannot normalize {f}: integral is {integ0}")
                nz_orig_plot = nz_orig_plot / integ0
            orig_for_bin[c - 1] = (zmid_in.copy(), nz_orig_plot)

            # --- interpolate to OUTPUT grid (ref/uniform/first-file-defined) ---
            nz_out = interp_to_grid(zmid_in, nz_in, zmid_out)

            if args.normalize:
                integ = trapz_norm(zmid_out, nz_out)
                if integ <= 0:
                    raise RuntimeError(f"Cannot normalize {f} on output grid: integral is {integ}")
                nz_out = nz_out / integ

            nz_cols[:, c - 1] = nz_out

    else:
        if args.input_file is None:
            raise ValueError("Single-file mode requires --input_file.")
        inpath = Path(args.input_file)
        zmid_in, nz_in = load_histo_2col(inpath)
        if not np.all(np.diff(zmid_in) > 0):
            raise ValueError(f"z_mid in {inpath} must be strictly increasing.")

        # If we still don't have an output grid, define it from this input
        if zmid_out is None:
            zlow_out, zhigh_out = mid_to_edges(zmid_in)
            zmid_out = zmid_in

        # store ORIGINAL once (for plotting), potentially normalized on input grid
        nz_orig_plot = nz_in.copy()
        if args.normalize:
            integ0 = trapz_norm(zmid_in, nz_orig_plot)
            if integ0 <= 0:
                raise RuntimeError(f"Cannot normalize {inpath}: integral is {integ0}")
            nz_orig_plot = nz_orig_plot / integ0

        # interpolate to output grid if needed
        nz_out = interp_to_grid(zmid_in, nz_in, zmid_out)

        if args.normalize:
            integ = trapz_norm(zmid_out, nz_out)
            if integ <= 0:
                raise RuntimeError(f"Cannot normalize {inpath} on output grid: integral is {integ}")
            nz_out = nz_out / integ

        nz_cols = np.zeros((zmid_out.size, nbins), dtype=float)
        fill_bins = parse_int_list(args.fill_bins)
        for b in fill_bins:
            if not (1 <= b <= nbins):
                raise ValueError(f"--fill_bins contains {b}, but --nbins={nbins}")
            nz_cols[:, b - 1] = nz_out
            orig_for_bin[b - 1] = (zmid_in.copy(), nz_orig_plot.copy())

    out = np.column_stack([zlow_out, zmid_out, zhigh_out, nz_cols])
    bin_names = " ".join([f"BIN{i}" for i in range(1, nbins + 1)])
    header = f"# Z_LOW Z_MID Z_HIGH {bin_names}"
    np.savetxt(outpath, out, fmt="%.6e", header=header, comments="")
    print(f"Wrote: {outpath}")
    print(f"Rows: {out.shape[0]}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        zplot = out[:, 1]
        plt.figure()

        # Plot interpolated/output curves (lines) and original inputs (red dots)
        for i in range(nbins):
            y_out = out[:, 3 + i]
            if np.any(y_out != 0):
                plt.plot(zplot, y_out, label=f"BIN{i+1}")
                if i in orig_for_bin:
                    z0, y0 = orig_for_bin[i]
                    plt.plot(z0, y0, "r.", ms=4, alpha=0.8)

        plt.xlabel("z")
        plt.ylabel("n(z) (arb.)")
        plt.title(outpath.name)
        plt.legend()
        plt.tight_layout()
        figfile = outpath.with_suffix(outpath.suffix + ".png")
        plt.savefig(figfile, dpi=200)
        print(f"Saved plot: {figfile}")


if __name__ == "__main__":
    main()