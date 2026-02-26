#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from pathlib import Path

# -----------------------------
# User knobs
# -----------------------------
OUTDIR = Path(".")          # run inside ccl_data/
ROOT = "FAKE"               # file prefix
nzbins = 2                  # source bins
nwbins = 2                  # lens bins
ntheta = 15                 # theta bins
theta_min_arcmin = 2.5
theta_max_arcmin = 250.0

# Diagonal covariance: sigma per data point (same for all)
sigma = 1e-5

# Which data types to include
data_types = ["xip", "xim", "gammat", "wtheta"]
used_data_types = data_types[:]  # keep same

# -----------------------------
# Helpers
# -----------------------------
def write_measurements(tp: str, pairs: list[tuple[int, int]], thetas: np.ndarray, out: Path) -> int:
    """
    Write a DES-like measurements file:
    # BIN1 BIN2 ANGBIN VALUE
    bins are 1-based, ANGBIN is 1..ntheta
    Returns number of rows written.
    """
    rows = []
    for (b1, b2) in pairs:
        for i in range(ntheta):
            angbin = i + 1
            rows.append([b1, b2, angbin, 0.0])

    arr = np.array(rows, dtype=float)
    header = "BIN1 BIN2 ANGBIN VALUE"
    fmt = ["%d", "%d", "%d", "%.8e"]
    np.savetxt(out, arr, header=header, comments="# ", fmt=fmt)
    return arr.shape[0]


def write_selection(out: Path):
    """
    data_selection format:
    #  type bin1 bin2 theta_min theta_max
    bins are 1-based in the file.
    """
    lines = ["#  type bin1 bin2 theta_min theta_max"]
    # xip/xim: source-source pairs (we'll include (1,1), (1,2), (2,2))
    src_pairs = [(1, 1), (1, 2), (2, 2)]
    # gammat: lens-source (include all 2x2)
    gt_pairs = [(1, 1), (1, 2), (2, 1), (2, 2)]
    # wtheta: lens-lens (only auto: (1,1), (2,2))
    w_pairs = [(1, 1), (2, 2)]

    for (b1, b2) in src_pairs:
        lines.append(f"xip {b1} {b2} {theta_min_arcmin:.6f} {theta_max_arcmin:.6f}")
        lines.append(f"xim {b1} {b2} {theta_min_arcmin:.6f} {theta_max_arcmin:.6f}")
    for (b1, b2) in gt_pairs:
        lines.append(f"gammat {b1} {b2} {theta_min_arcmin:.6f} {theta_max_arcmin:.6f}")
    for (b1, b2) in w_pairs:
        lines.append(f"wtheta {b1} {b2} {theta_min_arcmin:.6f} {theta_max_arcmin:.6f}")

    out.write_text("\n".join(lines) + "\n")


def write_toy_nz(out: Path, nbins: int, zmax: float = 2.0, nz: int = 300, z0: float = 0.5):
    """
    Writes a simple n(z) table similar to DES files:
    columns: index  z  <two filler cols>  then nbins columns starting at col 3
    Your likelihood reads: zmid = col 1, and uses cols b+3
    """
    z = np.linspace(0.0, zmax, nz)
    # Simple gamma-like shape
    base = (z**2) * np.exp(-(z / z0) ** 1.5)
    base /= np.trapz(base, z)

    cols = []
    cols.append(np.arange(nz, dtype=int))  # col 0
    cols.append(z)                         # col 1 (Z_MID)
    cols.append(np.zeros_like(z))          # col 2 dummy
    # bin curves: slightly shifted/scaled
    for b in range(nbins):
        zb = z0 * (1.0 + 0.15 * (b - (nbins - 1) / 2))
        nb = (z**2) * np.exp(-(z / zb) ** 1.5)
        nb /= np.trapz(nb, z)
        cols.append(nb)

    arr = np.column_stack(cols)
    header = "I Z_MID DUMMY " + " ".join([f"BIN{b+1}" for b in range(nbins)])
    np.savetxt(out, arr, header=header, comments="# ", fmt="%.8e")


# -----------------------------
# Main generation
# -----------------------------
OUTDIR.mkdir(parents=True, exist_ok=True)

# Theta bins (arcmin)
thetas = np.geomspace(theta_min_arcmin, theta_max_arcmin, ntheta)
np.savetxt(OUTDIR / f"{ROOT}_theta_bins.dat", thetas, header="theta_arcmin", comments="# ", fmt="%.8e")

# Measurement pair definitions (1-based bins in files)
src_pairs = [(1, 1), (1, 2), (2, 2)]
gt_pairs = [(1, 1), (1, 2), (2, 1), (2, 2)]
w_pairs  = [(1, 1), (2, 2)]

nrows = {}
nrows["xip"]    = write_measurements("xip", src_pairs, thetas, OUTDIR / f"{ROOT}_xip.dat")
nrows["xim"]    = write_measurements("xim", src_pairs, thetas, OUTDIR / f"{ROOT}_xim.dat")
nrows["gammat"] = write_measurements("gammat", gt_pairs, thetas, OUTDIR / f"{ROOT}_gammat.dat")
nrows["wtheta"] = write_measurements("wtheta", w_pairs, thetas, OUTDIR / f"{ROOT}_wtheta.dat")

# Total data vector length (matches cov)
ndata = sum(nrows[tp] for tp in data_types)

# Diagonal covariance
cov = (sigma**2) * np.eye(ndata)
np.savetxt(OUTDIR / f"{ROOT}_cov.dat", cov, fmt="%.8e")

# Selection (all theta range)
write_selection(OUTDIR / f"{ROOT}_selection.dat")

# Toy nz files (read by likelihood)
write_toy_nz(OUTDIR / f"{ROOT}_nz_source.dat", nbins=nzbins, zmax=2.0, nz=400, z0=0.6)
write_toy_nz(OUTDIR / f"{ROOT}_nz_lens.dat",   nbins=nwbins, zmax=2.0, nz=400, z0=0.4)

# Dataset file
dataset_lines = [
    "measurements_format = DES",
    "kmax = 10",
    "intrinsic_alignment_model = DES1YR",
    f"data_types = {' '.join(data_types)}",
    f"used_data_types = {' '.join(used_data_types)}",
    f"num_z_bins = {nzbins}",
    f"num_gal_bins = {nwbins}",
    f"num_theta_bins = {ntheta}",
    f"theta_bins_file = {ROOT}_theta_bins.dat",
    f"cov_file = {ROOT}_cov.dat",
]
for tp in data_types:
    dataset_lines.append(f"measurements[{tp}] = {ROOT}_{tp}.dat")
dataset_lines += [
    f"nz_file = {ROOT}_nz_source.dat",
    f"nz_gal_file = {ROOT}_nz_lens.dat",
    f"data_selection = {ROOT}_selection.dat",
    "nuisance_params = DES.paramnames",
]
(OUTDIR / f"{ROOT}.dataset").write_text("\n".join(dataset_lines) + "\n")

print("Wrote:")
print(" ", OUTDIR / f"{ROOT}.dataset")
print("Data vector length =", ndata, "with sigma =", sigma)

