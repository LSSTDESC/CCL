r"""Benchmark: CCL van Daalen 2019 model vs Marcel's fbar tables.

Each file in benchmarks/data/baryons_vd20_from_marcel has the form

    VD20_rXXX_kY.txt

where:
    - rXXX encodes the mass definition:
        r500 -> 500c
        r200 -> 200c
    - kY encodes a fixed wavenumber in h/Mpc, e.g. k0.1, k0.5, k1.0

The file contents are two columns:

    col0:  fbar = \bar{f}_{bar,R_{\rm crit}}(10^{14} M_\odot) / (\Omega_b / \Omega_m)
    col1:  DeltaP_over_P = (P_{\rm vD} - P_{\rm DMO}) / P_{\rm DMO}  (negative for bary suppression)

i.e. the *fractional change* in the power spectrum at the given k and mass
definition, as a function of the baryon fraction parameter fbar. DMO is dark matter only (no baryons).

CCL's BaryonsvanDaalen19 implements the analytic model of van Daalen+2019,
which provides the ratio

    ratio(k, fbar) = P_{\rm vD} / P_{\rm DMO}

so the corresponding fractional change is

    DeltaP_over_P = ratio - 1.

This benchmark checks that CCL reproduces Marcel's tables for all six
(k, mass_def) combinations.

Note
----
The VD20_rXXX_kY.txt tables were obtained by (Niko Šarčević
directly from Marcel van Daalen on December 1st 2025 via personal
correspondence, and are used here as an external validation target for
the BaryonsvanDaalen19 implementation.
"""

from pathlib import Path

import pytest

import numpy as np

import pyccl as ccl

DATA_DIR = Path(__file__).parent / "data" / "baryons_vd20_from_marcel"

# Relative tol for DeltaP/P
VD19_FBAR_RTOL = 5e-3  # 0.5%


def _make_cosmo() -> ccl.Cosmology:
    """Creates a pyccl Cosmology object used for the test.

    The van Daalen+19 model is cosmology-independent except for the
    conversion between k [1/Mpc] and k [h/Mpc], so a Vanilla LCDM
    cosmology is sufficient here.
    """
    return ccl.CosmologyVanillaLCDM()


def _parse_meta(stem: str) -> tuple[str, float]:
    """Parses mass_def and k[h/Mpc] from filename stem.

    Example stems:
        "VD20_r500_k0.1"
        "VD20_r200_k1.0"
    """
    parts = stem.split("_")
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename stem format: {stem!r}")

    r_part = parts[1]  # e.g. "r500" or "r200"
    k_part = parts[2]  # e.g. "k0.1"

    if r_part == "r500":
        mass_def = "500c"
    elif r_part == "r200":
        mass_def = "200c"
    else:
        raise ValueError(f"Cannot infer mass_def from stem {stem!r}")

    if not k_part.startswith("k"):
        raise ValueError(f"Cannot parse k from stem {stem!r}")
    k_hmpc = float(k_part[1:])  # strip leading 'k'

    return mass_def, k_hmpc


@pytest.mark.parametrize("path", sorted(DATA_DIR.glob("VD20_r*.txt")))
def test_vd19_against_marcel_fbar_tables(path: Path) -> None:
    """Compares CCL van Daalen19 ΔP/P to Marcel's tables at fixed k and mass_def."""
    cosmo = _make_cosmo()
    mass_def, k_hmpc = _parse_meta(path.stem)

    # Load fbar and fractional change: DeltaP/P_DMO
    fbar_tab, delta_tab = np.loadtxt(path, unpack=True)

    # Make sure fbar is sorted (just in case smth weird in the files)
    order = np.argsort(fbar_tab)
    fbar_tab = fbar_tab[order]
    delta_tab = delta_tab[order]

    # CCL instance with correct mass definition; we'll vary fbar per point
    vd = ccl.baryons.BaryonsvanDaalen19(mass_def=mass_def)

    # Now we convert k from h/Mpc to 1/Mpc for CCL; internal formula uses k/h
    k = k_hmpc * cosmo["h"]
    a = 1.0  # z = 0

    # Evaluate CCL model at each fbar in the table
    delta_ccl = np.empty_like(delta_tab)
    for i, fbar in enumerate(fbar_tab):
        vd.update_parameters(fbar=fbar)
        ratio = vd.boost_factor(cosmo, k, a)  # P_vD / P_DMO
        delta_ccl[i] = ratio - 1.0  # (P_vD - P_DMO) / P_DMO

    # Compare fractional changes
    np.testing.assert_allclose(
        delta_ccl,
        delta_tab,
        rtol=VD19_FBAR_RTOL,
        atol=0.0,
    )
