"""Benchmark tests for SP(k) baryonic suppression in pyccl."""

from typing import Any, Callable, cast

import numpy as np
import pyccl as ccl
import pytest  # pyright: ignore[reportMissingImports]

SPK_TOLERANCE = 1e-5
SPK_MATRIX_TOLERANCE = 3e-5
SPK_A = 0.8


def _benchmark_cosmology() -> ccl.Cosmology:
    """Build a deterministic cosmology used by SP(k) benchmark checks."""
    return ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=3.046,
        mass_split="normal",
        m_nu=0.0,
        Omega_g=0,
        Omega_k=0,
        w0=-1,
        wa=0,
        transfer_function="bbks",
    )


_MATRIX_CASES = [
    (200, "power_law", {"fb_a": 0.4, "fb_pow": 0.3, "fb_pivot": 10**13.5}),
    (200, "binned", {
        "M_halo": [1.0e13, 3.0e13, 1.0e14],
        "fb": [0.12, 0.15, 0.18],
        "extrapolate": False,
    }),
    (500, "cosmo_power_law", {"alpha": 4.16, "beta": 1.2, "gamma": 0.39}),
    (500, "double_power_law", {
        "epsilon": 0.3,
        "alpha": 1.1,
        "beta": 0.2,
        "gamma": 0.5,
        "m_pivot": 10**13.5,
    }),
]


def test_spk_power_law_boost_against_reference() -> None:
    """Validate SP(k) boost against benchmark reference data."""
    pytest.importorskip("pyspk")
    cosmo = _benchmark_cosmology()

    data = np.loadtxt("./benchmarks/data/spk_power_law_fk.txt")
    h = cast(float, cosmo["h"])
    k = data[:, 0] * h
    fk_ref = data[:, 1]

    baryons = ccl.BaryonsSPK(
        SO=200,
        relation_kind="power_law",
        fb_a=0.4,
        fb_pow=0.3,
        fb_pivot=10**13.5,
        k_min_hmpc=1e-2,
        k_max_hmpc=2.0,
        n_k=k.size,
        out_of_bounds_policy="unity",
    )

    fk = baryons.boost_factor(cosmo, k, SPK_A)
    err = np.abs(fk_ref / fk - 1)
    assert np.allclose(err, 0, atol=SPK_TOLERANCE, rtol=0)


def test_spk_power_law_include_matches_boost() -> None:
    """Validate include_baryonic_effects matches boost-based expectation."""
    pytest.importorskip("pyspk")
    cosmo = _benchmark_cosmology()

    h = cast(float, cosmo["h"])
    k = np.geomspace(1e-2, 1.0, 128) * h

    baryons = ccl.BaryonsSPK(
        SO=200,
        relation_kind="power_law",
        fb_a=0.4,
        fb_pow=0.3,
        fb_pivot=10**13.5,
        k_min_hmpc=1e-2,
        k_max_hmpc=2.0,
        n_k=k.size,
        out_of_bounds_policy="unity",
    )

    pk_nobar = ccl.nonlin_matter_power(cosmo, k, SPK_A)
    fk = baryons.boost_factor(cosmo, k, SPK_A)
    pkb = baryons.include_baryonic_effects(cosmo, cosmo.get_nonlin_power())
    pkb_eval = cast(Callable[[np.ndarray, float], np.ndarray], pkb)
    pk_wbar = pkb_eval(k, SPK_A)

    err = np.abs(pk_wbar / (pk_nobar * fk) - 1)
    assert np.allclose(err, 0, atol=SPK_TOLERANCE, rtol=0)


@pytest.mark.parametrize(
    "SO, relation_kind, relation_params", _MATRIX_CASES)
def test_spk_relation_matrix_matches_pyspk(
        SO: int,
        relation_kind: str,
        relation_params: dict[str, Any]) -> None:
    """Validate all supported relation kinds and both SO values.

    This benchmark-level matrix guards against unit-conversion and
    interpolation regressions while keeping runtime short.
    """
    pyspk = pytest.importorskip("pyspk")
    cosmo = _benchmark_cosmology()

    k_hmpc = np.geomspace(1e-2, 2.0, 96)
    h = cast(float, cosmo["h"])
    k_mpc = k_hmpc * h

    baryons = ccl.BaryonsSPK(
        SO=SO,
        relation_kind=relation_kind,
        k_min_hmpc=1e-2,
        k_max_hmpc=2.0,
        n_k=160,
        out_of_bounds_policy="error",
        **relation_params,
    )
    fk_ccl = baryons.boost_factor(cosmo, k_mpc, SPK_A)

    evaluator = pyspk.build_sup_model_evaluator(
        SO=SO,
        relation_kind=relation_kind,
        k_array=k_hmpc,
    )
    kwargs = dict(relation_params)
    if relation_kind in ("cosmo_power_law", "double_power_law"):
        h_over_h0 = cast(
            Callable[[float], float], getattr(cosmo, "h_over_h0"))
        kwargs["efunc"] = lambda z: h_over_h0(1.0 / (1.0 + z))

    z = 1.0 / SPK_A - 1.0
    _, fk_pyspk = evaluator(z=z, **kwargs)

    err = np.abs(np.asarray(fk_ccl) / np.asarray(fk_pyspk) - 1.0)
    assert np.allclose(err, 0.0, atol=SPK_MATRIX_TOLERANCE, rtol=0.0)
