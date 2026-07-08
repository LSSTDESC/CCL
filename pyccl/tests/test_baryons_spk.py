"""Unit tests for the SP(k) baryonic suppression wrapper."""

import sys
import warnings as warnings_builtin
from typing import Any, Callable, cast
from unittest import mock

import numpy as np
import pyccl as ccl
import pytest  # pyright: ignore[reportMissingImports]

from .test_cclobject import check_eq_repr_hash


COSMO = ccl.CosmologyVanillaLCDM(transfer_function="bbks")


def _power_law_model(**kwargs: Any) -> ccl.BaryonsSPK:
    """Build a default SP(k) power-law model for tests.

    Args:
        **kwargs: Parameter overrides for ``BaryonsSPK``.

    Returns:
        Configured ``BaryonsSPK`` instance.
    """
    return ccl.BaryonsSPK(
        SO=200,
        relation_kind="power_law",
        fb_a=0.4,
        fb_pow=0.3,
        fb_pivot=10**13.5,
        **kwargs,
    )


@pytest.mark.parametrize(
    "k",
    [
        1,
        1.0,
        [0.2, 0.5, 1.0],
        np.array([0.2, 0.5, 1.0]),
    ],
)
def test_spk_smoke(k: Any) -> None:
    """Smoke-test SP(k) boost evaluation for scalar/array ``k`` inputs."""
    pytest.importorskip("pyspk")
    bar = _power_law_model()
    a = 0.8
    fka = bar.boost_factor(COSMO, k, a)
    assert np.all(np.isfinite(fka))
    assert np.shape(fka) == np.shape(k)


@pytest.mark.parametrize(
    "relation_kind, relation_params",
    [
        ("power_law", {"fb_a": 0.4, "fb_pow": 0.3, "fb_pivot": 10**13.5}),
        (
            "binned",
            {
                "M_halo": [1.0e13, 3.0e13, 1.0e14],
                "fb": [0.12, 0.15, 0.18],
                "extrapolate": False,
            },
        ),
        ("cosmo_power_law", {"alpha": 4.16, "beta": 1.2, "gamma": 0.39}),
        (
            "double_power_law",
            {
                "epsilon": 0.3,
                "alpha": 1.1,
                "beta": 0.2,
                "gamma": 0.5,
                "m_pivot": 10**13.5,
            },
        ),
    ],
)
def test_spk_matches_pyspk(relation_kind, relation_params):
    """Check numerical agreement with direct ``pyspk`` evaluator output."""
    pyspk = pytest.importorskip("pyspk")
    bar = ccl.BaryonsSPK(
        SO=(
            500
            if relation_kind in ("cosmo_power_law", "double_power_law")
            else 200
        ),
        relation_kind=relation_kind,
        **relation_params,
    )
    a = 0.8
    z = 1 / a - 1
    k = np.geomspace(1e-2, 2.0, 128)
    ccl_fk = bar.boost_factor(COSMO, k, a)

    h = cast(float, COSMO["h"])
    k_hmpc = k / h
    evaluator = pyspk.build_sup_model_evaluator(
        SO=bar.SO, relation_kind=relation_kind, k_array=k_hmpc
    )
    direct_kwargs = dict(relation_params)
    if relation_kind in ("cosmo_power_law", "double_power_law"):
        h_over_h0 = cast(Callable[[float], float], getattr(COSMO, "h_over_h0"))
        direct_kwargs["efunc"] = lambda z_: h_over_h0(1.0 / (1.0 + z_))
    _, pyspk_fk = evaluator(z=z, **direct_kwargs)

    assert np.allclose(ccl_fk, pyspk_fk, atol=1e-3, rtol=0)


def test_spk_k_unit_conversion_is_transparent() -> None:
    """CCL-facing k in Mpc^-1 should internally map to pyspk's h/Mpc."""
    pyspk = pytest.importorskip("pyspk")
    bar = _power_law_model()

    a = 0.8
    z = 1.0 / a - 1.0
    k_hmpc = np.geomspace(1e-2, 2.0, 128)
    k_mpc = k_hmpc * cast(float, COSMO["h"])

    fk_ccl = bar.boost_factor(COSMO, k_mpc, a)
    evaluator = pyspk.build_sup_model_evaluator(
        SO=bar.SO,
        relation_kind=bar.relation_kind,
        k_array=k_hmpc,
    )
    _, fk_pyspk = evaluator(z=z, **dict(bar.relation_params))

    assert np.allclose(fk_ccl, fk_pyspk, atol=1e-3, rtol=0)


def test_spk_correct_smoke() -> None:
    """Validate consistency between boost_factor and
    include_baryonic_effects.
    """
    pytest.importorskip("pyspk")
    bar = _power_law_model(k_out_of_range="unity")
    k_arr = np.geomspace(1e-2, 1, 16)
    fka = bar.boost_factor(COSMO, k_arr, 0.5)
    pk_nobar = ccl.nonlin_matter_power(COSMO, k_arr, 0.5)
    pkb = bar.include_baryonic_effects(COSMO, COSMO.get_nonlin_power())
    pkb_eval = cast(Callable[[np.ndarray, float], np.ndarray], pkb)
    pk_wbar = pkb_eval(k_arr, 0.5)
    assert np.all(np.fabs(pk_wbar / (pk_nobar * fka) - 1) < 1e-5)


def test_spk_high_k_raises() -> None:
    """Requests beyond pyspk's calibrated k range should raise."""
    pytest.importorskip("pyspk")
    k = np.array([0.2, 20.0])
    a = 0.9

    with pytest.raises(Exception):
        _power_law_model().boost_factor(COSMO, k, a)


def test_spk_include_baryons_k_out_of_range_unity() -> None:
    """Include path should apply unity suppression above calibrated k."""
    pytest.importorskip("pyspk")
    cosmo_hi = ccl.CosmologyVanillaLCDM(
        transfer_function="bbks",
        matter_power_spectrum="halofit",
    )
    bar = _power_law_model(k_out_of_range="unity")
    pk_nb = cosmo_hi.get_nonlin_power()
    pk_wb = bar.include_baryonic_effects(cosmo_hi, pk_nb)
    pk_nb_eval = cast(Callable[[np.ndarray, float], np.ndarray], pk_nb)
    pk_wb_eval = cast(Callable[[np.ndarray, float], np.ndarray], pk_wb)
    k_hi = np.array([20.0])
    a = 0.8
    ratio = pk_wb_eval(k_hi, a) / pk_nb_eval(k_hi, a)
    assert np.allclose(ratio, 1.0, atol=0, rtol=1e-12)


def test_spk_z_out_of_range_raise() -> None:
    """z-out-of-range policy should raise when configured as strict."""
    pyspk = pytest.importorskip("pyspk")
    z_hi = pyspk.constants.CALIBRATED_Z_MAX + 0.1
    a_hi = 1.0 / (1.0 + z_hi)

    with pytest.raises(
        ValueError,
        match="Requested z exceeds pyspk calibration range",
    ):
        _power_law_model(z_out_of_range="raise").boost_factor(COSMO, 0.2, a_hi)


def test_spk_z_out_of_range_nan() -> None:
    """z-out-of-range policy should return NaN when configured as nan."""
    pyspk = pytest.importorskip("pyspk")
    z_hi = pyspk.constants.CALIBRATED_Z_MAX + 0.1
    a_hi = 1.0 / (1.0 + z_hi)

    fk = _power_law_model(z_out_of_range="nan").boost_factor(COSMO, 0.2, a_hi)
    assert np.isnan(fk)


def test_spk_include_baryons_nan_k_drops_columns() -> None:
    """Include path with k_out_of_range='nan' drops non-finite columns."""
    pytest.importorskip("pyspk")
    cosmo_hi = ccl.CosmologyVanillaLCDM(
        transfer_function="bbks",
        matter_power_spectrum="halofit",
    )
    bar = _power_law_model(k_out_of_range="nan")
    pk_nb = cosmo_hi.get_nonlin_power()
    with warnings_builtin.catch_warnings(record=True) as caught:
        warnings_builtin.simplefilter("always")
        pk_wb = bar.include_baryonic_effects(cosmo_hi, pk_nb)
    msgs = [
        str(w.message)
        for w in caught
        if issubclass(w.category, ccl.CCLWarning)
    ]
    assert any("non-finite" in m for m in msgs)
    # Result should still be finite (NaN columns were dropped).
    pk_wb_eval = cast(Callable[[np.ndarray, float], np.ndarray], pk_wb)
    assert np.all(np.isfinite(pk_wb_eval(np.array([0.5]), 0.8)))


def test_spk_include_baryons_nan_z_drops_rows() -> None:
    """Include path with z_out_of_range='nan' drops non-finite rows."""
    pyspk = pytest.importorskip("pyspk")
    cosmo_hi = ccl.CosmologyVanillaLCDM(
        transfer_function="bbks",
        matter_power_spectrum="halofit",
    )
    # Use a low a (high z) that exceeds calibration.
    z_hi = pyspk.constants.CALIBRATED_Z_MAX + 0.5
    a_hi = 1.0 / (1.0 + z_hi)
    bar = _power_law_model(z_out_of_range="nan", k_out_of_range="unity")
    # Build a Pk2D that includes the high-z scale factor.
    pk_nb = cosmo_hi.get_nonlin_power()
    a_arr, lk_arr, pk_arr = pk_nb.get_spline_arrays()
    # Only check if the Pk2D actually contains a > calibrated z.
    if np.min(a_arr) > a_hi:
        pytest.skip("Pk2D a-grid does not extend beyond calibration")
    with warnings_builtin.catch_warnings(record=True) as caught:
        warnings_builtin.simplefilter("always")
        pk_wb = bar.include_baryonic_effects(cosmo_hi, pk_nb)
    msgs = [
        str(w.message)
        for w in caught
        if issubclass(w.category, ccl.CCLWarning)
    ]
    assert any("non-finite" in m for m in msgs)
    pk_wb_eval = cast(Callable[[np.ndarray, float], np.ndarray], pk_wb)
    assert np.all(np.isfinite(pk_wb_eval(np.array([0.5]), 0.8)))


def test_spk_invalid_out_of_range_string() -> None:
    """Invalid out-of-range policy strings should raise at construction."""
    pytest.importorskip("pyspk")
    with pytest.raises(ValueError, match="k_out_of_range"):
        _power_law_model(k_out_of_range="invalid")
    with pytest.raises(ValueError, match="z_out_of_range"):
        _power_law_model(z_out_of_range="bad")


def test_spk_update_params_and_eq() -> None:
    """Check parameter updates and object equality/hash semantics."""
    pytest.importorskip("pyspk")
    bar1 = _power_law_model()
    bar2 = _power_law_model()
    assert check_eq_repr_hash(bar1, bar2)

    bar2.update_parameters(fb_a=0.5)
    assert check_eq_repr_hash(bar1, bar2, equal=False)

    bar2.update_parameters(fb_a=0.4)
    assert check_eq_repr_hash(bar1, bar2)


def test_spk_baryons_in_cosmology() -> None:
    """Ensure explicit and Cosmology-integrated baryons paths agree."""
    pytest.importorskip("pyspk")
    bar = _power_law_model(k_out_of_range="unity")
    cosmo_nb = ccl.CosmologyVanillaLCDM(
        transfer_function="bbks", baryonic_effects=None
    )
    pk_nb = cosmo_nb.get_nonlin_power()
    pk_wb = bar.include_baryonic_effects(cosmo_nb, pk_nb)

    cosmo_wb = ccl.CosmologyVanillaLCDM(
        transfer_function="bbks", baryonic_effects=bar
    )
    pk_wb_cosmo = cosmo_wb.get_nonlin_power()

    ks = np.geomspace(1e-2, 2, 128)
    pk_wb_eval = cast(Callable[[np.ndarray, float], np.ndarray], pk_wb)
    pk_wb_cosmo_eval = cast(
        Callable[[np.ndarray, float], np.ndarray],
        pk_wb_cosmo,
    )
    assert np.allclose(
        pk_wb_eval(ks, 1.0), pk_wb_cosmo_eval(ks, 1.0), atol=0, rtol=1e-6
    )


def test_spk_missing_dependency_error() -> None:
    """Raise a clear error when optional dependency ``pyspk`` is missing."""
    with mock.patch.dict(sys.modules, {"pyspk": None}):
        with pytest.raises(ModuleNotFoundError, match="pyspk>=2.0.1"):
            _power_law_model()


def test_spk_warnings_are_deduplicated_per_instance() -> None:
    """Repeated calls should not re-emit identical forwarded warnings."""
    pytest.importorskip("pyspk")
    bar = _power_law_model()
    # Query above Nyquist but below calibrated k-max to trigger a warning.
    k = np.geomspace(1e-2, 9.0 * cast(float, COSMO["h"]), 64)

    with warnings_builtin.catch_warnings(record=True) as first:
        warnings_builtin.simplefilter("always")
        _ = bar.boost_factor(COSMO, k, 0.8)

    first_msgs = {
        str(w.message) for w in first if issubclass(w.category, ccl.CCLWarning)
    }
    if not first_msgs:
        pytest.skip("pyspk did not emit calibrations warnings in this setup")

    with warnings_builtin.catch_warnings(record=True) as second:
        warnings_builtin.simplefilter("always")
        _ = bar.boost_factor(COSMO, k, 0.8)

    second_msgs = {
        str(w.message)
        for w in second
        if issubclass(w.category, ccl.CCLWarning)
    }
    assert first_msgs.isdisjoint(second_msgs)
