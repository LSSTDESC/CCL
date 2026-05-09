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


@pytest.mark.parametrize("k", [
    1,
    1.0,
    [0.2, 0.5, 1.0],
    np.array([0.2, 0.5, 1.0]),
])
def test_spk_smoke(k: Any) -> None:
    """Smoke-test SP(k) boost evaluation for scalar/array ``k`` inputs."""
    pytest.importorskip("pyspk")
    bar = _power_law_model()
    a = 0.8
    fka = bar.boost_factor(COSMO, k, a)
    assert np.all(np.isfinite(fka))
    assert np.shape(fka) == np.shape(k)


@pytest.mark.parametrize("relation_kind, relation_params", [
    ("power_law", {"fb_a": 0.4, "fb_pow": 0.3, "fb_pivot": 10**13.5}),
    ("binned", {"M_halo": [1.0e13, 3.0e13, 1.0e14],
                "fb": [0.12, 0.15, 0.18], "extrapolate": False}),
    ("cosmo_power_law", {"alpha": 4.16, "beta": 1.2, "gamma": 0.39}),
    ("double_power_law", {"epsilon": 0.3, "alpha": 1.1, "beta": 0.2,
                          "gamma": 0.5, "m_pivot": 10**13.5}),
])
def test_spk_matches_pyspk(relation_kind, relation_params):
    """Check numerical agreement with direct ``pyspk`` evaluator output."""
    pyspk = pytest.importorskip("pyspk")
    bar = ccl.BaryonsSPK(
        SO=(500
            if relation_kind in ("cosmo_power_law", "double_power_law")
            else 200),
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
        SO=bar.SO, relation_kind=relation_kind, k_array=k_hmpc)
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
    bar = _power_law_model(out_of_bounds_policy="unity")
    k_arr = np.geomspace(1E-2, 1, 16)
    fka = bar.boost_factor(COSMO, k_arr, 0.5)
    pk_nobar = ccl.nonlin_matter_power(COSMO, k_arr, 0.5)
    pkb = bar.include_baryonic_effects(COSMO, COSMO.get_nonlin_power())
    pkb_eval = cast(Callable[[np.ndarray, float], np.ndarray], pkb)
    pk_wbar = pkb_eval(k_arr, 0.5)
    assert np.all(np.fabs(pk_wbar / (pk_nobar * fka) - 1) < 1E-5)


def test_spk_out_of_bounds_policies() -> None:
    """Verify high-k policy behavior for direct boost queries."""
    pytest.importorskip("pyspk")
    k = np.array([0.2, 0.8])
    a = 0.9

    with pytest.raises(ValueError):
        model = _power_law_model(
            k_max_mpc=0.5, out_of_bounds_policy="error")
        model.boost_factor(COSMO, k, a)

    fk_unity = _power_law_model(
        k_max_mpc=0.5, out_of_bounds_policy="unity").boost_factor(COSMO, k, a)
    assert fk_unity[-1] == 1.0

    fk_nan = _power_law_model(
        k_max_mpc=0.5, out_of_bounds_policy="nan").boost_factor(COSMO, k, a)
    assert np.isnan(fk_nan[-1])


def test_spk_out_of_bounds_policies_include_baryons() -> None:
    """Verify high-k policy behavior in ``include_baryonic_effects`` path."""
    pytest.importorskip("pyspk")
    pk_nobar = COSMO.get_nonlin_power()
    k_hi = np.array([1.0])
    a = 0.8

    bar_unity = _power_law_model(k_max_mpc=0.5, out_of_bounds_policy="unity")
    pk_unity = bar_unity.include_baryonic_effects(COSMO, pk_nobar)
    pk_unity_eval = cast(Callable[[np.ndarray, float], np.ndarray], pk_unity)
    pk_nobar_eval = cast(Callable[[np.ndarray, float], np.ndarray], pk_nobar)
    ratio_unity = pk_unity_eval(k_hi, a) / pk_nobar_eval(k_hi, a)
    assert np.allclose(ratio_unity, 1.0, atol=0, rtol=1e-12)

    bar_nan = _power_law_model(
        k_max_mpc=0.5, out_of_bounds_policy="nan")
    pk_nan = bar_nan.include_baryonic_effects(COSMO, pk_nobar)
    pk_nan_eval = cast(Callable[[np.ndarray, float], np.ndarray], pk_nan)
    ratio_nan = pk_nan_eval(k_hi, a) / pk_nobar_eval(k_hi, a)
    assert np.isnan(ratio_nan[0])

    bar_error = _power_law_model(
        k_min_mpc=1e-4, k_max_mpc=1e-3, out_of_bounds_policy="error")
    with pytest.raises(ValueError):
        bar_error.include_baryonic_effects(COSMO, pk_nobar)


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
    bar = _power_law_model(out_of_bounds_policy="unity")
    cosmo_nb = ccl.CosmologyVanillaLCDM(
        transfer_function="bbks", baryonic_effects=None)
    pk_nb = cosmo_nb.get_nonlin_power()
    pk_wb = bar.include_baryonic_effects(cosmo_nb, pk_nb)

    cosmo_wb = ccl.CosmologyVanillaLCDM(
        transfer_function="bbks", baryonic_effects=bar)
    pk_wb_cosmo = cosmo_wb.get_nonlin_power()

    ks = np.geomspace(1E-2, 2, 128)
    pk_wb_eval = cast(Callable[[np.ndarray, float], np.ndarray], pk_wb)
    pk_wb_cosmo_eval = cast(
        Callable[[np.ndarray, float], np.ndarray], pk_wb_cosmo)
    assert np.allclose(pk_wb_eval(ks, 1.0), pk_wb_cosmo_eval(ks, 1.0),
                       atol=0, rtol=1E-6)


def test_spk_missing_dependency_error() -> None:
    """Raise a clear error when optional dependency ``pyspk`` is missing."""
    with mock.patch.dict(sys.modules, {"pyspk": None}):
        with pytest.raises(ModuleNotFoundError, match="pyspk>=2.0.0"):
            _power_law_model()


def test_spk_warnings_are_deduplicated_per_instance() -> None:
    """Repeated calls should not re-emit identical forwarded warnings."""
    pytest.importorskip("pyspk")
    # High internal h/Mpc coverage (set via k_max_mpc) typically emits warning.
    bar = _power_law_model(k_max_mpc=12.0 * COSMO["h"], n_k=64)
    k = np.geomspace(1e-2, 1.0, 32)

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
        str(w.message) for w in second if issubclass(w.category, ccl.CCLWarning)
    }
    assert first_msgs.isdisjoint(second_msgs)
