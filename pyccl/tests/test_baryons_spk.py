import sys
from unittest import mock

import numpy as np
import pyccl as ccl
import pytest

from .test_cclobject import check_eq_repr_hash


COSMO = ccl.CosmologyVanillaLCDM(transfer_function="bbks")


def _power_law_model(**kwargs):
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
def test_spk_smoke(k):
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
    pyspk = pytest.importorskip("pyspk")
    bar = ccl.BaryonsSPK(
        SO=500 if relation_kind in ("cosmo_power_law", "double_power_law") else 200,
        relation_kind=relation_kind,
        **relation_params,
    )
    a = 0.8
    z = 1 / a - 1
    k = np.geomspace(1e-2, 2.0, 128)
    ccl_fk = bar.boost_factor(COSMO, k, a)

    k_hmpc = k / COSMO["h"]
    evaluator = pyspk.build_sup_model_evaluator(
        SO=bar.SO, relation_kind=relation_kind, k_array=k_hmpc)
    direct_kwargs = dict(relation_params)
    if relation_kind in ("cosmo_power_law", "double_power_law"):
        direct_kwargs["efunc"] = lambda z_: COSMO.h_over_h0(1.0 / (1.0 + z_))
    _, pyspk_fk = evaluator(z=z, **direct_kwargs)

    assert np.allclose(ccl_fk, pyspk_fk, atol=1e-3, rtol=0)


def test_spk_correct_smoke():
    pytest.importorskip("pyspk")
    bar = _power_law_model()
    k_arr = np.geomspace(1E-2, 1, 16)
    fka = bar.boost_factor(COSMO, k_arr, 0.5)
    pk_nobar = ccl.nonlin_matter_power(COSMO, k_arr, 0.5)
    pkb = bar.include_baryonic_effects(COSMO, COSMO.get_nonlin_power())
    pk_wbar = pkb(k_arr, 0.5)
    assert np.all(np.fabs(pk_wbar / (pk_nobar * fka) - 1) < 1E-5)


def test_spk_out_of_bounds_policies():
    pytest.importorskip("pyspk")
    k = np.array([0.2, 0.8]) * COSMO["h"]
    a = 0.9

    with pytest.raises(ValueError):
        _power_law_model(k_max_hmpc=0.5, out_of_bounds_policy="error").boost_factor(
            COSMO, k, a)

    fk_unity = _power_law_model(
        k_max_hmpc=0.5, out_of_bounds_policy="unity").boost_factor(COSMO, k, a)
    assert fk_unity[-1] == 1.0

    fk_nan = _power_law_model(
        k_max_hmpc=0.5, out_of_bounds_policy="nan").boost_factor(COSMO, k, a)
    assert np.isnan(fk_nan[-1])


def test_spk_update_params_and_eq():
    pytest.importorskip("pyspk")
    bar1 = _power_law_model()
    bar2 = _power_law_model()
    assert check_eq_repr_hash(bar1, bar2)

    bar2.update_parameters(fb_a=0.5)
    assert check_eq_repr_hash(bar1, bar2, equal=False)

    bar2.update_parameters(fb_a=0.4)
    assert check_eq_repr_hash(bar1, bar2)


def test_spk_baryons_in_cosmology():
    pytest.importorskip("pyspk")
    bar = _power_law_model()
    cosmo_nb = ccl.CosmologyVanillaLCDM(
        transfer_function="bbks", baryonic_effects=None)
    pk_nb = cosmo_nb.get_nonlin_power()
    pk_wb = bar.include_baryonic_effects(cosmo_nb, pk_nb)

    cosmo_wb = ccl.CosmologyVanillaLCDM(
        transfer_function="bbks", baryonic_effects=bar)
    pk_wb_cosmo = cosmo_wb.get_nonlin_power()

    ks = np.geomspace(1E-2, 2, 128)
    assert np.allclose(pk_wb(ks, 1.0), pk_wb_cosmo(ks, 1.0), atol=0, rtol=1E-6)


def test_spk_missing_dependency_error():
    with mock.patch.dict(sys.modules, {"pyspk": None}):
        with pytest.raises(ModuleNotFoundError, match="pyspk>=2.0.0"):
            _power_law_model()
