"""Unit tests for ``pyccl.baryons.mead20_hmcode.mead20_utils``."""

from __future__ import annotations

import numpy as np
import pytest

from pyccl.baryons.mead20_hmcode import mead20_utils as u


class _FakeCosmology:
    """Test stub for Cosmology with only the fields mead20_utils touches."""
    def __init__(self, d: dict, *, baryons=None):
        self._d = d
        self.baryons = baryons

    def to_dict(self) -> dict:
        return dict(self._d)


def test_cosmo_signature_equal_for_same_cosmo() -> None:
    """Test that identical cosmology dicts produce identical signatures."""
    c1 = _FakeCosmology({"Omega_c": 0.25, "h": 0.7})
    c2 = _FakeCosmology({"Omega_c": 0.25, "h": 0.7})
    assert u._cosmo_signature(c1) == u._cosmo_signature(c2)


def test_cosmo_signature_diff_for_diff_cosmo() -> None:
    """Test that different cosmology dicts produce different signatures."""
    c1 = _FakeCosmology({"Omega_c": 0.25, "h": 0.7})
    c2 = _FakeCosmology({"Omega_c": 0.26, "h": 0.7})
    assert u._cosmo_signature(c1) != u._cosmo_signature(c2)


def test_raise_if_already_hmcode_feedback_noop_if_not_camb() -> None:
    """Test that non-CAMB power spectrum settings do not trigger a raise."""
    cosmo = _FakeCosmology({"matter_power_spectrum": "halofit"})
    u._raise_if_already_hmcode_feedback(cosmo)  # no raise


def test_raise_if_already_hmcode_feedback_raises_if_fback_in_halofit_version(
) -> None:
    """Test that CAMB halofit_version containing feedback raises ValueError."""
    cosmo = _FakeCosmology(
        {
            "matter_power_spectrum": "camb",
            "extra_parameters": {"camb":
                                     {"halofit_version": "mead2020_feedback"}
                                 },
        }
    )
    with pytest.raises(ValueError):
        u._raise_if_already_hmcode_feedback(cosmo)


def test_raise_if_baryons_already_attached() -> None:
    """Test that an attached baryon model triggers a ValueError."""
    cosmo = _FakeCosmology({"Omega_c": 0.25}, baryons=object())
    with pytest.raises(ValueError):
        u._raise_if_baryons_already_attached(cosmo)


def test_build_internal_camb_kwargs_forces_camb_pipeline_and_disables_baryons(

) -> None:
    """Test that internal kwargs force CAMB and disable baryonic_effects."""
    cosmo = _FakeCosmology(
        {
            "transfer_function": "eisenstein_hu",
            "matter_power_spectrum": "halofit",
            "baryonic_effects": "something",
            "extra_parameters": {},
        }
    )

    kw = u._build_internal_camb_kwargs(
        cosmo,
        halofit_version="mead2020",
        include_feedback=False,
        kmax=10.0,
        HMCode_logT_AGN=7.8,
        HMCode_A_baryon=None,
        HMCode_eta_baryon=None,
        lmax=None,
        dark_energy_model=None,
        camb_overrides=None,
    )

    assert kw["transfer_function"] == "boltzmann_camb"
    assert kw["matter_power_spectrum"] == "camb"
    assert kw["baryonic_effects"] is None
    assert kw["extra_parameters"]["camb"]["halofit_version"] == "mead2020"
    # include_feedback=False should ensure HMCode_logT_AGN not present
    assert "HMCode_logT_AGN" not in kw["extra_parameters"]["camb"]


def test_build_internal_camb_kwargs_kmax_never_decreases_user_value() -> None:
    """Test that internal kmax never reduces an existing CAMB kmax setting."""
    cosmo = _FakeCosmology(
        {
            "extra_parameters": {"camb": {"kmax": 50.0}},
        }
    )
    kw = u._build_internal_camb_kwargs(
        cosmo,
        halofit_version="mead2020",
        include_feedback=False,
        kmax=20.0,
        HMCode_logT_AGN=7.8,
        HMCode_A_baryon=None,
        HMCode_eta_baryon=None,
        lmax=None,
        dark_energy_model=None,
        camb_overrides=None,
    )
    assert kw["extra_parameters"]["camb"]["kmax"] == 50.0


def test_build_internal_camb_kwargs_does_not_stomp_optional_user_params(

) -> None:
    """Test that user CAMB options are preserved when already set."""
    cosmo = _FakeCosmology(
        {
            "extra_parameters": {
                "camb": {
                    "lmax": 123,
                    "dark_energy_model": "fluid",
                    "HMCode_A_baryon": 3.0,
                    "HMCode_eta_baryon": 0.6,
                }
            }
        }
    )
    kw = u._build_internal_camb_kwargs(
        cosmo,
        halofit_version="mead2020",
        include_feedback=True,
        kmax=20.0,
        HMCode_logT_AGN=7.8,
        HMCode_A_baryon=9.0,
        HMCode_eta_baryon=9.0,
        lmax=999,
        dark_energy_model="ppf",
        camb_overrides=None,
    )
    camb = kw["extra_parameters"]["camb"]
    assert camb["lmax"] == 123
    assert camb["dark_energy_model"] == "fluid"
    assert camb["HMCode_A_baryon"] == 3.0
    assert camb["HMCode_eta_baryon"] == 0.6
    # include_feedback=True must force HMCode_logT_AGN
    assert camb["HMCode_logT_AGN"] == 7.8


def test_build_internal_camb_kwargs_overrides_win() -> None:
    """Test that camb_overrides entries take precedence over derived values."""
    cosmo = _FakeCosmology({"extra_parameters": {"camb": {"kmax": 5.0}}})
    kw = u._build_internal_camb_kwargs(
        cosmo,
        halofit_version="mead2020",
        include_feedback=True,
        kmax=20.0,
        HMCode_logT_AGN=7.8,
        HMCode_A_baryon=None,
        HMCode_eta_baryon=None,
        lmax=None,
        dark_energy_model=None,
        camb_overrides={"kmax": 999.0, "new_param": 1},
    )
    camb = kw["extra_parameters"]["camb"]
    assert camb["kmax"] == 999.0
    assert camb["new_param"] == 1


def test_ensure_log_pk_arr_converts_absurd_values() -> None:
    """Test that absurdly large values are treated as linear and logged."""
    pk_lin = np.array([1.0, 10.0, 1e300], dtype=float)
    out = u._ensure_log_pk_arr(pk_lin, absurd_log_threshold=200.0)
    assert np.all(np.isfinite(out))
    assert np.allclose(out[:-1], np.log(pk_lin[:-1]))


def test_apply_boost_linear() -> None:
    """Test that boosts multiply directly when the input spectrum is linear."""
    pk = np.array([2.0, 3.0], dtype=float)
    boost = np.array([10.0, 0.5], dtype=float)
    out = u._apply_boost_to_pk_arr(pk, boost, is_log=False)
    assert np.allclose(out, pk * boost)


def test_apply_boost_log() -> None:
    """Test that boosts add in log-space when the input spectrum is log(P)."""
    pk_log = np.log(np.array([2.0, 3.0], dtype=float))
    boost = np.array([10.0, 0.5], dtype=float)
    out = u._apply_boost_to_pk_arr(pk_log, boost, is_log=True)
    assert np.allclose(out, pk_log + np.log(boost))


def test_to_2d_na_nk_accepts_transpose() -> None:
    """Test that (nk, na) inputs are transposed to (na, nk)."""
    na, nk = 2, 3
    p = np.arange(na * nk, dtype=float).reshape(nk, na)
    out = u._to_2d_na_nk(p, na, nk, name="P")
    assert out.shape == (na, nk)
    assert np.allclose(out, p.T)


def test_to_2d_na_nk_raises_on_bad_shape() -> None:
    """Test that incompatible shapes raise a RuntimeError."""
    with pytest.raises(RuntimeError):
        u._to_2d_na_nk(np.zeros((2, 2)), na=2, nk=3, name="P")


def test_safe_pos_ratio_handles_tiny_den_and_requires_positive() -> None:
    """Test that ratios handle tiny denominators and enforce positivity."""
    num = np.array([1.0, 2.0], dtype=float)
    den = np.array([0.0, 1e-400], dtype=float)  # both tiny
    out = u._safe_pos_ratio(num, den, name="T")
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)

    with pytest.raises(FloatingPointError):
        u._safe_pos_ratio(np.array([-1.0]), np.array([1.0]), name="T")


def test_restore_input_ndim() -> None:
    """Test that output shapes match scalar/array conventions for a and k."""
    out2d = np.arange(6, dtype=float).reshape(2, 3)

    # a scalar, k array -> (nk,)
    out = u._restore_input_ndim(out2d, a=1.0, k=np.array([1.0, 2.0, 3.0]))
    assert out.shape == (3,)

    # a array, k scalar -> (na,)
    out = u._restore_input_ndim(out2d, a=np.array([0.5, 1.0]), k=2.0)
    assert out.shape == (2,)

    # both scalars -> float
    out = u._restore_input_ndim(out2d, a=1.0, k=2.0)
    assert isinstance(out, float)


def test_raise_if_already_hmcode_feedback_raises_case_insensitive() -> None:
    """Test that feedback detection in halofit_version is case-insensitive."""
    cosmo = _FakeCosmology(
        {
            "matter_power_spectrum": "camb",
            "extra_parameters": {"camb":
                                     {"halofit_version": "MeAd2020_FeedBack"}},
        }
    )
    with pytest.raises(ValueError):
        u._raise_if_already_hmcode_feedback(cosmo)


def test_raise_if_already_hmcode_feedback_noop_for_camb_without_feedback(

) -> None:
    """Test that CAMB without 'feedback' in halofit_version does not raise."""
    cosmo = _FakeCosmology(
        {
            "matter_power_spectrum": "camb",
            "extra_parameters": {"camb": {"halofit_version": "mead2020"}},
        }
    )
    u._raise_if_already_hmcode_feedback(cosmo)  # no raise


def test_raise_if_already_hmcode_feedback_message_includes_version(

) -> None:
    """Test that raised error message includes the halofit_version value."""
    cosmo = _FakeCosmology(
        {
            "matter_power_spectrum": "camb",
            "extra_parameters": {"camb":
                                     {"halofit_version": "mead2020_feedback"}},
        }
    )
    with pytest.raises(
            ValueError,
            match=r"halofit_version=.*mead2020_feedback"
    ):
        u._raise_if_already_hmcode_feedback(cosmo)
