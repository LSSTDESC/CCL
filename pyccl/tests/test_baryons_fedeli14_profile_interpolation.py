"""Unit tests for `baryons.fedeli14_bhm.profile_interpolation`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

import pyccl.baryons.fedeli14_bhm.profile_interpolation as pi


@dataclass
class _ProfileNoUpd:
    """Profile without update_precision_fftlog."""
    def fourier(
        self,
        cosmo: Any,
        k: np.ndarray,
        mass: np.ndarray,
        a: float
    ) -> np.ndarray:
        """Fourier transform of mass profile."""
        _ = cosmo, a
        k = np.asarray(k, float)
        mass = np.asarray(mass, float)
        # expected shape: (len(mass), len(k))
        return np.ones((mass.size, k.size), dtype=float) * mass[:, None]


@dataclass
class _ProfileBadUpd:
    """Profile with non-callable update_precision_fftlog."""
    update_precision_fftlog: Any = 123

    def fourier(
        self,
        cosmo: Any,
        k: np.ndarray,
        mass: np.ndarray,
        a: float
    ) -> np.ndarray:
        """Fourier transform of mass profile."""
        _ = cosmo, a
        k = np.asarray(k, float)
        mass = np.asarray(mass, float)
        return np.ones((mass.size, k.size), dtype=float) * mass[:, None]


@dataclass
class _ProfileUpdOK:
    """Profile with callable update_precision_fftlog."""
    called: dict[str, Any]

    def update_precision_fftlog(self, **kwargs: Any) -> None:
        self.called["kwargs"] = dict(kwargs)

    def fourier(
        self,
        cosmo: Any,
        k: np.ndarray,
        mass: np.ndarray,
        a: float
    ) -> np.ndarray:
        _ = cosmo, a
        k = np.asarray(k, float)
        mass = np.asarray(mass, float)
        return np.ones((mass.size, k.size), dtype=float) * mass[:, None]


@dataclass
class _ProfileFourierBadShape:
    """Profile with a fourier method that returns the wrong shape."""
    def fourier(
        self,
        cosmo: Any,
        k: np.ndarray,
        mass: np.ndarray,
        a: float
    ) -> np.ndarray:
        _ = cosmo, a
        k = np.asarray(k, float)
        mass = np.asarray(mass, float)
        # wrong shape on purpose
        return np.ones((mass.size, k.size + 1), dtype=float)


@dataclass
class _ProfileNoFourier:
    """Profile missing fourier attribute."""
    pass


def _grid_mass_k() -> tuple[np.ndarray, np.ndarray]:
    """Return a tiny strictly increasing positive mass and k grid."""
    mass = np.array([1e13, 2e13, 5e13], dtype=float)
    k = np.array([1e-2, 1e-1, 1.0], dtype=float)
    return mass, k


def test_update_precision_fftlog_no_method_returns_profile_unchanged() -> None:
    """Tests that the profile is returned unchanged if it doesn't have an
    update_precision_fftlog method."""
    prof = _ProfileNoUpd()
    out = pi.update_precision_fftlog(prof)
    assert out is prof


def test_update_precision_fftlog_noncallable_raises() -> None:
    """Tests that an error is raised if the profile's
    update_precision_fftlog attribute is not callable."""
    prof = _ProfileBadUpd()
    with pytest.raises(TypeError, match=r"not callable"):
        _ = pi.update_precision_fftlog(prof)


def test_update_precision_fftlog_callable_is_invoked_and_returns_profile(

) -> None:
    """Tests that the profile's update_precision_fftlog method is invoked and
    that the profile is returned unchanged."""
    called: dict[str, Any] = {}
    prof = _ProfileUpdOK(called=called)

    out = pi.update_precision_fftlog(
        prof,
        padding_hi_fftlog=9.0,
        padding_lo_fftlog=8.0,
        n_per_decade=7,
        plaw_fourier=-1.5,
    )
    assert out is prof
    assert "kwargs" in called
    assert called["kwargs"]["padding_hi_fftlog"] == 9.0
    assert called["kwargs"]["padding_lo_fftlog"] == 8.0
    assert called["kwargs"]["n_per_decade"] == 7
    assert called["kwargs"]["plaw_fourier"] == -1.5


def test_interpolate_profile_u_over_m_validates_a_and_grids() -> None:
    """Tests that invalid inputs to interpolate_profile_u_over_m raise errors."""
    mass, k = _grid_mass_k()
    prof = _ProfileNoUpd()

    with pytest.raises(ValueError, match=r"a must be finite"):
        _ = pi.interpolate_profile_u_over_m(
            cosmo={}, a=float("nan"), profile=prof, mass=mass, k=k
        )

    with pytest.raises(ValueError, match=r"non-empty"):
        _ = pi.interpolate_profile_u_over_m(
            cosmo={}, a=1.0, profile=prof, mass=np.array([]), k=k
        )

    with pytest.raises(ValueError, match=r"1D grid vectors"):
        _ = pi.interpolate_profile_u_over_m(
            cosmo={}, a=1.0, profile=prof, mass=mass.reshape(3, 1), k=k
        )

    with pytest.raises(ValueError, match=r"strictly increasing"):
        _ = pi.interpolate_profile_u_over_m(
            cosmo={}, a=1.0, profile=prof,
            mass=np.array([1.0, 1.0, 2.0]), k=k
        )

    with pytest.raises(ValueError, match=r"log_axes=True"):
        _ = pi.interpolate_profile_u_over_m(
            cosmo={}, a=1.0, profile=prof,
            mass=np.array([0.0, 1.0, 2.0]), k=k, log_axes=True
        )


def test_interpolate_profile_u_over_m_requires_fourier_and_callable() -> None:
    """Tests that invalid inputs to interpolate_profile_u_over_m raise errors."""
    mass, k = _grid_mass_k()

    with pytest.raises(TypeError):
        _ = pi.interpolate_profile_u_over_m(
            cosmo={}, a=1.0, profile=_ProfileNoFourier(), mass=mass, k=k
        )

    class _P:
        fourier = 123

    with pytest.raises(TypeError, match=r"must be callable"):
        _ = pi.interpolate_profile_u_over_m(
            cosmo={},
            a=1.0,
            profile=_P(),
            mass=mass, k=k)


def test_interpolate_profile_u_over_m_rejects_wrong_fourier_shape() -> None:
    """Tests that invalid inputs to interpolate_profile_u_over_m raise errors."""
    mass, k = _grid_mass_k()
    prof = _ProfileFourierBadShape()
    with pytest.raises(ValueError, match=r"profile\.fourier must return"):
        _ = pi.interpolate_profile_u_over_m(
            cosmo={},
            a=1.0,
            profile=prof,
            mass=mass,
            k=k)


def test_interpolate_profile_u_over_m_log_values_requires_positive_floor(

) -> None:
    """Tests that invalid inputs to interpolate_profile_u_over_m raise errors."""
    mass, k = _grid_mass_k()
    prof = _ProfileNoUpd()

    with pytest.raises(ValueError, match=r"value_floor must be > 0"):
        _ = pi.interpolate_profile_u_over_m(
            cosmo={}, a=1.0, profile=prof, mass=mass, k=k,
            log_values=True, value_floor=0.0
        )


def test_interpolator_evaluates_and_broadcasts_log_axes_and_log_values(

) -> None:
    """Tests that the interpolator evaluates and broadcasts log_axes and
    log_values."""
    mass, k = _grid_mass_k()
    prof = _ProfileNoUpd()

    # prof.fourier returns u = mass[:,None], so u/M = 1 everywhere
    f = pi.interpolate_profile_u_over_m(
        cosmo={}, a=1.0, profile=prof, mass=mass, k=k,
        log_axes=True, log_values=True
    )

    Mq = np.array([1e13, 3e13], dtype=float)[:, None]
    kq = np.array([1e-2, 1e-1, 1.0], dtype=float)[None, :]
    out = f(Mq, kq)

    assert out.shape == (2, 3)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 1.0, rtol=0, atol=0)


def test_interpolator_rejects_nonpositive_queries_when_log_axes_true() -> None:
    """Tests that the interpolator rejects nonpositive queries when
    log_axes=True."""
    mass, k = _grid_mass_k()
    prof = _ProfileNoUpd()

    f = pi.interpolate_profile_u_over_m(
        cosmo={},
        a=1.0,
        profile=prof,
        mass=mass,
        k=k,
        log_axes=True)

    with pytest.raises(ValueError, match=r"must be > 0"):
        _ = f(np.array([0.0, 1.0]), np.array([0.1, 0.2]))

    with pytest.raises(ValueError, match=r"must be > 0"):
        _ = f(np.array([1.0, 2.0]), np.array([0.0, 0.2]))


def test_interpolator_allows_nonpositive_queries_when_log_axes_false() -> None:
    """Tests that the interpolator allows nonpositive queries when
    log_axes=False."""
    mass, k = _grid_mass_k()
    prof = _ProfileNoUpd()

    f = pi.interpolate_profile_u_over_m(
        cosmo={},
        a=1.0,
        profile=prof,
        mass=mass,
        k=k,
        log_axes=False)

    out = f(np.array([-1.0, 2.0]), np.array([0.1, -0.2]))
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))


def test_interpolator_bounds_error_true_raises_out_of_bounds() -> None:
    """Tests that the interpolator raises an error when bounds_error=True and
    the query is out of bounds."""
    mass, k = _grid_mass_k()
    prof = _ProfileNoUpd()

    f = pi.interpolate_profile_u_over_m(
        cosmo={}, a=1.0, profile=prof, mass=mass, k=k,
        bounds_error=True, log_axes=True
    )

    # out-of-bounds mass (below min mass)
    with pytest.raises(ValueError):
        _ = f(np.array([1e12]), np.array([1e-2]))


def test_build_profile_interpolators_validates_mapping_types() -> None:
    """Tests that invalid inputs to build_profile_interpolators raise errors."""
    mass, k = _grid_mass_k()
    prof = _ProfileNoUpd()

    grid = {
        "dark_matter": {"mass": mass, "k": k},
        "gas": {"mass": mass, "k": k},
        "stars": {"mass": mass, "k": k},
    }
    profiles = {"dark_matter": prof, "gas": prof, "stars": prof}

    with pytest.raises(TypeError, match=r"profiles"):
        _ = pi.build_profile_interpolators(
            cosmo={},
            a=1.0,
            interpolation_grid=grid,
            profiles=123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"interpolation_grid"):
        _ = pi.build_profile_interpolators(
            cosmo={},
            a=1.0,
            interpolation_grid=123,
            profiles=profiles)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"fftlog_kwargs"):
        _ = pi.build_profile_interpolators(
            cosmo={}, a=1.0, interpolation_grid=grid, profiles=profiles,
            fftlog_kwargs=123  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match=r"rgi_kwargs"):
        _ = pi.build_profile_interpolators(
            cosmo={}, a=1.0, interpolation_grid=grid, profiles=profiles,
            rgi_kwargs=123  # type: ignore[arg-type]
        )


def test_build_profile_interpolators_missing_components_raises_keyerror(

) -> None:
    """Tests that invalid inputs to build_profile_interpolators raise errors."""
    mass, k = _grid_mass_k()
    prof = _ProfileNoUpd()

    grid = {"dark_matter": {"mass": mass, "k": k}}  # missing gas/stars
    profiles = {"dark_matter": prof, "gas": prof, "stars": prof}

    with pytest.raises(
            KeyError,
            match=r"Missing profiles|missing grids|Fedeli requires"):
        _ = pi.build_profile_interpolators(
            cosmo={},
            a=1.0,
            interpolation_grid=grid,
            profiles=profiles)


def test_build_profile_interpolators_grid_schema_checks() -> None:
    """Tests that invalid inputs to build_profile_interpolators raise errors."""
    mass, k = _grid_mass_k()
    prof = _ProfileNoUpd()
    profiles = {"dark_matter": prof, "gas": prof, "stars": prof}

    grid_bad_type = {
        "dark_matter": {"mass": mass, "k": k},
        "gas": 123,  # not a mapping
        "stars": {"mass": mass, "k": k},
    }
    with pytest.raises(TypeError, match=r"must be a mapping"):
        _ = pi.build_profile_interpolators(
            cosmo={},
            a=1.0,
            interpolation_grid=grid_bad_type,
            profiles=profiles)

    grid_missing_key = {
        "dark_matter": {"mass": mass, "k": k},
        "gas": {"mass": mass},  # missing 'k'
        "stars": {"mass": mass, "k": k},
    }
    with pytest.raises(KeyError, match=r"must contain keys"):
        _ = pi.build_profile_interpolators(
            cosmo={},
            a=1.0,
            interpolation_grid=grid_missing_key,
            profiles=profiles)


def test_build_profile_interpolators_subset_components_and_update_toggle(
) -> None:
    """Tests that build_profile_interpolators supports subset builds and
    FFTLog updates."""
    mass, k = _grid_mass_k()
    called: dict[str, Any] = {}
    prof_upd = _ProfileUpdOK(called=called)

    grid = {
        "dark_matter": {"mass": mass, "k": k},
        "gas": {"mass": mass, "k": k},
        "stars": {"mass": mass, "k": k},
    }
    profiles = {"dark_matter": prof_upd, "gas": prof_upd, "stars": prof_upd}

    # subset build: only gas
    out = pi.build_profile_interpolators(
        cosmo={},
        a=1.0,
        interpolation_grid=grid,
        profiles=profiles,
        components=("gas",),
        update_fftlog_precision=False,  # ensure we hit the "no update" branch
    )
    assert set(out.keys()) == {"gas"}
    assert callable(out["gas"])

    # full build with update: ensure update_precision_fftlog gets invoked
    called.clear()
    out2 = pi.build_profile_interpolators(
        cosmo={},
        a=1.0,
        interpolation_grid=grid,
        profiles=profiles,
        update_fftlog_precision=True,
        fftlog_kwargs={"n_per_decade": 999},
    )
    assert set(out2.keys()) == set(pi.FEDELI_COMPONENTS)
    assert "kwargs" in called
    assert called["kwargs"]["n_per_decade"] == 999

    # sanity check returned interpolator works
    f = out2["dark_matter"]
    val = f(np.array([2e13]), np.array([1e-1]))
    assert np.all(np.isfinite(val))
