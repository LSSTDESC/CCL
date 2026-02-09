"""Unit tests for `pyccl.baryons.fedeli14_bhm.halo_profiles`."""

from __future__ import annotations

import numpy as np
import pytest

import pyccl as ccl

from pyccl.baryons.fedeli14_bhm.halo_profiles import (
    GasHaloProfile,
    StellarHaloProfile,
    nfw_profile,
    nfw_profile_dmo,
)


def _cosmo() -> ccl.Cosmology:
    """Return a simple, valid CCL cosmology."""
    return ccl.Cosmology(
        Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.965, sigma8=0.8
    )


def _mass_def_200c() -> ccl.halos.MassDef:
    """Mass definition used across halo-profile tests."""
    return ccl.halos.MassDef(200, "critical")


def _f_const(val: float):
    """Return f(M)=const with shape preserved."""
    v = float(val)

    def f(M):
        M = np.asarray(M, dtype=float)
        return np.full_like(M, v, dtype=float)

    return f


# -----------------------------------------------------------------------------
# nfw_profile / nfw_profile_dmo
# -----------------------------------------------------------------------------

def test_nfw_profile_requires_mass_def() -> None:
    with pytest.raises(TypeError, match=r"mass_def must be provided"):
        _ = nfw_profile(None)  # type: ignore[arg-type]


def test_nfw_profile_rejects_invalid_concentration_type() -> None:
    md = _mass_def_200c()
    with pytest.raises(TypeError, match=r"concentration must be"):
        _ = nfw_profile(md, concentration=123)  # type: ignore[arg-type]


def test_nfw_profile_accepts_concentration_instance_and_returns_nfw() -> None:
    md = _mass_def_200c()
    conc = ccl.halos.ConcentrationDuffy08(mass_def=md)
    prof = nfw_profile(md, concentration=conc)
    assert isinstance(prof, ccl.halos.HaloProfileNFW)


def test_nfw_profile_callable_concentration_errors_from_HaloProfileNFW() -> None:
    """Your wrapper allows callables, but CCL's HaloProfileNFW rejects them."""
    md = _mass_def_200c()

    def c_of_m(_cosmo, M, a):
        M = np.asarray(M, dtype=float)
        _ = a
        return np.ones_like(M)

    with pytest.raises(TypeError, match=r"Expected Concentration or str"):
        _ = nfw_profile(md, concentration=c_of_m)


def test_nfw_profile_uses_default_concentration_when_none() -> None:
    md = _mass_def_200c()
    prof = nfw_profile(md, concentration=None)
    assert isinstance(prof, ccl.halos.HaloProfileNFW)


def test_nfw_profile_dmo_delegates_to_nfw_profile_and_returns_nfw() -> None:
    md = _mass_def_200c()
    prof = nfw_profile_dmo(mass_def=md, concentration_dmo=None)
    assert isinstance(prof, ccl.halos.HaloProfileNFW)


# -----------------------------------------------------------------------------
# GasHaloProfile: init validation
# -----------------------------------------------------------------------------

def test_gas_profile_init_validates_inputs() -> None:
    md = _mass_def_200c()

    with pytest.raises(TypeError, match=r"mass_def must be provided"):
        _ = GasHaloProfile(mass_def=None, f_gas=_f_const(0.1))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"f_gas must be callable"):
        _ = GasHaloProfile(mass_def=md, f_gas=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"x_max must be > x_min"):
        _ = GasHaloProfile(mass_def=md, f_gas=_f_const(0.1), x_min=1.0, x_max=1.0)

    with pytest.raises(ValueError, match=r"n_x must be >= 2"):
        _ = GasHaloProfile(mass_def=md, f_gas=_f_const(0.1), n_x=1)

    with pytest.raises(ValueError, match=r"beta"):
        _ = GasHaloProfile(mass_def=md, f_gas=_f_const(0.1), beta=-1.0)

    with pytest.raises(ValueError, match=r"r_co"):
        _ = GasHaloProfile(mass_def=md, f_gas=_f_const(0.1), r_co=0.0)

    with pytest.raises(ValueError, match=r"r_ej"):
        _ = GasHaloProfile(mass_def=md, f_gas=_f_const(0.1), r_ej=-1.0)

    with pytest.raises(ValueError, match=r"x_min"):
        _ = GasHaloProfile(mass_def=md, f_gas=_f_const(0.1), x_min=0.0)

    with pytest.raises(ValueError, match=r"x_max"):
        _ = GasHaloProfile(mass_def=md, f_gas=_f_const(0.1), x_max=0.0)

    with pytest.raises(ValueError, match=r"n_x"):
        _ = GasHaloProfile(mass_def=md, f_gas=_f_const(0.1), n_x=0)


# -----------------------------------------------------------------------------
# GasHaloProfile: _rs error paths (use real MassDef, patch get_radius)
# -----------------------------------------------------------------------------

def test_gas_profile_rs_validates_a_and_radius_values(monkeypatch) -> None:
    cosmo = _cosmo()
    M = np.array([1.0e14, 2.0e14], dtype=float)

    md = _mass_def_200c()
    gp = GasHaloProfile(mass_def=md, f_gas=_f_const(0.1))

    with pytest.raises(ValueError, match=r"\ba\b"):
        _ = gp._rs(cosmo, M, a=0.0)  # noqa: SLF001

    def bad_radius_nan(_cosmo, _M, _a):
        _M = np.asarray(_M, dtype=float)
        return np.full_like(_M, np.nan, dtype=float)

    monkeypatch.setattr(md, "get_radius", bad_radius_nan, raising=True)
    with pytest.raises(ValueError, match=r"get_radius returned invalid radii"):
        _ = gp._rs(cosmo, M, a=1.0)  # noqa: SLF001


# -----------------------------------------------------------------------------
# GasHaloProfile: _norm validation
# -----------------------------------------------------------------------------

def test_gas_profile_norm_validates_m_and_fgas_shape_and_values() -> None:
    cosmo = _cosmo()
    md = _mass_def_200c()
    gp = GasHaloProfile(mass_def=md, f_gas=_f_const(0.1))
    M = np.array([1.0e14, 2.0e14], dtype=float)

    with pytest.raises(ValueError, match=r"M must be finite and > 0"):
        _ = gp._norm(cosmo, np.array([1.0e14, -1.0], dtype=float), a=1.0)  # noqa: SLF001

    with pytest.raises(ValueError, match=r"M must be finite and > 0"):
        _ = gp._norm(cosmo, np.array([np.nan, 1.0e14], dtype=float), a=1.0)  # noqa: SLF001

    def fgas_bad_shape(m):
        m = np.asarray(m, dtype=float)
        return np.ones(m.size - 1, dtype=float)

    gp2 = GasHaloProfile(mass_def=md, f_gas=fgas_bad_shape)
    with pytest.raises(ValueError, match=r"must return an array with the same"):
        _ = gp2._norm(cosmo, M, a=1.0)  # noqa: SLF001

    gp3 = GasHaloProfile(mass_def=md, f_gas=_f_const(-0.1))
    with pytest.raises(ValueError, match=r"must be finite and >= 0"):
        _ = gp3._norm(cosmo, M, a=1.0)  # noqa: SLF001

    def fgas_nan(m):
        m = np.asarray(m, dtype=float)
        out = np.ones_like(m)
        out[0] = np.nan
        return out

    gp4 = GasHaloProfile(mass_def=md, f_gas=fgas_nan)
    with pytest.raises(ValueError, match=r"must be finite and >= 0"):
        _ = gp4._norm(cosmo, M, a=1.0)  # noqa: SLF001


def test_gas_profile_norm_returns_scalar_for_scalar_m() -> None:
    cosmo = _cosmo()
    gp = GasHaloProfile(mass_def=_mass_def_200c(), f_gas=_f_const(0.1))
    rho0 = gp._norm(cosmo, 1.0e14, a=1.0)  # noqa: SLF001
    assert isinstance(rho0, float)
    assert np.isfinite(rho0)
    assert rho0 >= 0.0


def test_gas_profile_norm_raises_if_integral_invalid() -> None:
    cosmo = _cosmo()
    md = _mass_def_200c()
    gp = GasHaloProfile(mass_def=md, f_gas=_f_const(0.1), x_min=1.0e-3, x_max=2.0e-3, n_x=2)

    # Force integral ~ 0 by zeroing x (x^2 factor kills integrand).
    gp._x = np.zeros(2, dtype=float)  # noqa: SLF001

    with pytest.raises(ValueError, match=r"Gas normalization integral is invalid"):
        _ = gp._norm(cosmo, np.array([1.0e14, 2.0e14], dtype=float), a=1.0)  # noqa: SLF001


# -----------------------------------------------------------------------------
# GasHaloProfile: _real validation + shape rules
# -----------------------------------------------------------------------------

def test_gas_profile_real_validates_inputs_and_shapes() -> None:
    cosmo = _cosmo()
    gp = GasHaloProfile(mass_def=_mass_def_200c(), f_gas=_f_const(0.1))

    with pytest.raises(ValueError, match=r"r must be finite and >= 0"):
        _ = gp._real(cosmo, r=np.array([0.1, -1.0]), M=1.0e14, a=1.0)  # noqa: SLF001

    with pytest.raises(ValueError, match=r"r must be finite and >= 0"):
        _ = gp._real(cosmo, r=np.array([0.1, np.nan]), M=1.0e14, a=1.0)  # noqa: SLF001

    with pytest.raises(ValueError, match=r"M must be finite and > 0"):
        _ = gp._real(cosmo, r=np.array([0.1, 1.0]), M=np.array([1.0e14, 0.0]), a=1.0)  # noqa: SLF001

    r = np.array([0.01, 0.1, 1.0], dtype=float)
    M = np.array([1.0e14, 2.0e14], dtype=float)

    prof = gp._real(cosmo, r=r, M=M, a=1.0)  # noqa: SLF001
    assert isinstance(prof, np.ndarray)
    assert prof.shape == (M.size, r.size)
    assert np.all(np.isfinite(prof))

    prof2 = gp._real(cosmo, r=0.1, M=M, a=1.0)  # noqa: SLF001
    assert isinstance(prof2, np.ndarray)
    assert prof2.shape == (M.size,)
    assert np.all(np.isfinite(prof2))

    prof3 = gp._real(cosmo, r=r, M=1.0e14, a=1.0)  # noqa: SLF001
    assert isinstance(prof3, np.ndarray)
    assert prof3.shape == (r.size,)
    assert np.all(np.isfinite(prof3))

    prof4 = gp._real(cosmo, r=0.1, M=1.0e14, a=1.0)  # noqa: SLF001
    assert np.isscalar(prof4)
    assert np.isfinite(float(prof4))


def test_gas_profile_real_raises_if_profile_invalid() -> None:
    """Hit the final safety check in _real by forcing negative rho0."""
    cosmo = _cosmo()
    gp = GasHaloProfile(mass_def=_mass_def_200c(), f_gas=_f_const(0.1))

    def bad_norm(_cosmo, M, a):
        M = np.atleast_1d(M).astype(float)
        _ = a
        return -np.ones_like(M)

    gp._norm = bad_norm  # type: ignore[method-assign]  # noqa: SLF001

    with pytest.raises(ValueError, match=r"Gas profile evaluation produced invalid values"):
        _ = gp._real(
            cosmo,
            r=np.array([0.1, 1.0]),
            M=np.array([1.0e14, 2.0e14]),
            a=1.0,
        )  # noqa: SLF001


# -----------------------------------------------------------------------------
# StellarHaloProfile: init + _rs + _real
# -----------------------------------------------------------------------------

def test_stellar_profile_init_validates_inputs() -> None:
    md = _mass_def_200c()

    with pytest.raises(TypeError, match=r"mass_def must be provided"):
        _ = StellarHaloProfile(mass_def=None, f_star=_f_const(0.01))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match=r"f_star must be callable"):
        _ = StellarHaloProfile(mass_def=md, f_star=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=r"x_delta"):
        _ = StellarHaloProfile(mass_def=md, f_star=_f_const(0.01), x_delta=0.0)

    with pytest.raises(ValueError, match=r"alpha"):
        _ = StellarHaloProfile(mass_def=md, f_star=_f_const(0.01), alpha=0.0)


def test_stellar_profile_rs_validates_a_and_radius_values(monkeypatch) -> None:
    cosmo = _cosmo()
    M = np.array([1.0e14, 2.0e14], dtype=float)

    md = _mass_def_200c()
    sp = StellarHaloProfile(mass_def=md, f_star=_f_const(0.01))

    with pytest.raises(ValueError, match=r"\ba\b"):
        _ = sp._rs(cosmo, M, a=0.0)  # noqa: SLF001

    def bad_radius_nan(_cosmo, _M, _a):
        _M = np.asarray(_M, dtype=float)
        return np.full_like(_M, np.nan, dtype=float)

    monkeypatch.setattr(md, "get_radius", bad_radius_nan, raising=True)
    with pytest.raises(ValueError, match=r"get_radius returned invalid radii"):
        _ = sp._rs(cosmo, M, a=1.0)  # noqa: SLF001


def test_stellar_profile_real_validates_inputs_fstar_and_shapes() -> None:
    cosmo = _cosmo()
    md = _mass_def_200c()

    sp = StellarHaloProfile(mass_def=md, f_star=_f_const(1.0e-4), x_delta=1.0 / 0.03, alpha=1.0)

    with pytest.raises(ValueError, match=r"r must be finite and >= 0"):
        _ = sp._real(cosmo, r=np.array([0.1, -1.0]), M=1.0e14, a=1.0)  # noqa: SLF001

    with pytest.raises(ValueError, match=r"M must be finite and > 0"):
        _ = sp._real(cosmo, r=np.array([0.1, 1.0]), M=np.array([1.0e14, 0.0]), a=1.0)  # noqa: SLF001

    def fstar_bad_shape(m):
        m = np.asarray(m, dtype=float)
        return np.ones(m.size - 1, dtype=float)

    sp2 = StellarHaloProfile(mass_def=md, f_star=fstar_bad_shape)
    with pytest.raises(ValueError, match=r"must return an array with the same shape"):
        _ = sp2._real(cosmo, r=np.array([0.1, 1.0]), M=np.array([1.0e14, 2.0e14]), a=1.0)  # noqa: SLF001

    sp3 = StellarHaloProfile(mass_def=md, f_star=_f_const(-0.01))
    with pytest.raises(ValueError, match=r"must be finite and >= 0"):
        _ = sp3._real(cosmo, r=np.array([0.1, 1.0]), M=np.array([1.0e14, 2.0e14]), a=1.0)  # noqa: SLF001

    # Valid evaluation: avoid r=0 to prevent overflow (even with x_safe).
    r = np.array([0.01, 0.1, 1.0], dtype=float)
    M = np.array([1.0e14, 2.0e14], dtype=float)
    prof = sp._real(cosmo, r=r, M=M, a=1.0)  # noqa: SLF001
    assert isinstance(prof, np.ndarray)
    assert prof.shape == (M.size, r.size)
    assert np.all(np.isfinite(prof))

    prof2 = sp._real(cosmo, r=0.1, M=M, a=1.0)  # noqa: SLF001
    assert prof2.shape == (M.size,)

    prof3 = sp._real(cosmo, r=r, M=1.0e14, a=1.0)  # noqa: SLF001
    assert prof3.shape == (r.size,)

    prof4 = sp._real(cosmo, r=0.1, M=1.0e14, a=1.0)  # noqa: SLF001
    assert np.isscalar(prof4)
    assert np.isfinite(float(prof4))


def test_stellar_profile_real_raises_if_profile_invalid(monkeypatch) -> None:
    """Force r_t -> 0 by making get_radius -> 0, which should trigger invalid profile."""
    cosmo = _cosmo()
    md = _mass_def_200c()
    sp = StellarHaloProfile(mass_def=md, f_star=_f_const(1.0e-4))

    def zero_radius(_cosmo, _M, _a):
        _M = np.asarray(_M, dtype=float)
        return np.zeros_like(_M, dtype=float)

    monkeypatch.setattr(md, "get_radius", zero_radius, raising=True)

    with pytest.raises(ValueError,
                       match=r"mass_def\.get_radius returned invalid radii"):
        _ = sp._real(
            cosmo,
            r=np.array([0.1, 1.0]),
            M=np.array([1.0e14, 2.0e14]),
            a=1.0,
        )  # noqa: SLF001# noqa: SLF001
