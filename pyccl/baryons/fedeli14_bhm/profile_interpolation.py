from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from scipy import interpolate

from pyccl.baryons.fedeli14_bhm.numerics import (
    _require_attr,
    _require_float,
    _require_mapping,
)

__all__ = [
    "update_precision_fftlog",
    "interpolate_profile_u_over_m",
    "build_profile_interpolators",
    "FEDELI_COMPONENTS",
]

ProfileEvalFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

FEDELI_COMPONENTS = ("dark_matter", "gas", "stars")


def update_precision_fftlog(
    profile: Any,
    /,
    *,
    padding_hi_fftlog: float = 1.0e3,
    padding_lo_fftlog: float = 1.0e-3,
    n_per_decade: int = 1000,
    plaw_fourier: float = -2.0,
) -> Any:
    """Update FFTLog settings on a halo profile, if supported.

    If ``profile`` provides a ``update_precision_fftlog`` method, call it with
    the provided keyword arguments and return ``profile``. Otherwise, return
    ``profile`` unchanged.

    Args:
        profile: Halo profile-like object.
        padding_hi_fftlog (:obj:`float`): High-k padding factor.
        padding_lo_fftlog (:obj:`float`): Low-k padding factor.
        n_per_decade (:obj:`int`): FFTLog sampling density.
        plaw_fourier (:obj:`float`): FFTLog Fourier power-law tilt.

    Returns:
        The input ``profile`` (possibly modified in-place).

    Raises:
        TypeError: If ``update_precision_fftlog`` exists but is not callable.
    """
    upd = getattr(profile, "update_precision_fftlog", None)
    if upd is None:
        return profile
    if not callable(upd):
        raise TypeError(
            "profile.update_precision_fftlog exists but is not callable.")

    upd(
        padding_hi_fftlog=padding_hi_fftlog,
        padding_lo_fftlog=padding_lo_fftlog,
        n_per_decade=n_per_decade,
        plaw_fourier=plaw_fourier,
    )
    return profile


def interpolate_profile_u_over_m(
    *,
    cosmo: Any,
    a: float,
    profile: Any,
    mass: np.ndarray,
    k: np.ndarray,
    method: str = "linear",
    bounds_error: bool = False,
    fill_value: float | None = None,
    log_axes: bool = True,
    log_values: bool = False,
    value_floor: float = 1.0e-300,
) -> ProfileEvalFn:
    """Return a callable for evaluating :math:`u(k,M)/M` by interpolation.

    The returned function evaluates the Fourier-space halo profile
    ``profile.fourier(cosmo, k, M, a)`` on a fixed mass and wavenumber grid,
    converts it to :math:`u/M`, and interpolates it in 2D.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float`): Scale factor.
        profile: Halo profile-like object implementing a ``fourier`` method.
        mass (:obj:`numpy.ndarray`): 1D, strictly increasing mass grid.
        k (:obj:`numpy.ndarray`): 1D, strictly increasing wavenumber grid.
        method (:obj:`str`): Interpolation method passed to
            :class:`scipy.interpolate.RegularGridInterpolator`.
        bounds_error (:obj:`bool`): If True, raise an error for query points
            outside the grid. If False, use ``fill_value`` or clamp behavior as
            implemented by ``RegularGridInterpolator``.
        fill_value (:obj:`float` or :obj:`None`): Fill value for out-of-bounds
            queries when ``bounds_error=False``. If None, SciPy uses its
            default behavior.
        log_axes (:obj:`bool`): If True, interpolate over
            :math:`(\\ln M, \\ln k)`.
        log_values (:obj:`bool`): If True, interpolate :math:`\\ln(u/M)`
            instead of :math:`u/M`.
        value_floor (:obj:`float`): Floor applied to values before taking logs
            when ``log_values=True``.

    Returns:
        callable: Function ``f(M, k)`` returning :math:`u(k|M)/M` evaluated at
        the supplied ``M`` and ``k`` (broadcasting over inputs).

    Raises:
        TypeError: If ``profile.fourier`` is missing or not callable.
        ValueError: If inputs are invalid, if grids are not 1D and strictly
            increasing, if ``log_axes=True`` with non-positive grid values, or
            if the Fourier grid has an unexpected shape.
    """
    mass = np.asarray(mass, dtype=float)
    k = np.asarray(k, dtype=float)

    _require_float("a", a)
    a = float(a)
    if not np.isfinite(a):
        raise ValueError("a must be finite.")

    if mass.size == 0 or k.size == 0:
        raise ValueError("mass and k must be non-empty 1D arrays.")

    if mass.ndim != 1 or k.ndim != 1:
        raise ValueError("mass and k must be 1D grid vectors.")
    if not (np.all(np.diff(mass) > 0) and np.all(np.diff(k) > 0)):
        raise ValueError("mass and k grids must be strictly increasing.")
    if log_axes and (np.any(mass <= 0.0) or np.any(k <= 0.0)):
        raise ValueError("mass and k grids must be > 0 if log_axes=True.")

    _require_attr(profile, "fourier", who="profile")
    if not callable(profile.fourier):
        raise TypeError("profile.fourier must be callable.")

    fourier_raw = np.asarray(profile.fourier(cosmo, k, mass, a), dtype=float)

    expected = (mass.size, k.size)
    if fourier_raw.shape != expected:
        raise ValueError(
            "profile.fourier must return an array of shape (len(mass),"
            " len(k)) = {expected}, got {fourier_raw.shape}."
        )

    values = fourier_raw / mass[:, None]  # u/M

    if log_values:
        if value_floor <= 0:
            raise ValueError("value_floor must be > 0 when log_values=True.")

    if log_values:
        values = np.log(np.maximum(values, float(value_floor)))

    xM = np.log(mass) if log_axes else mass
    xk = np.log(k) if log_axes else k

    rgi = interpolate.RegularGridInterpolator(
        (xM, xk),
        values,
        method=method,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )

    def eval_u_over_m(M: np.ndarray, kq: np.ndarray) -> np.ndarray:
        """Evaluate u(k|M)/M at given M and k."""
        M = np.asarray(M, dtype=float)
        kq = np.asarray(kq, dtype=float)
        M, kq = np.broadcast_arrays(M, kq)

        if log_axes:
            if np.any(M <= 0.0) or np.any(kq <= 0.0):
                raise ValueError("M and k must be > 0 when log_axes=True.")
            p0 = np.log(M.ravel())
            p1 = np.log(kq.ravel())
        else:
            p0 = M.ravel()
            p1 = kq.ravel()

        pts = np.column_stack([p0, p1])
        out = rgi(pts).reshape(M.shape)

        if log_values:
            out = np.exp(out)

        return out

    return eval_u_over_m


def build_profile_interpolators(
    *,
    cosmo,
    a: float,
    interpolation_grid,
    profiles,
    components: tuple[str, ...] | None = None,
    update_fftlog_precision: bool = True,
    fftlog_kwargs=None,
    rgi_kwargs=None,
):
    """Build per-component interpolators for :math:`u(k,M)/M`.

    This is a convenience wrapper that constructs :math:`u/M` interpolators for
    multiple halo-profile components (e.g. ``dark_matter``, ``gas``, ``stars``)
    using per-component (mass, k) grids.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float`): Scale factor.
        interpolation_grid (:obj:`dict`): Mapping from component name to a
            mapping with keys ``'mass'`` and ``'k'`` defining the interpolation
            grid vectors.
        profiles (:obj:`dict`): Mapping from component name to halo
            profile-like objects implementing ``fourier``.
        components (:obj:`tuple` or :obj:`None`): Components to build. If None,
            defaults to ``FEDELI_COMPONENTS``.
        update_fftlog_precision (:obj:`bool`): If True, call
            :func:`update_precision_fftlog` on each profile before sampling.
        fftlog_kwargs (:obj:`dict` or :obj:`None`): Keyword arguments passed to
            :func:`update_precision_fftlog`.
        rgi_kwargs (:obj:`dict` or :obj:`None`): Keyword arguments forwarded to
            :func:`interpolate_profile_u_over_m`.

    Returns:
        dict: Mapping from component name to an interpolator callable
        ``f(M, k)`` returning :math:`u(k|M)/M`.

    Raises:
        TypeError: If ``profiles`` or ``interpolation_grid`` are not mappings,
            or if ``fftlog_kwargs`` / ``rgi_kwargs`` are not mappings when
            provided.
        KeyError: If required components are missing from ``profiles`` or
            ``interpolation_grid``.
    """
    _require_mapping(profiles, who="profiles")
    _require_mapping(interpolation_grid, who="interpolation_grid")

    if fftlog_kwargs is not None and not isinstance(fftlog_kwargs, Mapping):
        raise TypeError("fftlog_kwargs must be a mapping (dict-like) or None.")
    if rgi_kwargs is not None and not isinstance(rgi_kwargs, Mapping):
        raise TypeError("rgi_kwargs must be a mapping (dict-like) or None.")

    fftlog_kwargs = dict(fftlog_kwargs or {})
    rgi_kwargs = dict(rgi_kwargs or {})

    # NEW: allow subset builds (default = full Fedeli set)
    if components is None:
        components = FEDELI_COMPONENTS
    components = tuple(components)

    missing_p = [c for c in components if c not in profiles]
    missing_g = [c for c in components if c not in interpolation_grid]
    if missing_p or missing_g:
        raise KeyError(
            "Fedeli requires components "
            f"{components}. Missing profiles={missing_p},"
            f" missing grids={missing_g}"
        )

    for comp in components:
        grid = interpolation_grid[comp]
        if not isinstance(grid, Mapping):
            raise TypeError(
                f"interpolation_grid[{comp!r}] must be a mapping with keys"
                f" 'mass' and 'k'.")
        if "mass" not in grid or "k" not in grid:
            raise KeyError(
                f"interpolation_grid[{comp!r}] must contain keys"
                f" 'mass' and 'k'.")

    out = {}
    for comp in components:
        prof = profiles[comp]
        grid = interpolation_grid[comp]
        mass = np.asarray(grid["mass"], float)
        k = np.asarray(grid["k"], float)

        if update_fftlog_precision:
            prof = update_precision_fftlog(prof, **fftlog_kwargs)

        out[comp] = interpolate_profile_u_over_m(
            cosmo=cosmo, a=a, profile=prof, mass=mass, k=k, **rgi_kwargs
        )
    return out
