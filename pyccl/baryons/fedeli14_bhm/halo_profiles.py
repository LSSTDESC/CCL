"""Halo profile models used by the Fedeli14 baryonification setup."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from ...halos.profiles import HaloProfile, HaloProfileNFW
from ...halos.concentration import ConcentrationDuffy08
from .numerics import _pos_float, _pos_int, _trapz_compat


__all__ = (
    "GasHaloProfile",
    "StellarHaloProfile",
    "nfw_profile",
    "nfw_profile_dmo",
)

MassFracFn = Callable[[np.ndarray | float], np.ndarray | float]


def nfw_profile(mass_def, concentration=None):
    """Return a dark-matter-only NFW halo profile.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): Halo mass definition.
        concentration (:class:`~pyccl.halos.concentration.Concentration`
            or :obj:`str` or callable or :obj:`None`): Concentration-mass
            relation. If ``None``, uses
            :class:`~pyccl.halos.concentration.ConcentrationDuffy08`
            for the supplied ``mass_def``.

    Returns:
        :class:`~pyccl.halos.profiles.HaloProfileNFW`: NFW halo profile.
    """
    if mass_def is None:
        raise TypeError("mass_def must be provided.")

    if concentration is not None and not isinstance(concentration,
                                                    str) and not callable(
            concentration):
        raise TypeError(
            "concentration must be a CCL Concentration (callable) or"
            " a callable (cosmo, M, a) -> c."
        )

    if concentration is None:
        concentration = ConcentrationDuffy08(mass_def=mass_def)

    return HaloProfileNFW(mass_def=mass_def, concentration=concentration)


def nfw_profile_dmo(*, mass_def, concentration_dmo=None):
    """Return the DMO baseline NFW profile.

    This helper exists so the rest of the baryonification setup can keep a
    stable dark-matter-only (DMO) reference profile even if the baryonic
    ``dark_matter`` profile is modified elsewhere.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): Halo mass definition.
        concentration_dmo (:class:`~pyccl.halos.concentration.Concentration`
            or :obj:`str` or callable or :obj:`None`): Concentration-mass
            relation for the DMO profile. If ``None``, the default in
            :func:`nfw_profile` is used.

    Returns:
        :class:`~pyccl.halos.profiles.HaloProfileNFW`: DMO NFW halo profile.
    """
    return nfw_profile(mass_def=mass_def, concentration=concentration_dmo)


class GasHaloProfile(HaloProfile):
    """Fedeli14 gas density profile.

    Gas density profile used by the Fedeli14 baryonification model. The profile
    is normalized so that the enclosed gas mass equals
    :math:`f_{\\rm gas}(M) M` for a halo of mass :math:`M`.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): Halo mass definition.
        f_gas (callable): Gas mass fraction :math:`f_{\\rm gas}(M)`.
        beta (:obj:`float`): Outer slope parameter.
        r_co (:obj:`float`): Core radius in units of the halo radius.
        r_ej (:obj:`float`): Ejection radius in units of the halo radius.
        x_min (:obj:`float`): Minimum dimensionless radius for the internal
            normalization grid.
        x_max (:obj:`float`): Maximum dimensionless radius for the internal
            normalization grid.
        n_x (:obj:`int`): Number of samples in the internal normalization grid.

    Notes:
        The normalization is computed numerically on a fixed dimensionless
        radius grid :math:`x=r/R`, where :math:`R` is the halo radius returned
        by ``mass_def``. Radii are treated as comoving, following the CCL
        convention that ``mass_def.get_radius`` returns a physical radius.

    Raises:
        TypeError: If ``f_gas`` is not callable.
        ValueError: If any inputs are invalid, or if the normalization fails.
    """
    def __init__(
        self,
        *,
        mass_def,
        f_gas: MassFracFn,
        beta: float = 2.9,
        r_co: float = 0.1,
        r_ej: float = 4.5,
        x_min: float = 1.0e-3,
        x_max: float = 50.0,
        n_x: int = 500,
    ):
        """Initializes the gas profile class."""
        if mass_def is None:
            raise TypeError("mass_def must be provided.")
        if not callable(f_gas):
            raise TypeError("f_gas must be callable.")
        beta = _pos_float(beta, "beta")
        r_co = _pos_float(r_co, "r_co")
        r_ej = _pos_float(r_ej, "r_ej")
        x_min = _pos_float(x_min, "x_min")
        x_max = _pos_float(x_max, "x_max")
        if x_max <= x_min:
            raise ValueError("x_max must be > x_min.")
        n_x = _pos_int(n_x, "n_x")
        if n_x < 2:
            raise ValueError("n_x must be >= 2.")

        super().__init__(mass_def=mass_def)
        self._f_gas = f_gas
        self._beta = beta
        self._r_co = r_co
        self._r_ej = r_ej

        self._x = np.linspace(x_min, x_max, n_x)

    def _rs(self, cosmo, M, a):
        """Return the halo radius for each mass."""
        a = _pos_float(a, "a")
        # CCL convention: get_radius returns physical;
        # divide by a for comoving.
        R = self.mass_def.get_radius(cosmo, M, a) / a
        if np.any(~np.isfinite(R)) or np.any(np.asarray(R) <= 0.0):
            raise ValueError("mass_def.get_radius returned invalid radii.")
        return R

    def _norm(self, cosmo, M, a):
        """Return the gas-density normalization for each halo mass."""
        M_use = np.atleast_1d(M).astype(float)
        if np.any(~np.isfinite(M_use)) or np.any(M_use <= 0.0):
            raise ValueError("M must be finite and > 0.")

        Rd = np.asarray(self._rs(cosmo, M_use, a), dtype=float)

        x = self._x
        beta = self._beta

        u = x[None, :] / self._r_co
        v = x[None, :] / self._r_ej
        prof_shape = x[None, :] ** 2 / (
            (1.0 + u) ** beta * (1.0 + v**2) ** ((7.0 - beta) / 2.0)
        )

        integ = _trapz_compat(prof_shape, x, axis=-1)

        if np.any(~np.isfinite(integ)) or np.any(integ <= 0.0):
            raise ValueError("Gas normalization integral is invalid"
                             " (<=0 or non-finite).")

        fgas = np.asarray(self._f_gas(M_use), dtype=float)
        if fgas.shape != M_use.shape:
            raise ValueError("f_gas(M) must return an array with the same"
                             " shape as M."
                             f"Got f_gas(M).shape={fgas.shape},"
                             f" M.shape={M_use.shape}.")
        if np.any(~np.isfinite(fgas)) or np.any(fgas < 0.0):
            raise ValueError("f_gas(M) must be finite and >= 0.")

        rho0 = fgas * M_use / (4.0 * np.pi * (Rd**3) * integ)
        if np.any(~np.isfinite(rho0)) or np.any(rho0 < 0.0):
            raise ValueError("Gas normalization produced invalid values.")

        if np.ndim(M) == 0:
            return float(rho0[0])
        return rho0

    def _real(self, cosmo, r, M, a):
        """Return real-space gas density profile."""
        r_use = np.atleast_1d(r).astype(float)
        M_use = np.atleast_1d(M).astype(float)

        if np.any(~np.isfinite(r_use)) or np.any(r_use < 0.0):
            raise ValueError("r must be finite and >= 0.")
        if np.any(~np.isfinite(M_use)) or np.any(M_use <= 0.0):
            raise ValueError("M must be finite and > 0.")

        Rd = np.asarray(self._rs(cosmo, M_use, a), dtype=float)
        rho0 = np.asarray(self._norm(cosmo, M_use, a), dtype=float)

        x = r_use[None, :] / Rd[:, None]  # dimensionless radius
        beta = self._beta

        prof = rho0[:, None] / (
            (1.0 + x / self._r_co) ** beta
            * (1.0 + (x / self._r_ej) ** 2) ** ((7.0 - beta) / 2.0)
        )

        if np.any(~np.isfinite(prof)) or np.any(prof < 0.0):
            raise ValueError("Gas profile evaluation produced invalid values.")

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof


class StellarHaloProfile(HaloProfile):
    """Fedeli14 stellar density profile.

    Stellar density profile used by the Fedeli14 baryonification model. The
    profile is normalized so that the enclosed stellar mass equals
    :math:`f_\\star(M) M` for a halo of mass :math:`M`.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef`): Halo mass definition.
        f_star (callable): Stellar mass fraction :math:`f_\\star(M)`.
        x_delta (:obj:`float`): Dimensionless truncation parameter defining the
            truncation radius :math:`r_t = R/x_\\Delta`, where :math:`R` is the
            halo radius for ``mass_def``.
        alpha (:obj:`float`): Exponential shape parameter.

    Notes:
        Radii are treated as comoving, following the CCL convention that
        ``mass_def.get_radius`` returns a physical radius.

    Raises:
        TypeError: If ``f_star`` is not callable.
        ValueError: If any inputs are invalid, or if the profile evaluation
            produces non-finite values.
    """
    def __init__(
        self,
        *,
        mass_def,
        f_star: MassFracFn,
        x_delta: float = 1.0 / 0.03,
        alpha: float = 1.0,
    ):
        """Initializes the stellar profile class."""
        if mass_def is None:
            raise TypeError("mass_def must be provided.")
        if not callable(f_star):
            raise TypeError("f_star must be callable.")

        x_delta = _pos_float(x_delta, "x_delta")
        alpha = _pos_float(alpha, "alpha")

        super().__init__(mass_def=mass_def)
        self._f_star = f_star
        self._x_delta = x_delta
        self._alpha = alpha

    def _rs(self, cosmo, M, a):
        """Return the halo radius for each mass."""
        a = _pos_float(a, "a")
        R = self.mass_def.get_radius(cosmo, M, a) / a
        if np.any(~np.isfinite(R)) or np.any(np.asarray(R) <= 0.0):
            raise ValueError("mass_def.get_radius returned invalid radii.")
        return R

    def _real(self, cosmo, r, M, a):
        """Return real-space stellar density profile."""
        r_use = np.atleast_1d(r).astype(float)
        M_use = np.atleast_1d(M).astype(float)

        if np.any(~np.isfinite(r_use)) or np.any(r_use < 0.0):
            raise ValueError("r must be finite and >= 0.")
        if np.any(~np.isfinite(M_use)) or np.any(M_use <= 0.0):
            raise ValueError("M must be finite and > 0.")

        Rd = np.asarray(self._rs(cosmo, M_use, a), dtype=float)
        r_t = Rd / self._x_delta

        # Avoid division by zero at r=0.
        # CCL never really needs r=0, but it's good to be safe.
        x = r_use[None, :] / r_t[:, None]
        x_safe = np.where(x == 0.0, np.finfo(float).tiny, x)

        fstar = np.asarray(self._f_star(M_use), dtype=float)
        if fstar.shape != M_use.shape:
            raise ValueError("f_star(M) must return an array with the same"
                             " shape as M."
                             f"Got f_star(M).shape={fstar.shape}, M.shape"
                             f"={M_use.shape}.")

        if np.any(~np.isfinite(fstar)) or np.any(fstar < 0.0):
            raise ValueError("f_star(M) must be finite and >= 0.")

        rho_t = M_use * fstar / (4.0 * np.pi * r_t**3)
        prof = rho_t[:, None] * np.exp(-(x_safe**self._alpha)) / x_safe

        if np.any(~np.isfinite(prof)) or np.any(prof < 0.0):
            raise ValueError("Stellar profile evaluation produced"
                             " invalid values.")

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
