"""Fedeli14 halo mass-fraction functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import integrate
from scipy.special import erf

MassFracFn = Callable[[np.ndarray | float], np.ndarray | float]
MassFunction = Callable[[Any, np.ndarray, float], np.ndarray]

__all__ = ["mass_fractions"]


def mass_fractions(
    *,
    cosmo: Any,
    a: float,
    mass_function: MassFunction,
    rho_star: float | None = None,
    m0_star: float = 5.0e12,  # in Msun/h
    sigma_star: float = 1.2,
    mmin_star: float = 1.0e10,
    mmax_star: float = 1.0e15,
    m0_gas: float = 1.0e12,
    sigma_gas: float = 3.0,
) -> tuple[MassFracFn, MassFracFn, MassFracFn]:
    """Return Fedeli14 halo mass-fraction callables.

    This function returns three callables ``(f_gas, f_star, f_dm)`` mapping
    halo mass :math:`M` to gas, stellar, and dark-matter mass fractions.

    The stellar fraction normalization is set by matching a target cosmic
    stellar mass density :math:`\\rho_\\star` via

    .. math::
        \\rho_\\star = \\int d\\log_{10}M\\, M\\, f_\\star(M)\\,
        \\frac{dn}{d\\log_{10}M}(M,a).

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
        a (:obj:`float`): Scale factor.
        mass_function (callable): Halo mass function evaluated as
            ``mass_function(cosmo, M, a)``. It is assumed to return
            :math:`dn/d\\log_{10}M`.
        rho_star (:obj:`float` or :obj:`None`): Target cosmic stellar mass
            density used to normalize :math:`f_\\star`. If ``None``, a default
            scaled by :math:`h^2` is used.
        m0_star (:obj:`float`): Pivot mass for :math:`f_\\star` (default in
            :math:`M_\\odot/h`).
        sigma_star (:obj:`float`): Width of :math:`f_\\star` in
            :math:`\\log_{10}(M/m0_\\star)`.
        mmin_star (:obj:`float`): Lower integration bound for the stellar
            normalization.
        mmax_star (:obj:`float`): Upper integration bound for the stellar
            normalization.
        m0_gas (:obj:`float`): Characteristic mass for the gas transition.
        sigma_gas (:obj:`float`): Width of the gas transition in
            :math:`\\log_{10}(M/m0_{\\rm gas})`.

    Returns:
        tuple: Tuple ``(f_gas, f_star, f_dm)``. Each element is a callable
        accepting :obj:`float` or `array` halo masses and returning the
        corresponding mass fraction.

    Raises:
        TypeError: If ``mass_function`` is not callable.
        KeyError: If ``cosmo`` does not provide ``h``, ``Omega_b``, or
            ``Omega_m``.
        ValueError: If inputs are non-finite or non-positive, if
            :math:`\\Omega_b > \\Omega_m`, or if the stellar normalization
            integral is non-finite or non-positive.
    """
    # Basic input sanity checks
    if not np.isfinite(a) or a <= 0.0:
        raise ValueError(f"mass_fractions: a must be finite"
                         f" and > 0. Got a={a}.")

    if not callable(mass_function):
        raise TypeError("mass_fractions: mass_function must be callable.")

    for key in ("h", "Omega_b", "Omega_m"):
        try:
            _ = cosmo[key]
        except Exception as e:  # noqa: BLE001
            raise KeyError(f"mass_fractions:"
                           f" cosmo must provide '{key}'.") from e

    h = float(cosmo["h"])
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError(
            f"mass_fractions: cosmo['h'] must be finite and > 0. Got h={h}.")

    if rho_star is None:
        rho_star = 7.0e8 * h ** 2
    rho_star = float(rho_star)
    if not np.isfinite(rho_star) or rho_star <= 0.0:
        raise ValueError(
            "mass_fractions: rho_star must be finite and > 0. "
            f"Got rho_star={rho_star}."
        )

    omega_b = float(cosmo["Omega_b"])
    omega_m = float(cosmo["Omega_m"])
    if (not np.isfinite(omega_m)) or omega_m <= 0.0:
        raise ValueError(
            f"mass_fractions: cosmo['Omega_m'] must be finite "
            f"and > 0. Got {omega_m}.")
    if (not np.isfinite(omega_b)) or omega_b < 0.0:
        raise ValueError(
            f"mass_fractions: cosmo['Omega_b'] must be finite"
            f" and >= 0. Got {omega_b}.")
    if omega_b > omega_m:
        raise ValueError(
            "mass_fractions: require Omega_b <= Omega_m. "
            f"Got Omega_b={omega_b}, Omega_m={omega_m}."
        )

    omega_ratio = float(omega_b / omega_m)

    # Mass-scale parameter sanity checks (do not change conventions)
    for name, val in (
        ("m0_star", m0_star),
        ("sigma_star", sigma_star),
        ("mmin_star", mmin_star),
        ("mmax_star", mmax_star),
        ("m0_gas", m0_gas),
        ("sigma_gas", sigma_gas),
    ):
        val = float(val)
        if (not np.isfinite(val)) or val <= 0.0:
            raise ValueError(f"mass_fractions: {name} must be finite"
                             f" and > 0. Got {val}.")
    if not (float(mmin_star) < float(mmax_star)):
        raise ValueError(
            "mass_fractions: require mmin_star < mmax_star. "
            f"Got mmin_star={mmin_star}, mmax_star={mmax_star}."
        )

    # Detect what mass units the mass function expects.
    # We test the mass function at a representative mass:
    # if mf(M) is approx 0 but mf(M/h) is >0, then the MF likely expects
    # Msun/h.
    m_test = float(m0_star)
    mf_test_msun = float(np.atleast_1d(mass_function(cosmo,
                                                     m_test, a))[0])
    mf_test_msunh = float(np.atleast_1d(mass_function(cosmo,
                                                      m_test / h, a))[0])

    if (not np.isfinite(mf_test_msun)) or (not np.isfinite(mf_test_msunh)):
        raise ValueError(
            "mass_fractions: mass_function returned non-finite values"
            " at the test mass. Got mf(M0)={mf_test_msun},"
            " mf(M0/h)={mf_test_msunh}."
        )

    # scale to apply to masses before calling mass_function
    # (Msun -> Msun/h if needed)
    if (mf_test_msun == 0.0) and (mf_test_msunh > 0.0):
        mf_mass_scale = 1.0 / h
    else:
        mf_mass_scale = 1.0

    A_cache: dict[str, float] = {}

    def _compute_stellar_A() -> float:
        """Return the cached stellar normalization factor."""
        A = A_cache.get("A")
        if A is not None:
            return A

        def integrand(m: float) -> float:
            """Integrand for stellar normalization integral."""
            # Use the same m for the Gaussian,
            # but convert ONLY for the MF call.
            m_mf = m * mf_mass_scale

            # Here we assume mass_function returns dn/dlog10M (common in CCL).
            # If that's true, then the integrand for rho_star is:
            # rho_* = integral dlog10M [ M * f_*(M) * dn/dlog10M ]
            # and since quad integrates over dm,
            # we include dlog10M/dm = 1/(m ln 10)
            dn_dlog10m = mass_function(cosmo, m_mf, a)
            dn_dlog10m = float(np.atleast_1d(dn_dlog10m)[0])
            if not np.isfinite(dn_dlog10m):
                raise ValueError(
                    "mass_fractions: mass_function returned non-finite"
                    " dn/dlog10M at m={m}, m_mf={m_mf}, a={a}."
                )
            dn_dm = dn_dlog10m / (m * np.log(10.0))

            numer = -(np.log10(m / m0_star)) ** 2
            denom = 2.0 * sigma_star ** 2
            fshape = np.exp(numer / denom)
            return m * fshape * dn_dm

        integral = integrate.quad(
            integrand,
            mmin_star,
            mmax_star,
            epsabs=0.0,
            epsrel=1.0e-3,
            limit=5000,
        )[0]

        if not np.isfinite(integral) or integral <= 0.0:
            raise ValueError(
                "mass_fractions: stellar normalization integral is "
                f"{integral}. This usually means the halo mass function is ~0 "
                "over the integration range (often a Msun vs Msun/h mismatch),"
                " or the dn/dlog10M vs dn/dM Jacobian is wrong.\n"
                f"Diagnostics: h={h}, a={a}, "
                f"mf(M0)={mf_test_msun}, mf(M0/h)={mf_test_msunh}, "
                f"mf_mass_scale={'1/h' if mf_mass_scale != 1.0 else '1'}, "
                f"mmin_star={mmin_star:g}, mmax_star={mmax_star:g}, "
                f"m0_star={m0_star:g}, sigma_star={sigma_star:g}."
            )

        A = float(rho_star / integral)
        A_cache["A"] = A
        return A

    def f_star(M):
        """Stellar mass fraction as a function of halo mass."""
        scalar = np.ndim(M) == 0
        M = np.atleast_1d(M).astype(float)

        if np.any(~np.isfinite(M)) or np.any(M <= 0.0):
            raise ValueError("mass_fractions: M must be finite and > 0.")

        A = _compute_stellar_A()
        out = A * np.exp(-(np.log10(M / m0_star)) ** 2 / (2.0 * sigma_star**2))
        return float(out[0]) if scalar else out

    def f_gas(M):
        """Gas mass fraction as a function of halo mass."""
        scalar = np.ndim(M) == 0
        M = np.atleast_1d(M).astype(float)

        if np.any(~np.isfinite(M)) or np.any(M <= 0.0):
            raise ValueError("mass_fractions: M must be finite and > 0.")

        out = np.zeros_like(M)
        mask = M >= m0_gas
        if np.any(mask):
            M_o_m = M[mask] / m0_gas
            out[mask] = omega_ratio * erf(np.log10(M_o_m) / sigma_gas)
        return float(out[0]) if scalar else out

    def f_dm(M):
        """Dark-matter mass fraction as a function of halo mass."""
        scalar = np.ndim(M) == 0
        M = np.atleast_1d(M).astype(float)

        if np.any(~np.isfinite(M)) or np.any(M <= 0.0):
            raise ValueError("mass_fractions: M must be finite and > 0.")

        out = np.full_like(M, 1.0 - omega_ratio)
        return float(out[0]) if scalar else out

    return f_gas, f_star, f_dm
