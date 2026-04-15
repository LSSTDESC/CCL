"""Power spectra calculator for baryon halo model by Fedeli (2014)."""

from __future__ import annotations

from typing import Any, Callable, TypeAlias

import numpy as np
import pyccl as ccl

from pyccl.baryons.fedeli14_bhm.numerics import (
    _require_a,
    _require_k,
    _require_callable,
    _pos_int,
    _require_profiles_u_over_m,
    _require_mass_ranges,
    _require_densities,
    _require_gas_params,
    _add_pair_aliases,
    _add_weighted_pair_aliases,
    _require_component,
    _trapz_compat
)

_INV_LN10 = 1.0 / np.log(10.0)
YCacheKey: TypeAlias = tuple[str, str, float, float, float, int, float, float]
YCache: TypeAlias = dict[YCacheKey, np.ndarray]


def _dndm_from_dndlog10m(dndlog10m: np.ndarray, m: np.ndarray) -> np.ndarray:
    r"""Convert dn/dlog10M to dn/dM for a mass grid M:

    .. math::
        \frac{dn}{dM} = \frac{1}{M \ln 10}\,\frac{dn}{d\log_{10} M}.

    Args:
        dndlog10m: Mass function evaluated as dn/dlog10M on the grid ``m``.
        m: Mass grid (must be positive).

    Returns:
        dn/dM evaluated on ``m``.
    """
    dndlog10m = np.asarray(dndlog10m, dtype=float)
    m = np.asarray(m, dtype=float)
    if dndlog10m.shape != m.shape:
        raise ValueError("dndlog10m and m must have the same shape.")
    if np.any(m <= 0.0):
        raise ValueError("m must be > 0.")
    return dndlog10m * _INV_LN10 / m


def _k_cache_key(k: float, ndp: int = 12) -> float:
    """Return a rounded float key for caching quantities as a function of k."""
    return float(np.round(float(k), ndp))


class FedeliPkCalculator:
    r"""Halo-model matter power spectrum for the Fedeli14 baryonification
    setup.

    This class builds halo-model power spectra for the Fedeli14 components
    (dark matter, gas, stars) using Fourier-space halo profiles u(k,M).

    .. math::
        u(k, M) \equiv
        \int d^3r\,\rho(r|M)\,e^{i\mathbf{k}\cdot\mathbf{r}},

        y(M,k) \equiv \frac{u(k,M)}{M}.

    The halo-model cross-spectrum is written as
    :math:`P_{ij}(k) = P_{ij}^{1h}(k) + P_{ij}^{2h}(k)`.

    It provides:
      - per-pair halo-model spectra (1-halo and 2-halo pieces),
      - the gas auto-spectrum including a diffuse gas contribution,
      - mixed cross-spectra between gas and (dark matter or stars),
      - an assembled "packet" convenient for plotting and diagnostics,
      - a dark-matter-only (DMO) halo-model baseline and the resulting boost.

    The user supplies:
      - a mass function dn/dlog10M(M,a),
      - a halo bias b(M,a),
      - per-component evaluators for u(k,M)/M on the requested k grid,
      - component mass ranges and background densities.

    Notes:
        All returned spectra are functions of ``self.k`` at the scale factor
        ``a``.
    """
    def __init__(
        self,
        *,
        cosmo: Any,
        a: float,
        k: np.ndarray,
        profiles_u_over_m: dict[str, Any],
        dmo_dm_u_over_m: Any | None = None,
        mass_function: Any,
        halo_bias: Any,
        mass_ranges: dict[str, dict[str, float]],
        densities: dict[str, float],
        gas_params: dict[str, float],
        n_m: int = 512,
    ):
        """Initializes the power spectrum calculator."""
        self.cosmo = cosmo
        self.a = _require_a(a)
        self.k = _require_k(k)

        _require_callable(mass_function, who="mass_function")
        _require_callable(halo_bias, who="halo_bias")
        self.hmf = mass_function
        self.hb = halo_bias

        self.y = _require_profiles_u_over_m(profiles_u_over_m)
        self.y_dmo_dm = (
            dmo_dm_u_over_m
            if dmo_dm_u_over_m is not None
            else self.y["dark_matter"]
        )
        self.mass_ranges = _require_mass_ranges(mass_ranges)
        self.rho = _require_densities(densities)
        self.Fg, self.bd = _require_gas_params(gas_params)

        self.n_m = _pos_int(n_m, "n_m")
        if self.n_m < 2:
            raise ValueError("n_m must be >= 2.")

        self._init_caches()

    def _init_caches(self) -> None:
        """Initialize internal caches for expensive intermediate quantities.

        Caches are used to avoid recomputing linear P(k), mass grids, profile
        evaluations, and halo-model integrals across repeated calls.
        """
        self._linpk_cache: dict[float, float] = {}
        self._P_lin: np.ndarray | None = None

        self._m_grid_cache: dict[tuple[float, float], np.ndarray] = {}
        self._dndm_cache: dict[tuple[float, float], np.ndarray] = {}
        self._b_cache: dict[tuple[float, float], np.ndarray] = {}

        # y(M,k) meshes and vector integrals
        self._y_cache: YCache = {}
        self._Ib_vec_cache: dict[
            tuple[str, str, float, float], np.ndarray] = {}
        self._I2_vec_cache: dict[
            tuple[str, str, str, float, float], np.ndarray] = {}

        # rho integrals from fractions
        self._rho_cache: dict[tuple, float] = {}

        # final assembled spectra caches
        self._pk_dmo: np.ndarray | None = None
        self._pk_packet_cache: dict[str, Any] | None = None

    def _invalidate_final_caches(self) -> None:
        """Clear cached outputs that depend on densities or gas mixing
        parameters."""
        self._pk_dmo = None
        self._pk_packet_cache = None

    def rho_from_fraction(
            self,
            *,
            f_of_m: Callable[[np.ndarray], np.ndarray],
            mmin: float,
            mmax: float,
            n_m: int | None = None,
            cache_key: str | None = None,
    ) -> float:
        r"""Compute a mean density from a halo mass fraction model.

        This evaluates the mass-weighted integral

        .. math::
            \rho_X = \int dM\,\frac{dn}{dM}\,M\,f_X(M).

        Here :math:`f_X(M)` is the (dimensionless) halo mass fraction in
        component X.

        Args:
            f_of_m: Function returning the fraction f_X(M) on an array of
                masses.
            mmin, mmax: Integration limits in mass.
            n_m: Number of mass samples used in the integral (optional).
            cache_key: Optional label used to cache the result for the given
                limits.

        Returns:
            rho_X in the same mass-density units implied by the mass function.
        """
        if not callable(f_of_m):
            raise TypeError("f_of_m must be callable.")

        mmin = float(mmin)
        mmax = float(mmax)
        if mmin <= 0.0 or mmax <= 0.0 or mmax <= mmin:
            raise ValueError(f"Invalid mass range: mmin={mmin}, mmax={mmax}")

        if n_m is None:
            n_m = max(1024, self.n_m)

        key = (cache_key, mmin, mmax, int(n_m))
        if cache_key is not None:
            hit = self._rho_cache.get(key)
            if hit is not None:
                return hit

        m = np.geomspace(mmin, mmax, int(n_m))
        dndlog10m = np.asarray(self.hmf(self.cosmo, m, self.a), dtype=float)
        dndm = _dndm_from_dndlog10m(dndlog10m, m)

        fx = np.asarray(f_of_m(m), dtype=float)
        if fx.shape != m.shape:
            raise ValueError("f_of_m must return an array with the"
                             " same shape as m.")
        if not np.all(np.isfinite(fx)):
            raise ValueError("f_of_m(m) must be finite.")

        rho_x = float(_trapz_compat(dndm * m * fx, m))

        if cache_key is not None:
            self._rho_cache[key] = rho_x
        return rho_x

    def ensure_densities(
        self,
        *,
        f_gas: Callable[[np.ndarray], np.ndarray],
        f_star: Callable[[np.ndarray], np.ndarray],
        mmin: float,
        mmax: float,
        n_m: int | None = None,
    ) -> None:
        """Ensure component mean densities required by the model are available.

        This fills missing entries in ``self.rho``:
          - 'matter' and 'dark_matter' from the cosmology,
          - 'gas' and 'stars' from the supplied fraction functions via
            :meth:`rho_from_fraction`.

        Call this before requesting total spectra or packets if you did not
        provide all densities at initialization.

        Args:
            f_gas: Function returning the fraction of gas mass in each halo.
            f_star: Function returning the fraction of star mass in each halo.
            mmin: Integration limit mininum mass in halo mass units.
            mmax: Integration limit maximum mass in halo mass units.
            n_m: Number of mass samples used in the integral (optional).

        Raises:
            KeyError: If any of the required cosmology parameters are missing.
        """
        for key in ("Omega_c", "Omega_m"):
            try:
                float(self.cosmo[key])
            except Exception as e:  # noqa: BLE001
                raise KeyError(
                    f"cosmo must provide {key!r}"
                    f" for dark_matter density.") from e

        changed = False

        if "matter" not in self.rho:
            self.rho["matter"] = float(ccl.rho_x(
                self.cosmo, self.a, "matter")
            )
            changed = True

        if "dark_matter" not in self.rho:
            rho_m = float(self.rho["matter"])
            rho_m = float(self.rho["matter"])
            omc = float(self.cosmo["Omega_c"])
            omm = float(self.cosmo["Omega_m"])
            self.rho["dark_matter"] = rho_m * (omc / omm)
            changed = True

        if "gas" not in self.rho:
            self.rho["gas"] = self.rho_from_fraction(
                f_of_m=f_gas, mmin=mmin, mmax=mmax, n_m=n_m, cache_key="gas"
            )
            changed = True

        if "stars" not in self.rho:
            self.rho["stars"] = self.rho_from_fraction(
                f_of_m=f_star, mmin=mmin, mmax=mmax, n_m=n_m, cache_key="stars"
            )
            changed = True

        if changed:
            self._invalidate_final_caches()

    def linpk(self, kval: float) -> float:
        """Return the linear matter power spectrum :math:`P_L(k,a)`
        at a single k."""
        kval = float(kval)
        key = _k_cache_key(kval, ndp=12)
        pk = self._linpk_cache.get(key)
        if pk is None:
            pk = float(ccl.linear_matter_power(self.cosmo, kval, self.a))
            self._linpk_cache[key] = pk
        return pk

    def P_lin(self) -> np.ndarray:
        """Return :math:`P_L(k,a)` evaluated on the stored k grid."""
        if self._P_lin is None:
            try:
                self._P_lin = np.asarray(
                    ccl.linear_matter_power(self.cosmo, self.k, self.a),
                    dtype=float,
                )
            except Exception:
                self._P_lin = np.array(
                    [self.linpk(float(kk)) for kk in self.k], dtype=float
                )
        return self._P_lin

    def _mass_grid(self, mmin: float, mmax: float) -> np.ndarray:
        """Return the internal log-spaced mass grid for an integration
        range."""
        key = (float(mmin), float(mmax))
        m = self._m_grid_cache.get(key)
        if m is None:
            if mmin <= 0.0 or mmax <= 0.0 or mmax <= mmin:
                raise ValueError(
                    f"Invalid mass range: mmin={mmin}, mmax={mmax}"
                )
            m = np.logspace(np.log10(mmin), np.log10(mmax), self.n_m)
            self._m_grid_cache[key] = m
        return m

    def _dndm(self, mmin: float, mmax: float) -> np.ndarray:
        """Return dn/dM on the internal mass grid for an integration range."""
        key = (float(mmin), float(mmax))
        arr = self._dndm_cache.get(key)
        if arr is None:
            m = self._mass_grid(mmin, mmax)
            dndlog10m = np.asarray(
                self.hmf(self.cosmo, m, self.a), dtype=float)

            if dndlog10m.shape != m.shape:
                raise ValueError("mass_function must return an array with the"
                                 " same shape as M.")
            arr = _dndm_from_dndlog10m(dndlog10m, m)
            self._dndm_cache[key] = arr
        return arr

    def _bias(self, mmin: float, mmax: float) -> np.ndarray:
        """Return the halo bias b(M,a) on the internal mass grid for an
        integration range."""
        key = (float(mmin), float(mmax))
        arr = self._b_cache.get(key)
        if arr is None:
            m = self._mass_grid(mmin, mmax)
            arr = np.asarray(self.hb(self.cosmo, m, self.a), dtype=float)
            if arr.shape != m.shape:
                raise ValueError("halo_bias must return an array with the"
                                 " same shape as M.")
            self._b_cache[key] = arr
        return arr

    def _y_grid_mm(
        self,
        comp: str,
        mmin: float,
        mmax: float,
        *,
        y_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        cache_tag: str = "",
    ) -> np.ndarray:
        """Return u(k,M)/M evaluated on the internal (M,k) mesh for one
        component."""
        k_sig = (int(self.k.size), float(self.k[0]), float(self.k[-1]))
        key = (cache_tag, comp, float(mmin), float(mmax), float(self.a),
               *k_sig)

        # Cache is only used for the "default" evaluator or when the caller
        # explicitly opts in by providing a cache_tag.
        use_cache = (y_fn is None) or (cache_tag != "")

        if use_cache:
            hit = self._y_cache.get(key)
            if hit is not None:
                return hit

        if y_fn is None:
            y_fn = self.y[comp]

        m = self._mass_grid(mmin, mmax)
        y = np.asarray(y_fn(m[:, None], self.k[None, :]), dtype=float)

        if y.shape != (m.size, self.k.size):
            raise ValueError(
                f"u_over_m({comp!r}) must return shape"
                f" {(m.size, self.k.size)}, got {y.shape}."
            )

        if use_cache:
            self._y_cache[key] = y

        return y

    def _Ib_vec(
        self,
        comp: str,
        mmin: float,
        mmax: float,
        *,
        y_fn=None,
        cache_tag=""
    ) -> np.ndarray:
        r"""Return the 2-halo bias-weighted integral I_b(k) for one component.

        .. math::
            I_b(k) = \int dM\,\frac{dn}{dM}\,b(M)\,M\,y(M,k),

        where :math:`y(M,k) = u(k,M)/M`.
        """
        _require_component(comp, allowed=self.y)
        _require_component(comp, allowed=self.mass_ranges)
        key = (cache_tag, comp, float(mmin), float(mmax))
        hit = self._Ib_vec_cache.get(key)
        if hit is not None:
            return hit

        m = self._mass_grid(mmin, mmax)
        dndm = self._dndm(mmin, mmax)
        b = self._bias(mmin, mmax)
        y = self._y_grid_mm(comp, mmin, mmax, y_fn=y_fn, cache_tag=cache_tag)

        Ib = _trapz_compat(
            dndm[:, None] * (m[:, None] * b[:, None]) * y, m, axis=0)
        Ib = np.asarray(Ib, dtype=float)
        if Ib.shape != (self.k.size,):
            raise ValueError("Internal error: Ib has wrong shape.")
        self._Ib_vec_cache[key] = Ib
        return Ib

    def _I2_vec(
        self,
        comp1: str,
        comp2: str,
        mmin: float,
        mmax: float,
        *,
        y1_fn=None,
        y2_fn=None,
        cache_tag=""
    ) -> np.ndarray:
        r"""Return the 1-halo profile integral I_2(k) for a component pair.

        .. math::
            I_2(k) = \int dM\,\frac{dn}{dM}\,M^2\,y_1(M,k)\,y_2(M,k),

        where :math:`y_i(M,k) = u_i(k,M)/M`.
        """
        _require_component(comp1, allowed=self.y)
        _require_component(comp2, allowed=self.y)
        _require_component(comp1, allowed=self.mass_ranges)
        _require_component(comp2, allowed=self.mass_ranges)

        key = (cache_tag, comp1, comp2, float(mmin), float(mmax))
        hit = self._I2_vec_cache.get(key)
        if hit is not None:
            return hit

        m = self._mass_grid(mmin, mmax)
        dndm = self._dndm(mmin, mmax)
        y1 = self._y_grid_mm(comp1, mmin, mmax, y_fn=y1_fn,
                             cache_tag=cache_tag)
        y2 = self._y_grid_mm(comp2, mmin, mmax, y_fn=y2_fn,
                             cache_tag=cache_tag)

        I2 = _trapz_compat(
            dndm[:, None] * (m[:, None] ** 2) * y1 * y2, m, axis=0)
        I2 = np.asarray(I2, dtype=float)
        if I2.shape != (self.k.size,):
            raise ValueError("Internal error: I2 has wrong shape.")
        self._I2_vec_cache[key] = I2
        return I2

    def pk_halo_pair(
        self,
        *,
        comp1: str,
        comp2: str,
        rho1: float,
        rho2: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Return the (1-halo, 2-halo) spectra for a component pair.

        This builds the standard halo-model decomposition
          ``P(k) = P_1h(k) + P_2h(k)`
        for the cross-spectrum between two components, normalized by the
        supplied mean densities rho1 and rho2

        .. math::
            P_{12}^{1h}(k) = \frac{I_2(k)}{\rho_1\,\rho_2},

            P_{12}^{2h}(k) =
            P_L(k)\,\frac{I_{b,1}(k)\,I_{b,2}(k)}{\rho_1\,\rho_2}.

        Args:
            comp1, comp2: Component names (must exist in ``profiles`` and
                ``mass_ranges``).
            rho1, rho2: Mean densities used to normalize the halo-model terms.

        Returns:
            Tuple (P_1h, P_2h), each an array on ``self.k``.
        """
        rho1 = float(rho1)
        rho2 = float(rho2)
        if rho1 <= 0.0 or rho2 <= 0.0:
            raise ValueError("rho1 and rho2 must be > 0.")

        for comp in (comp1, comp2):
            if comp not in self.mass_ranges:
                raise KeyError(f"Unknown component {comp!r} in mass_ranges.")
            if comp not in self.y:
                raise KeyError(f"Unknown component {comp!r} in profiles.")

        mmin1 = float(self.mass_ranges[comp1]["min"])
        mmax1 = float(self.mass_ranges[comp1]["max"])
        mmin2 = float(self.mass_ranges[comp2]["min"])
        mmax2 = float(self.mass_ranges[comp2]["max"])

        mmin = max(mmin1, mmin2)
        mmax = min(mmax1, mmax2)
        if mmax <= mmin:
            raise ValueError(f"No overlap mass range "
                             f"for ({comp1},{comp2}): [{mmin},{mmax}]")

        P_lin = self.P_lin()

        I2 = self._I2_vec(comp1, comp2, mmin, mmax)
        Ib1 = self._Ib_vec(comp1, mmin1, mmax1)
        Ib2 = self._Ib_vec(comp2, mmin2, mmax2)

        P1 = I2 / (rho1 * rho2)
        P2 = P_lin * (Ib1 * Ib2) / (rho1 * rho2)
        return P1, P2

    def weights(self) -> dict[str, float]:
        r"""Return dimensionless weights used to assemble the total matter
        spectrum.

        .. math::
            w_{ii} = \left(\frac{\rho_i}{\rho_m}\right)^2, \qquad
            w_{ij} = 2\,\frac{\rho_i\,\rho_j}{\rho_m^2}\quad (i \ne j).

        The total matter spectrum is written as a weighted sum of auto- and
        cross- spectra of the component fields. The weights are built from
        the component mean densities in ``self.rho``.
        """
        rho_dm = float(self.rho["dark_matter"])
        rho_g = float(self.rho["gas"])
        rho_s = float(self.rho["stars"])
        rho_m = float(self.rho["matter"])
        return {
            "w_dm": (rho_dm / rho_m) ** 2,
            "w_g": (rho_g / rho_m) ** 2,
            "w_s": (rho_s / rho_m) ** 2,
            "w_dm_g": 2.0 * rho_dm * rho_g / (rho_m**2),
            "w_g_s": 2.0 * rho_g * rho_s / (rho_m**2),
            "w_dm_s": 2.0 * rho_dm * rho_s / (rho_m**2),
        }

    def pair_halo_packet(
        self,
        *,
        comp1: str,
        comp2: str,
        rho1: float,
        rho2: float
    ) -> dict[str, Any]:
        """Return a plot-friendly packet for a standard halo-model component
         pair.

        The packet includes the 1-halo and 2-halo pieces and their sum.
        """
        P1, P2 = self.pk_halo_pair(
            comp1=comp1, comp2=comp2, rho1=rho1, rho2=rho2)

        return {
            "kind": "halo_pair",
            "comp1": comp1,
            "comp2": comp2,
            "rho1": float(rho1),
            "rho2": float(rho2),
            "terms": {"1h": P1, "2h": P2},
            "pk": P1 + P2,
        }

    def pair_gas_auto_packet(self) -> dict[str, Any]:
        """Return the gas auto-spectrum packet including diffuse gas mixing.

        In the Fedeli14 model, the gas field is a mixture of a halo-tracing
        component and a diffuse component. This method returns the
        decomposition used for the gas auto-spectrum, including the
        halo-halo, diffuse-diffuse, and cross terms.
        """
        rho_g = float(self.rho["gas"])
        P_lin = self.P_lin()

        hh_1h, hh_2h = self.pk_halo_pair(
            comp1="gas",
            comp2="gas",
            rho1=self.Fg * rho_g,
            rho2=self.Fg * rho_g,
        )

        diffuse = (self.bd**2) * P_lin

        mmin_g = float(self.mass_ranges["gas"]["min"])
        mmax_g = float(self.mass_ranges["gas"]["max"])
        Ib_g = self._Ib_vec("gas", mmin_g, mmax_g)
        diffuse_halo = self.bd * P_lin * Ib_g / (self.Fg * rho_g)

        pk = (
            (self.Fg**2) * (hh_1h + hh_2h)
            + (1.0 - self.Fg) ** 2 * diffuse
            + 2.0 * self.Fg * (1.0 - self.Fg) * diffuse_halo
        )

        return {
            "kind": "gas_auto",
            "mix": {"Fg": float(self.Fg), "bd": float(self.bd)},
            "rho_g": rho_g,
            "terms": {
                "1h": hh_1h,
                "2h": hh_2h,
                "diffuse": diffuse,
                "diffuse_halo": diffuse_halo,
            },
            "pk": pk,
        }

    def pair_mixed_cross_packet(self, *, comp: str) -> dict[str, Any]:
        """Return the cross-spectrum packet for (dark matter or stars) x gas.

        The gas field is mixed, so the cross-spectrum includes a halo-tracing
        piece and a diffuse-gas contribution controlled by the model mixing
        parameters.
        """
        if comp not in {"dark_matter", "stars"}:
            raise ValueError("comp must be 'dark_matter' or 'stars'.")

        rho_c = float(self.rho[comp])
        rho_g = float(self.rho["gas"])
        P_lin = self.P_lin()

        halo_1h, halo_2h = self.pk_halo_pair(
            comp1=comp,
            comp2="gas",
            rho1=rho_c,
            rho2=self.Fg * rho_g,
        )

        mmin = float(self.mass_ranges[comp]["min"])
        mmax = float(self.mass_ranges[comp]["max"])
        Ib = self._Ib_vec(comp, mmin, mmax)
        diffuse = self.bd * P_lin * Ib / rho_c

        pk = self.Fg * (halo_1h + halo_2h) + (1.0 - self.Fg) * diffuse

        return {
            "kind": "mixed_cross",
            "comp": comp,
            "mix": {"Fg": float(self.Fg), "bd": float(self.bd)},
            "rho_comp": rho_c,
            "rho_g": rho_g,
            "terms": {
                "Fg_halo_1h": self.Fg * halo_1h,
                "Fg_halo_2h": self.Fg * halo_2h,
                "1mFg_diffuse": (1.0 - self.Fg) * diffuse,
            },
            "pk": pk,
        }

    def pk_gas_auto(self) -> np.ndarray:
        """Return the full gas auto power spectrum P_gg(k)
        on the stored k grid."""
        return np.asarray(self.pair_gas_auto_packet()["pk"], dtype=float)

    def pk_dm_gas(self) -> np.ndarray:
        """Return the full dark-matter x gas cross power spectrum on the
         stored k grid."""
        return np.asarray(
            self.pair_mixed_cross_packet(comp="dark_matter")["pk"],
            dtype=float)

    def pk_star_gas(self) -> np.ndarray:
        """Return the full stars x gas cross power spectrum on the stored
         k grid."""
        return np.asarray(
            self.pair_mixed_cross_packet(comp="stars")["pk"],
            dtype=float)

    def pk_packet(self, *, use_cache: bool = True) -> dict[str, Any]:
        """Return a dictionary of spectra and metadata.

        The returned dictionary is intended for plotting and downstream
        analysis.
        It includes:
          - the k grid and scale factor,
          - reference spectra (linear, nonlinear, and a DMO halo-model
            baseline),
          - per-component and per-pair spectra,
          - weighted contributions used to form the total matter spectrum,
          - model metadata (mixing parameters, densities, weights).

        Args:
            use_cache: If True, reuse cached results when available.

        Returns:
            Dictionary containing spectra arrays on ``self.k`` and
            accompanying metadata.

        Raises:
            KeyError: If required densities are missing.
        """
        if use_cache and (self._pk_packet_cache is not None):
            return self._pk_packet_cache

        # sanity: these must exist
        for key in ("matter", "dark_matter", "gas", "stars"):
            if key not in self.rho:
                raise KeyError(
                    f"Missing density {key!r}. Call ensure_densities() first.")

        # reference spectra
        pk_lin = np.asarray(self.P_lin(), dtype=float)
        pk_nl = np.asarray(ccl.nonlin_matter_power(self.cosmo, self.k, self.a),
                           dtype=float)
        pk_dmo = np.asarray(self.pk_total_dmo(use_cache=use_cache),
                            dtype=float)

        # pair packets (decompositions)
        pairs: dict[str, Any] = {
            "dm_dm": self.pair_halo_packet(
                comp1="dark_matter",
                comp2="dark_matter",
                rho1=float(self.rho["dark_matter"]),
                rho2=float(self.rho["dark_matter"]),
            ),
            "stars_stars": self.pair_halo_packet(
                comp1="stars",
                comp2="stars",
                rho1=float(self.rho["stars"]),
                rho2=float(self.rho["stars"]),
            ),
            "dm_stars": self.pair_halo_packet(
                comp1="dark_matter",
                comp2="stars",
                rho1=float(self.rho["dark_matter"]),
                rho2=float(self.rho["stars"]),
            ),
            "gas_gas": self.pair_gas_auto_packet(),
            "dm_gas": self.pair_mixed_cross_packet(comp="dark_matter"),
            "stars_gas": self.pair_mixed_cross_packet(comp="stars"),
        }

        # weights + weighted contributions to total matter
        w_full = self.weights()
        weights = {
            "dm_dm": float(w_full["w_dm"]),
            "gas_gas": float(w_full["w_g"]),
            "stars_stars": float(w_full["w_s"]),
            "dm_gas": float(w_full["w_dm_g"]),
            "stars_gas": float(w_full["w_g_s"]),
            "dm_stars": float(w_full["w_dm_s"]),
        }

        contrib = {name: weights[name] * pairs[name]["pk"] for name in weights}

        pk_total = (
            contrib["dm_dm"]
            + contrib["gas_gas"]
            + contrib["stars_stars"]
            + contrib["dm_gas"]
            + contrib["stars_gas"]
            + contrib["dm_stars"]
        )

        # spectra dictionary
        pk = {
            "total": pk_total,
            "dm": pairs["dm_dm"]["pk"],
            "gas": pairs["gas_gas"]["pk"],
            "stars": pairs["stars_stars"]["pk"],
            "dm_gas": pairs["dm_gas"]["pk"],
            "stars_gas": pairs["stars_gas"]["pk"],
            "dm_stars": pairs["dm_stars"]["pk"],
            # If you want the *weighted* contributions too, keep these:
            "w_dm": contrib["dm_dm"],
            "w_gas": contrib["gas_gas"],
            "w_stars": contrib["stars_stars"],
            "w_dm_gas": contrib["dm_gas"],
            "w_stars_gas": contrib["stars_gas"],
            "w_dm_stars": contrib["dm_stars"],
        }

        # add commutative aliases (pk)
        _add_pair_aliases(pk, "dm", "gas")
        _add_pair_aliases(pk, "dm", "stars")
        _add_pair_aliases(pk, "stars", "gas")

        # add commutative aliases (weighted pair contributions)
        _add_weighted_pair_aliases(pk, "dm", "gas")
        _add_weighted_pair_aliases(pk, "dm", "stars")
        _add_weighted_pair_aliases(pk, "stars", "gas")

        out = {
            "grid": {"k": self.k, "a": float(self.a)},
            "pk_ref": {"pk_lin": pk_lin, "pk_nlin": pk_nl, "pk_dmo": pk_dmo},
            "pk": pk,
            "halo_pairs": pairs,
            "meta": {
                "mix": {"Fg": float(self.Fg), "bd": float(self.bd)},
                "rho": dict(self.rho),
                "weights": weights,
            },
        }

        if use_cache:
            self._pk_packet_cache = out
        return out

    def pk_total(self) -> np.ndarray:
        """Return the total Fedeli14 matter Pk on the stored k grid."""
        return np.asarray(self.pk_packet(use_cache=True)["pk"]["total"],
                          dtype=float)

    def pk_total_dmo(self, *, use_cache: bool = True) -> np.ndarray:
        """Return a dark-matter-only halo-model baseline spectrum.

        This uses a fixed DMO dark-matter profile evaluator (``y_dmo_dm``) so
        that the baseline is stable even if the baryonic dark-matter profile
         is modified.
        """
        if use_cache and (self._pk_dmo is not None):
            return self._pk_dmo

        if "matter" not in self.rho:
            self.rho["matter"] = float(ccl.rho_x(self.cosmo,
                                                 self.a,
                                                 "matter"))
        rho_m = float(self.rho["matter"])

        # Use baseline DMO DM profile evaluator,
        # NOT the (possibly modified) baryonic DM.
        mmin = float(self.mass_ranges["dark_matter"]["min"])
        mmax = float(self.mass_ranges["dark_matter"]["max"])

        P_lin = self.P_lin()

        I2 = self._I2_vec(
            "dark_matter", "dark_matter", mmin, mmax,
            y1_fn=self.y_dmo_dm, y2_fn=self.y_dmo_dm, cache_tag="dmo"
        )
        Ib = self._Ib_vec(
            "dark_matter", mmin, mmax,
            y_fn=self.y_dmo_dm, cache_tag="dmo"
        )

        P1 = I2 / (rho_m * rho_m)
        P2 = P_lin * (Ib * Ib) / (rho_m * rho_m)
        out = np.asarray(P1 + P2, dtype=float)

        if use_cache:
            self._pk_dmo = out
        return out

    def boost_hm_over_hm(self) -> np.ndarray:
        """Return the halo-model boost B(k) = P_baryon_HM(k) / P_dmo_HM(k)."""
        packet = self.pk_packet(use_cache=True)
        pb = np.asarray(packet["pk"]["total"], dtype=float)
        pdmo = np.asarray(packet["pk_ref"]["pk_dmo"], dtype=float)

        if np.any(pdmo <= 0.0) or not np.all(np.isfinite(pdmo)):
            raise ValueError(
                "DMO halo-model power is non-positive or non-finite;"
                " cannot form boost.")

        return np.asarray(pb / pdmo, dtype=float)
