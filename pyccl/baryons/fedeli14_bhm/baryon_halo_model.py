"""Baryon halo model by Fedeli (2014)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pyccl as ccl

from pyccl.baryons.fedeli14_bhm.mass_fractions import mass_fractions
from pyccl.baryons.fedeli14_bhm.halo_profiles import (
    GasHaloProfile,
    StellarHaloProfile,
    nfw_profile
)
from .profile_interpolation import build_profile_interpolators
from .power_spectra import FedeliPkCalculator
from .numerics import (
    _require_a,
    _require_k,
)


__all__ = ("BaryonHaloModel",)


class BaryonHaloModel:
    r"""Fedeli (2014) baryon halo model wrapper.

    This class provides a high-level interface for computing halo-model power
    spectra for a three-component matter field (dark matter, gas, stars). It
    connects:

    - a mass-fraction prescription that defines component fractions as
      functions of halo mass and scale factor,
    - halo density profiles for each component,
    - Fourier-space profile interpolation for efficient evaluation of
      :math:`y_X(M,k) = u_X(k|M)/M`,
    - and a halo-model power spectrum calculator.

    The main outputs are the baryonic halo-model total matter power spectrum
    :math:`P_{\rm bar}^{\rm HM}(k,a)` and a boost factor relative to a chosen
    reference spectrum:

    .. math::
        B(k,a) = \frac{P_{\rm bar}^{\rm HM}(k,a)}{P_{\rm ref}(k,a)}.

    Notes
    -----
    - Mass fractions, profiles, and Fourier interpolators are cached per scale
      factor ``a``.
    - The default boost reference ``pk_ref="pk_dmo"`` is a halo-model DMO
      baseline constructed consistently with the same machinery.
    - This class also outputs the individual halo-model components via
      ``pk_components`.
    """
    def __init__(
        self,
        *,
        cosmo: Any,
        # halo model plumbing
        mass_def: Any | None = None,
        mass_function: Any | None = None,
        halo_bias: Any | None = None,
        # DM profile knobs (optional)
        concentration: Any | None = None,
        # (1) MASS FRACTIONS
        rho_star: float | None = None,
        m0_star: float = 5.0e12,
        sigma_star: float = 1.2,
        mmin_star: float = 1.0e10,
        mmax_star: float = 1.0e15,
        m0_gas: float = 1.0e12,
        sigma_gas: float = 3.0,
        # (2) PROFILE SHAPES
        gas_beta: float = 2.9,
        gas_r_co: float = 0.1,
        gas_r_ej: float = 4.5,
        gas_x_min: float = 1.0e-3,
        gas_x_max: float = 50.0,
        gas_n_x: int = 500,
        star_x_delta: float = 1.0 / 0.03,
        star_alpha: float = 1.0,
        # (3) DIFFUSE GAS MIXING
        Fg: float = 0.9,
        bd: float = 0.6,
        # override profiles (optional)
        profiles: dict[str, Any] | None = None,
        # numerical knobs
        mass_ranges: dict[str, dict[str, float]] | None = None,
        interpolation_grid: dict[str, dict[str, np.ndarray]] | None = None,
        update_fftlog_precision: bool = True,
        fftlog_kwargs: dict[str, Any] | None = None,
        rgi_kwargs: dict[str, Any] | None = None,
        n_m: int = 512,
        density_mmin: float = 1e6,
        density_mmax: float = 1e16,
    ):
        r"""Initialize the Fedeli14 baryon halo model wrapper.

        Sets the model plumbing (mass definition, mass function, halo bias),
        the physical parameters for mass fractions and profiles, diffuse-gas
        mixing parameters, and numerical grids for interpolation and integrals.

        Args:
            cosmo: Cosmology used for densities and reference power spectra.
            mass_def: Halo mass definition. Defaults to
                ``MassDef(200, "matter")``.
            mass_function: Halo mass function (CCL convention).
                Defaults to Tinker08.
            halo_bias: Halo bias (CCL convention). Defaults to Tinker10.
            concentration: Optional concentration-mass relation for
                the NFW DM profile.
            rho_star: Optional stellar density normalization used by the
                mass-fraction model.
            m0_star: Stellar mass-fraction parameter.
            sigma_star: Stellar mass-fraction parameter.
            mmin_star: Minimum mass used by the stellar mass-fraction model.
            mmax_star: Maximum mass used by the stellar mass-fraction model.
            m0_gas: Gas mass-fraction parameter.
            sigma_gas: Gas mass-fraction parameter.
            gas_beta: Gas profile shape parameter.
            gas_r_co: Gas profile shape parameter.
            gas_r_ej: Gas profile shape parameter.
            gas_x_min: Minimum radius (in units of halo radius) used to
                tabulate gas profiles.
            gas_x_max: Maximum radius (in units of halo radius) used to
                tabulate gas profiles.
            gas_n_x: Number of radial samples used to tabulate gas profiles.
            star_x_delta: Scale parameter for the stellar profile.
            star_alpha: Slope parameter for the stellar profile.
            Fg: Fraction of gas in the halo-tracing component.
                Must satisfy 0 <= Fg <= 1.
            bd: Bias-like coupling for the diffuse gas contribution.
                Must be finite.
            profiles: Optional mapping component -> profile object.
                If provided, profiles
                are not constructed internally.
            mass_ranges: Component mass ranges for halo-model integrals,
                mapping component -> {"min": ..., "max": ...}.
            interpolation_grid: Interpolation grids for Fourier profiles,
                mapping component -> {"mass": mass_grid, "k": k_grid}.
            update_fftlog_precision: If True, update FFTLog precision on
                profiles when supported.
            fftlog_kwargs: Keyword arguments forwarded to FFTLog precision
                updates.
            rgi_kwargs: Keyword arguments forwarded to the interpolator
                builder.
            n_m: Number of mass samples used in halo-model integrals.
            density_mmin: Minimum mass used to integrate mass fractions into
                mean densities.
            density_mmax: Maximum mass used to integrate mass fractions into
                mean densities.

        Raises:
            ValueError: If Fg is outside [0, 1], if bd is non-finite, or if
                the density integration limits do not satisfy
                0 < density_mmin < density_mmax.
        """
        self.cosmo = cosmo

        self.mass_def = mass_def or ccl.halos.MassDef(200, "matter")
        self.mass_function = mass_function or ccl.halos.MassFuncTinker08(
            mass_def=self.mass_def, mass_def_strict=False
        )
        self.halo_bias = halo_bias or ccl.halos.HaloBiasTinker10(
            mass_def=self.mass_def, mass_def_strict=False
        )

        self.concentration = concentration

        # store physical params
        self._mass_frac_params = {
            "rho_star": rho_star,
            "m0_star": float(m0_star),
            "sigma_star": float(sigma_star),
            "mmin_star": float(mmin_star),
            "mmax_star": float(mmax_star),
            "m0_gas": float(m0_gas),
            "sigma_gas": float(sigma_gas),
        }
        self._gas_profile_params = {
            "beta": float(gas_beta),
            "r_co": float(gas_r_co),
            "r_ej": float(gas_r_ej),
            "x_min": float(gas_x_min),
            "x_max": float(gas_x_max),
            "n_x": int(gas_n_x),
        }
        self._star_profile_params = {
            "x_delta": float(star_x_delta),
            "alpha": float(star_alpha),
        }

        self.Fg = float(Fg)
        self.bd = float(bd)

        # numerical knobs
        self.update_fftlog_precision = bool(update_fftlog_precision)
        self.fftlog_kwargs = dict(fftlog_kwargs or {})
        self.rgi_kwargs = dict(rgi_kwargs or {})
        self.n_m = int(n_m)
        self.density_mmin = float(density_mmin)
        self.density_mmax = float(density_mmax)

        if not (0.0 <= self.Fg <= 1.0):
            raise ValueError(f"Fg must be in [0, 1], got {self.Fg}.")
        if not np.isfinite(self.bd):
            raise ValueError("bd must be finite.")
        if (
                self.density_mmin <= 0
                or self.density_mmax <= 0
                or self.density_mmax <= self.density_mmin
        ):
            raise ValueError(
                "density_mmin/density_mmax must satisfy 0 < min < max."
            )

        if mass_ranges is None:
            mass_ranges = {
                "dark_matter": {"min": 1e6, "max": 1e16},
                "gas": {"min": 1e6, "max": 1e16},
                "stars": {"min": 1e6, "max": 1e16},
            }
        self.mass_ranges = mass_ranges

        if interpolation_grid is None:
            mass_grid = np.logspace(6, 16, 128)
            k_grid = np.logspace(-4, 2, 256)
            interpolation_grid = {
                "dark_matter": {"mass": mass_grid, "k": k_grid},
                "gas": {"mass": mass_grid, "k": k_grid},
                "stars": {"mass": mass_grid, "k": k_grid},
            }
        self.interpolation_grid = interpolation_grid
        self._validate_ranges_vs_interp()

        self._profiles_override = profiles
        self._dmo_dm_interp_cache: dict[float, Any] = {}

        # caches (per a)
        self._frac_cache: dict[float, tuple[Any, Any, Any]] = {}
        self._profile_cache: dict[float, dict[str, Any]] = {}
        self._interp_cache: dict[float, dict[str, Any]] = {}

    def _validate_ranges_vs_interp(self) -> None:
        """Validate that halo-model mass ranges are contained in interpolation
           grids.

        Raises:
            ValueError: If any component mass range extends beyond the
                interpolation mass grid for that component.
        """
        for comp in ("dark_matter", "gas", "stars",):
            r = self.mass_ranges[comp]
            g = self.interpolation_grid[comp]
            mmin, mmax = float(r["min"]), float(r["max"])
            mgrid = np.asarray(g["mass"], dtype=float)
            if mmin < mgrid.min() or mmax > mgrid.max():
                raise ValueError(
                    f"{comp}: mass_ranges [{mmin:.3e}, {mmax:.3e}] exceed "
                    f"interpolation mass grid [{mgrid.min():.3e},"
                    f" {mgrid.max():.3e}]."
                )

    def clear_cache(self) -> None:
        """Clear cached mass fractions, profiles, and Fourier interpolators.

        This forces regeneration of mass fractions, profile objects, and
        interpolation objects on the next request at a given scale factor.
        """
        self._frac_cache.clear()
        self._profile_cache.clear()
        self._interp_cache.clear()
        self._dmo_dm_interp_cache.clear()

    def _mass_frac_kwargs(self) -> dict[str, Any]:
        """Return mass-fraction model parameters that are explicitly set.

        Only parameters with non-None values are forwarded to the mass-fraction
        model constructor.
        """
        out = {}
        for k, v in self._mass_frac_params.items():
            if v is not None:
                out[k] = v
        return out

    def _get_mass_fractions(self, a: float):
        """Return component mass-fraction callables at scale factor a.

        Returns callables (f_gas, f_star, f_dm) that map halo mass M to the
        corresponding mass fraction at the given scale factor.
        Results are cached per a.

        Args:
            a: Scale factor.

        Returns:
            Tuple of callables (f_gas, f_star, f_dm), each callable maps f(M)
            to fraction.
        """
        a = float(a)
        hit = self._frac_cache.get(a)
        if hit is not None:
            return hit

        f_gas, f_star, f_dm = mass_fractions(
            cosmo=self.cosmo,
            a=a,
            mass_function=self.mass_function,
            **self._mass_frac_kwargs(),
        )
        self._frac_cache[a] = (f_gas, f_star, f_dm)
        return f_gas, f_star, f_dm

    def _get_profiles(self, a: float) -> dict[str, Any]:
        """Return halo profile objects for dark matter, gas, and stars at scale
           factor a.

        If profile overrides were provided, they are returned directly.
        Otherwise, profiles are constructed from the current mass-fraction
        callables and cached per a.

        Args:
            a: Scale factor.

        Returns:
            Dict mapping component name -> profile object.
        """
        a = float(a)
        if self._profiles_override is not None:
            return self._profiles_override

        hit = self._profile_cache.get(a)
        if hit is not None:
            return hit

        f_gas, f_star, _ = self._get_mass_fractions(a)

        prof_g = GasHaloProfile(
            mass_def=self.mass_def, f_gas=f_gas, **self._gas_profile_params
        )
        prof_s = StellarHaloProfile(
            mass_def=self.mass_def, f_star=f_star, **self._star_profile_params
        )
        prof_dm = nfw_profile(
            mass_def=self.mass_def, concentration=self.concentration)

        out = {
            "dark_matter": prof_dm,
            "gas": prof_g,
            "stars": prof_s,
        }
        self._profile_cache[a] = out
        return out

    def _get_interpolators(self, a: float) -> dict[str, Any]:
        r"""Return Fourier-space profile interpolators at scale factor a.

        The returned mapping provides callables for each component that
        evaluate y(M,k) = u(k|M)/M on demand. Interpolators are cached per a.

        Args:
            a: Scale factor.

        Returns:
            Dict mapping component name to interpolator callable y(M, k).
        """
        a = float(a)
        hit = self._interp_cache.get(a)
        if hit is not None:
            return hit

        profiles = self._get_profiles(a)
        interps = build_profile_interpolators(
            cosmo=self.cosmo,
            a=a,
            interpolation_grid=self.interpolation_grid,
            profiles=profiles,
            update_fftlog_precision=self.update_fftlog_precision,
            fftlog_kwargs=self.fftlog_kwargs,
            rgi_kwargs=self.rgi_kwargs,
        )
        self._interp_cache[a] = interps
        return interps

    def _get_dmo_dm_interpolator(self, a: float) -> Any:
        """Return the baseline DMO dark-matter Fourier interpolator at scale
           factor a.

        This provides the reference dark-matter interpolator used for the DMO
        halo-model baseline, independent of any baryonic modification to the
        dark-matter profile. Cached per a.

        Args:
            a: Scale factor.

        Returns:
            Callable or interpolator object for the baseline DM y(M, k).
        """
        a = float(a)
        hit = self._dmo_dm_interp_cache.get(a)
        if hit is not None:
            return hit

        prof_dm_dmo = nfw_profile(mass_def=self.mass_def,
                                  concentration=self.concentration)

        # build only the DM interpolator on the DM grid
        interps = build_profile_interpolators(
            cosmo=self.cosmo,
            a=a,
            interpolation_grid={
                "dark_matter": self.interpolation_grid["dark_matter"]},
            profiles={"dark_matter": prof_dm_dmo},
            components=("dark_matter",),
            update_fftlog_precision=self.update_fftlog_precision,
            fftlog_kwargs=self.fftlog_kwargs,
            rgi_kwargs=self.rgi_kwargs,
        )

        self._dmo_dm_interp_cache[a] = interps["dark_matter"]
        return interps["dark_matter"]

    def pk_components(self, *, k: np.ndarray, a: float) -> dict[str, Any]:
        """Compute halo-model spectra components on an input (k, a) grid.

        Returns a nested packet containing:
          - the evaluation grid,
          - reference spectra (linear, nonlinear, and DMO halo-model baseline),
          - component and pair power spectra,
          - weights and metadata used to assemble the total matter spectrum.

        Args:
            k: Wavenumber grid in units of Mpc^{-1}.
            a: Scale factor.

        Returns:
            Nested dict ("packet") with spectra arrays and metadata on the
            input grid.
        """
        a = _require_a(a)
        k = _require_k(k)

        interpolators = self._get_interpolators(a)
        f_gas, f_star, _ = self._get_mass_fractions(a)
        dmo_dm = self._get_dmo_dm_interpolator(a)

        calc = FedeliPkCalculator(
            cosmo=self.cosmo,
            a=a,
            k=k,
            profiles_u_over_m=interpolators,
            mass_function=self.mass_function,
            dmo_dm_u_over_m=dmo_dm,
            halo_bias=self.halo_bias,
            mass_ranges=self.mass_ranges,
            densities={},  # filled below
            gas_params={"Fg": self.Fg, "bd": self.bd},
            n_m=self.n_m,
        )
        calc.ensure_densities(
            f_gas=f_gas,
            f_star=f_star,
            mmin=self.density_mmin,
            mmax=self.density_mmax,
        )

        # Packet schema: grid / pk_ref / pk / pairs / meta
        return calc.pk_packet()

    def pk_total(self, *, k: np.ndarray, a: float) -> np.ndarray:
        r"""Return the total baryonic halo-model matter power spectrum.

        This returns P_bar^HM(k,a).

        Args:
            k: Wavenumber grid in units of Mpc^{-1}.
            a: Scale factor.

        Returns:
            Total baryonic halo-model matter power spectrum evaluated on k.
        """
        pk_tot = np.asarray(self.pk_components(k=k, a=a)["pk"]["total"],
                            dtype=float)
        return pk_tot

    def boost(self, *, k: np.ndarray, a: float,
              pk_ref: str = "pk_dmo") -> np.ndarray:
        r"""Return the baryonic boost factor relative to a reference spectrum.

        The boost is defined as:

        .. math::
            B(k,a) = \frac{P_{\rm bar}^{\rm HM}(k,a)}{P_{\rm ref}(k,a)}.

        Args:
            k: Wavenumber grid in units of Mpc^{-1}.
            a: Scale factor.
            pk_ref: Reference spectrum key provided by the packet.
                Options are: "pk_dmo", "pk_nlin", "pk_lin".

        Returns:
            Boost factor evaluated on k.

        Raises:
            ValueError: If pk_ref is not available in the reference spectra
            block.
        """
        a = _require_a(a)
        k = _require_k(k)

        out = self.pk_components(k=k, a=a)
        pk_bar = np.asarray(out["pk"]["total"], dtype=float)

        pkref_block = out.get("pk_ref", {})
        if pk_ref not in pkref_block:
            raise ValueError(
                f"pk_ref must be one of {tuple(pkref_block.keys())},"
                f" got {pk_ref!r}.")
        pk_ref_arr = np.asarray(pkref_block[pk_ref], dtype=float)

        return pk_bar / pk_ref_arr

    def pk_total_dmo(self, *, k: np.ndarray, a: float) -> np.ndarray:
        r"""Return the DMO halo-model baseline spectrum.

        "DMO" here denotes a halo-model baseline constructed with the same
        machinery, with all matter assigned to the collisionless (dark matter)
        profile. The output corresponds to P_dmo^HM(k,a).

        Args:
            k: Wavenumber grid in units of Mpc^{-1}.
            a: Scale factor.

        Returns:
            DMO halo-model baseline matter power spectrum evaluated on k.
        """
        a = _require_a(a)
        k = _require_k(k)

        interpolators = self._get_interpolators(a)
        dmo_dm = self._get_dmo_dm_interpolator(a)

        rho_m = float(ccl.rho_x(self.cosmo, a, "matter"))
        calc = FedeliPkCalculator(
            cosmo=self.cosmo,
            a=a,
            k=k,
            profiles_u_over_m=interpolators,
            dmo_dm_u_over_m=dmo_dm,
            mass_function=self.mass_function,
            halo_bias=self.halo_bias,
            mass_ranges=self.mass_ranges,
            densities={"matter": rho_m},
            gas_params={"Fg": self.Fg, "bd": self.bd},
            n_m=self.n_m,
        )
        return calc.pk_total_dmo()

    def halo_profiles(
        self,
        *,
        a: float,
        space: str = "fourier"
    ) -> dict[str, Any]:
        """Return component halo profiles at scale factor a.

        Args:
            a: Scale factor.
            space: Profile representation to return:
                - "real": profile objects
                  (e.g. NFW / GasHaloProfile / StellarHaloProfile)
                - "fourier": Fourier-space interpolators callable as y(M, k)

        Returns:
            Dict mapping component name -> profile object (space="real") or
            component name -> interpolator callable (space="fourier").

        Raises:
            ValueError: If space is not "real" or "fourier".
        """
        a = _require_a(a)
        if space == "real":
            return self._get_profiles(a)
        if space == "fourier":
            return self._get_interpolators(a)
        raise ValueError("space must be 'real' or 'fourier'.")

    def mass_fractions(self, *, a: float):
        """Return Fedeli14 component mass-fraction callables at scale factor a.

        Returns three callables (f_gas, f_star, f_dm) mapping halo mass M to
        mass fractions. Results are cached per a.

        Args:
            a: Scale factor.

        Returns:
            Tuple (f_gas, f_star, f_dm) of callables f(M) mapping to fraction.
        """
        a = _require_a(a)
        return self._get_mass_fractions(a)

    def halo_radius(
        self,
        *,
        M: float | np.ndarray,
        a: float,
        frame: Literal["physical", "comoving"] = "physical",
    ) -> float | np.ndarray:
        r"""Return the spherical-overdensity halo radius for the configured
            mass definition.

        Args:
            M: Halo mass in the convention of the configured mass_def.
            a: Scale factor.
            frame: "physical" or "comoving". If "comoving", returns R_phys/a.

        Returns:
            Halo radius in Mpc (physical or comoving depending on frame).

        Raises:
            ValueError: If frame is not "physical" or "comoving".
        """
        a = _require_a(a)
        cosmo = self.cosmo
        # physical Mpc
        radius = self.mass_def.get_radius(cosmo, M, a)
        if frame == "comoving":
            radius = radius / a
        elif frame != "physical":
            raise ValueError("frame must be 'physical' or 'comoving'.")
        return radius

    def _get_interpolators_with_concentration(
        self,
        a: float,
        *,
        concentration_override: Any
    ) -> dict[str, Any]:
        """Return Fourier interpolators using a dark matter concentration
        override.

        This is intended for diagnostics and plotting. Only the dark-matter
        profile uses the overridden concentration; gas and stellar profiles
        follow the baseline model settings at the same scale factor.

        Args:
            a: Scale factor.
            concentration_override: Concentration-mass relation used for the DM
                profile.

        Returns:
            Dict mapping component name -> interpolator callable y(M, k).
        """
        a = float(a)

        # reuse gas/star fraction functions (same physics)
        f_gas, f_star, _ = self._get_mass_fractions(a)

        # build profiles with overridden DM concentration
        prof_dm = nfw_profile(mass_def=self.mass_def,
                              concentration=concentration_override)
        prof_g = GasHaloProfile(mass_def=self.mass_def, f_gas=f_gas,
                                **self._gas_profile_params)
        prof_s = StellarHaloProfile(mass_def=self.mass_def, f_star=f_star,
                                    **self._star_profile_params)
        profiles = {"dark_matter": prof_dm, "gas": prof_g, "stars": prof_s}

        # build interpolators
        # (do NOT write into the main cache, keep it “diagnostic-local”)
        interps = build_profile_interpolators(
            cosmo=self.cosmo,
            a=a,
            interpolation_grid=self.interpolation_grid,
            profiles=profiles,
            update_fftlog_precision=self.update_fftlog_precision,
            fftlog_kwargs=self.fftlog_kwargs,
            rgi_kwargs=self.rgi_kwargs,
        )
        return interps

    def pk_dmo_terms(
        self,
        *,
        k: np.ndarray,
        a: float,
        concentration_override: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return DMO halo-model terms (1-halo, 2-halo, total) with a
        concentration override.

        This computes the DMO baseline using the halo-model decomposition:

        .. math::
            P_{\rm dmo}^{\rm HM}(k,a) = P^{1h}(k,a) + P^{2h}(k,a).

        Args:
            k: Wavenumber grid in units of Mpc^{-1}.
            a: Scale factor.
            concentration_override: Concentration-mass relation used for the
                DM profile.

        Returns:
            Tuple (P_1h, P_2h, P_total) evaluated on k.
        """
        a = _require_a(a)
        k = _require_k(k)

        interps = self._get_interpolators_with_concentration(
            a, concentration_override=concentration_override
        )

        rho_m = float(ccl.rho_x(self.cosmo, a, "matter"))
        dmo_dm = interps["dark_matter"]

        calc = FedeliPkCalculator(
            cosmo=self.cosmo,
            a=a,
            k=k,
            profiles_u_over_m=interps,
            dmo_dm_u_over_m=dmo_dm,
            mass_function=self.mass_function,
            halo_bias=self.halo_bias,
            mass_ranges=self.mass_ranges,
            densities={"matter": rho_m},
            gas_params={"Fg": self.Fg, "bd": self.bd},
            n_m=self.n_m,
        )

        # DMO HM: use DM profile but normalize by rho_m
        P1, P2 = calc.pk_halo_pair(
            comp1="dark_matter",
            comp2="dark_matter",
            rho1=rho_m,
            rho2=rho_m,
        )
        return P1, P2, (P1 + P2)
