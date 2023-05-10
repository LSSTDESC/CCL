"""
==========================================
Halo model (:mod:`pyccl.halos.halo_model`)
==========================================

Functionality to compute halo model integrals.
"""

from __future__ import annotations

__all__ = ("HMCalculator",)

import warnings
from numbers import Real
from typing import Callable, Literal, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .. import CCLObject, CCLDeprecationWarning, unlock_instance
from .. import warn_api, deprecate_attr, deprecated
from .. import physical_constants as const
from . import HaloBias, HaloProfile, MassDef, MassFunc, Profile2pt
from ..cosmology import Cosmology
from ..pyutils import _spline_integrate


class HMCalculator(CCLObject):
    r"""Implementation of methods used to compute quantities related to the
    halo model. A lot of these quantities involve integrals of the sort:

    .. math::

        I^0_X &= \int {\rm d}M \, n(M, a) \, f(M, k, a), \\
        I^1_X &= \int {\rm d}M \, b(M, a) \, n(M, a) \, f(M, k, a),

    where :math:`n(M, a)` is the halo mass function, :math:`b(M, a)` is the
    halo bias function, and :math:`f` is an arbitrary function of mass,
    scale factor and Fourier scales.

    In the integrals, `I_0_X` denotes that the integrand is multiplied by
    the mass function only, while `I_1_X` denotes multiplication by mass
    function and halo bias function. The number `X` denotes the number of
    halo profiles involved in the calculation.

    The method of integration, as well as its precision, can be fine-tuned.

    Parameters
    ----------
    mass_function
        Mass function used in the calculations. If an instantiated mass
        function is provided, its mass definition must be equal to the mass
        definition passed into the halo model calculator.
    halo_bias
        Halo bias used in the calculations. If an instantiated halo bias is
        provided, its mass definition must be equal to the mass definition
        passed into the halo model calculator.
    mass_def
        Mass definition used in the calculations. It must be equal to the mass
        definitions of `mass_function` and `halo_bias`. If strings are
        provided, the instantiated models will share a common mass definition.
        May be omitted if `mass_function` and `halo_bias` are provided
        instantiated; their internal mass definition is used.
    log10M_min, log10M_max
        Lower and upper mass integration bounds.
        These are the base-10 logarithms of mass in units of
        :math:`\rm M_\odot`.
    nM
        Number of uniformly-spaced samples in :math:`log_{10}M` used in the
        mass integrals.
    integration_method_M
        Integration method to use.
        `'simpson'` uses `scipy.integrate.simpson`, while `'spline'`
        integrates using the knots of a cubic spline fitted to the integrand.
    k_min
        Deprecated; do not use.
        Large-scale value used for normalization of the Fourier-space halo
        profiles, expressed in units of :math:`\rm Mpc^{-1}`.

    Raises
    ------
    ValueError
        If the mass integration method does not exist.
    ValueError
        In the methods accepting a :class:`~pyccl.halos.HaloProfile`, if the
        mass definition of the halo profile is inconsistent with the one of the
        halo model workspace.
    """
    __repr_attrs__ = __eq_attrs__ = (
        "mass_function", "halo_bias", "mass_def", "precision",)
    __getattr__ = deprecate_attr(pairs=[('_mdef', 'mass_def'),
                                        ('_massfunc', 'mass_function'),
                                        ('_hbias', 'halo_bias'),
                                        ('_prec', 'precision')]
                                 )(super.__getattribute__)
    mass_function: MassFunc
    halo_bias: HaloBias
    mass_def: MassDef
    precision: dict
    """Integration settings."""

    @warn_api(pairs=[("massfunc", "mass_function"), ("hbias", "halo_bias"),
                     ("nlog10M", "nM")])
    def __init__(
            self, *,
            mass_function: Union[MassFunc, str],
            halo_bias: Union[HaloBias, str],
            mass_def: Optional[Union[MassDef, str]] = None,
            log10M_min: Real = 8,
            log10M_max: Real = 16,
            nM: int = 128,
            integration_method_M: Literal["simpson", "spline"] = "simpson",
            k_min: Real = 1e-5
    ):
        # Initialize halo model ingredients.
        self.mass_def, *specs = MassDef.from_specs(
            mass_def, mass_function=mass_function, halo_bias=halo_bias)
        if len(specs) != 2:
            raise ValueError("Invalid mass function or halo bias.")
        self.mass_function, self.halo_bias = specs

        self.precision = {
            'log10M_min': log10M_min, 'log10M_max': log10M_max, 'nM': nM,
            'integration_method_M': integration_method_M, 'k_min': k_min}
        self._lmass = np.linspace(log10M_min, log10M_max, nM)
        self._mass = 10.**self._lmass
        self._m0 = self._mass[0]

        if integration_method_M == "simpson":
            from scipy.integrate import simpson
            self._integrator = simpson
        elif integration_method_M == "spline":
            self._integrator = self._integ_spline
        else:
            raise ValueError("Invalid integration method.")

        # Cache last results for mass function and halo bias.
        self._cosmo_mf = self._cosmo_bf = None
        self._a_mf = self._a_bf = -1

    def _fix_profile_mass_def(self, prof):
        # TODO v3: remove this (in v3 all profiles have a mass_def).
        # If profile has no mass definition assigned, assign one.
        if prof.mass_def is None:
            warnings.warn("In v3 all profiles will need an associated "
                          "mass definition.", CCLDeprecationWarning)
            prof.mass_def = self.mass_def

    def _integ_spline(self, fM, log10M):
        # Spline integrator
        return _spline_integrate(log10M, fM, log10M[0], log10M[-1])

    def _check_mass_def(self, *others):
        # Verify that internal & external mass definitions are consistent.
        if set([x.mass_def for x in others]) != set([self.mass_def]):
            raise ValueError("Inconsistent mass definitions.")

    @unlock_instance
    def _get_mass_function(self, cosmo, a, rho0):
        # Compute the mass function at this cosmo and a.
        if a != self._a_mf or cosmo != self._cosmo_mf:
            self._mf = self.mass_function(cosmo, self._mass, a)
            integ = self._integrator(self._mf*self._mass, self._lmass)
            self._mf0 = (rho0 - integ) / self._m0
            self._cosmo_mf, self._a_mf = cosmo, a  # cache

    @unlock_instance
    def _get_halo_bias(self, cosmo, a, rho0):
        # Compute the halo bias at this cosmo and a.
        if a != self._a_bf or cosmo != self._cosmo_bf:
            self._bf = self.halo_bias(cosmo, self._mass, a)
            integ = self._integrator(self._mf*self._bf*self._mass, self._lmass)
            self._mbf0 = (rho0 - integ) / self._m0
            self._cosmo_bf, self._a_bf = cosmo, a  # cache

    def _get_ingredients(self, cosmo, a, *, get_bf):
        # Compute mass function and halo bias at some scale factor.
        rho0 = const.RHO_CRITICAL * cosmo["Omega_m"] * cosmo["h"]**2
        self._get_mass_function(cosmo, a, rho0)
        if get_bf:
            self._get_halo_bias(cosmo, a, rho0)

    def _integrate_over_mf(self, array_2):
        #  ∫ dM n(M) f(M)
        i1 = self._integrator(self._mf * array_2, self._lmass)
        return i1 + self._mf0 * array_2[..., 0]

    def _integrate_over_mbf(self, array_2):
        #  ∫ dM n(M) b(M) f(M)
        i1 = self._integrator(self._mf * self._bf * array_2, self._lmass)
        return i1 + self._mbf0 * array_2[..., 0]

    def integrate_over_massfunc(
            self,
            func: Callable[[NDArray[Real]], NDArray[Real]],
            cosmo: Cosmology,
            a: Real
    ) -> float:
        r"""Compute the integral over mass of a function, times mass function.

        .. math::

            I = \int {\rm d}M \, n(M) \, f(M)

        Arguments
        ---------
        func
            Function of halo mass.
        cosmo
            Cosmological parameters.
        a
            Scale factor.

        Returns
        -------
        I : float
            Integral over mass times mass function.
        """
        fM = func(self._mass)
        self._get_ingredients(cosmo, a, get_bf=False)
        return self._integrate_over_mf(fM)

    @deprecated
    def profile_norm(
            self,
            cosmo: Cosmology,
            a: Real,
            prof: HaloProfile,
    ) -> float:
        r"""Compute the large-scale normalization of a profile:

        .. deprecated:: 2.8.0

            Use ``HaloProfile.get_normalization``.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        a
            Scale factor.
        prof
            Halo profile.

        Returns
        -------
        norm : float
            Profile normalization at the given scale factor.
        """
        self._fix_profile_mass_def(prof)
        self._check_mass_def(prof)
        self._get_ingredients(cosmo, a, get_bf=False)
        uk0 = prof.fourier(cosmo, self.precision['k_min'], self._mass, a).T
        return 1. / self._integrate_over_mf(uk0)

    @warn_api(pairs=[("sel", "selection"),
                     ("amin", "a_min"),
                     ("amax", "a_max")],
              reorder=["na", "a_min", "a_max"])
    def number_counts(
            self,
            cosmo: Cosmology,
            *,
            selection: Callable[
                [NDArray[float],
                 NDArray[float]],
                NDArray[float]],
            a_min: Optional[Real] = None,
            a_max: Optional[Real] = None,
            na: int = 128
    ) -> float:
        r"""Compute halo number counts:

        .. math::

            {\rm n_c}({\rm sel}) = \int {\rm d}M \int {\rm d}a \,
            \frac{{\rm d}V}{{\rm d}a \, {\rm d}\Omega} \,
            n(M, a) \, {\rm sel}(M, a),

        where :math:`n(M, a)` is the halo mass function, and
        :math:`\mathrm{sel}(M, a)` is the selection function.

        The selection function represents the selection probability
        per unit mass per unit scale factor and integrates to :math:`1`.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        selection
            Selection function with signature `sel(M, a)`. The function must
            be vectorized in both `M` and `a`, and the output shape must
            be `(na, nM)` as per `numpy` broadcasting rules.
        a_min
            Minimum scale factor used in the selection function integrals.
            The default uses `cosmo.cosmo.spline_params.A_SPLINE_MIN`.
        a_max
            Maximum scale factor used in the selection function integrals.
            The default uses `cosmo.cosmo.spline_params.A_SPLINE_MAX`.
        na
            Number of scale factor samples for the integrals. The default is
            calculated from the spline parameters stored in `cosmo`.

        Returns
        -------
        n_c : float
            The total number of clusters.
        """
        # get a values for integral
        if a_min is None:
            a_min = cosmo.cosmo.spline_params.A_SPLINE_MIN
        if a_max is None:
            a_max = cosmo.cosmo.spline_params.A_SPLINE_MAX
        a = np.linspace(a_min, a_max, na)

        # compute the volume element
        abs_dzda = 1 / a / a
        dc = cosmo.comoving_angular_distance(a)
        ez = cosmo.h_over_h0(a)
        dh = const.CLIGHT_HMPC / cosmo['h']
        dvdz = dh * dc**2 / ez
        dVda = dvdz * abs_dzda

        # now do m intergrals in a loop
        mint = np.zeros_like(a)
        for i, _a in enumerate(a):
            self._get_ingredients(cosmo, _a, get_bf=False)
            _selm = np.atleast_2d(selection(self._mass, _a)).T
            mint[i] = self._integrator(
                dVda[i] * self._mf[..., :] * _selm[..., :],
                self._lmass
            )

        # now do scale factor integral
        return self._integrator(mint, a)

    def I_0_1(
            self,
            cosmo: Cosmology,
            k: Union[float, NDArray[float]],
            a: Real,
            prof: HaloProfile
    ) -> Union[float, NDArray[float]]:
        r"""Compute the integral:

        .. math::

            I^0_1(k,a|u) = \int {\rm d}M \, n(M, a) \,
            \langle u(k, a|M) \rangle,

        where :math:`n(M, a)` is the halo mass function, and
        :math:`\langle u(k, a|M) \rangle` is the halo profile as a
        function of wavenumber, scale factor and halo mass.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        k
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a
            Scale factor.
        prof
            Halo profile.

        Returns
        -------
        I_0_1 : float or (nk,) numpy.ndarray
            Integral value.
        """
        self._fix_profile_mass_def(prof)
        self._check_mass_def(prof)
        self._get_ingredients(cosmo, a, get_bf=False)
        uk = prof.fourier(cosmo, k, self._mass, a).T
        return self._integrate_over_mf(uk)

    def I_1_1(
            self,
            cosmo: Cosmology,
            k: Union[float, NDArray[float]],
            a: Real,
            prof: HaloProfile
    ) -> Union[float, NDArray[float]]:
        r"""Compute the integral:

        .. math::

            I^1_1(k, a|u) = \int {\rm d}M \, n(M, a) \, b(M, a) \,
            \langle u(k, a|M) \rangle,

        where :math:`n(M, a)` is the halo mass function, :math:`b(M, a)` is
        the halo bias function, and :math:`\langle u(k, a|M) \rangle` is the
        halo profile as a function of wavenumber, scale factor and halo mass.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        k
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a
            Scale factor.
        prof
            Halo profile.

        Returns
        -------
        I_1_1 : float or (nk,) numpy.ndarray
            Integral value.
        """
        self._fix_profile_mass_def(prof)
        self._check_mass_def(prof)
        self._get_ingredients(cosmo, a, get_bf=True)
        uk = prof.fourier(cosmo, k, self._mass, a).T
        return self._integrate_over_mbf(uk)

    @warn_api(pairs=[("prof1", "prof")], reorder=["prof_2pt", "prof2"])
    def I_0_2(
            self,
            cosmo: Cosmology,
            k: Union[float, NDArray[float]],
            a: Real,
            prof: HaloProfile,
            *,
            prof2: Optional[HaloProfile] = None,
            prof_2pt: Profile2pt
    ) -> Union[float, NDArray[float]]:
        r"""Compute the integral:

        .. math::

            I^0_2(k, a | u,v) = \int \mathrm{d}M \, n(M, a) \,
            \langle u(k, a|M) v(k, a|M) \rangle,

        where :math:`n(M, a)` is the halo mass function, and
        :math:`\langle u(k,a|M) v(k,a|M)\rangle` is the two-point
        moment of the two halo profiles.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        k
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a
            Scale factor.
        prof, prof2
            Halo profiles. If `prof2 is None`, `prof` is used.
        prof_2pt
            Covariance of `prof` and `prof2`.

        Returns
        -------
        I_0_2 : float or (nk,) numpy.ndarray
            Integral value.
        """
        if prof2 is None:
            prof2 = prof
        self._fix_profile_mass_def(prof)
        self._fix_profile_mass_def(prof2)

        self._fix_profile_mass_def(prof)
        self._fix_profile_mass_def(prof2)
        self._check_mass_def(prof, prof2)
        self._get_ingredients(cosmo, a, get_bf=False)
        uk = prof_2pt.fourier_2pt(cosmo, k, self._mass, a, prof, prof2=prof2).T
        return self._integrate_over_mf(uk)

    @warn_api(pairs=[("prof1", "prof")], reorder=["prof_2pt", "prof2"])
    def I_1_2(
            self,
            cosmo: Cosmology,
            k: Union[float, NDArray[float]],
            a: Real,
            prof: HaloProfile,
            *,
            prof2: Optional[HaloProfile] = None,
            prof_2pt: Profile2pt
    ) -> Union[float, NDArray[float]]:
        r"""Compute the integral:

        .. math::

            I^1_2(k, a|u,v) = \int {\rm d}M \, n(M, a) \, b(M, a) \,
            \langle u(k, a|M) v(k, a|M) \rangle,

        where :math:`n(M, a)` is the halo mass function, :math:`b(M, a)` is
        the halo bias, and :math:`\langle u(k,a|M) v(k,a|M) \rangle` is the
        two-point moment of the two halo profiles.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        k
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a
            Scale factor.
        prof, prof2
            Halo profiles. If `prof2 is None`, `prof` is used.
        prof_2pt
            Covariance of `prof` and `prof2`.

        Returns
        -------
        I_1_2 : float or (nk,) numpy.ndarray
            Integral value.
        """
        if prof2 is None:
            prof2 = prof
        self._fix_profile_mass_def(prof)
        self._fix_profile_mass_def(prof2)

        self._fix_profile_mass_def(prof)
        self._fix_profile_mass_def(prof2)
        self._check_mass_def(prof, prof2)
        self._get_ingredients(cosmo, a, get_bf=True)
        uk = prof_2pt.fourier_2pt(cosmo, k, self._mass, a, prof, prof2=prof2).T
        return self._integrate_over_mbf(uk)

    @warn_api(pairs=[("prof1", "prof")],
              reorder=["prof12_2pt", "prof2", "prof3", "prof34_2pt", "prof4"])
    def I_0_22(
            self,
            cosmo: Cosmology,
            k: Union[float, NDArray[float]],
            a: Real,
            prof: HaloProfile,
            *,
            prof2: Optional[HaloProfile] = None,
            prof3: Optional[HaloProfile] = None,
            prof4: Optional[HaloProfile] = None,
            prof12_2pt: Profile2pt,
            prof34_2pt: Optional[Profile2pt] = None
    ) -> Union[float, NDArray[float]]:
        r"""Compute the integral:

        .. math::

            I^0_{2,2}(k_u, k_v, a|(u_{1,2},v_{1,2})) =
            \int {\rm d}M \, n(M, a) \,
            \langle u_1(k_u, a|M) u_2(k_u, a|M) \rangle \,
            \langle v_1(k_v, a|M) v_2(k_v, a|M) \rangle,

        where :math:`n(M,a)` is the halo mass function, and
        :math:`\langle u(k,a|M) v(k,a|M) \rangle` is the
        two-point moment of the two halo profiles.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        k
            Comoving wavenumber in :math:`\rm Mpc^{-1}`.
        a
            Scale factor.
        prof
            First halo profile.
        prof2
            Second halo profile. If `None`, `prof` is used.
        prof3
            Third halo profile. If `None`, `prof` is used.
        prof4
            Fourth halo profile. If `None`, `prof2` is used.
        prof12_2pt
            Covariance of `prof` and `prof2`.
        prof34_2pt
            Covariance of `prof3` and `prof4`
            If `None`, `prof12_2pt` is used.

        Returns
        -------
        I_0_22 : float or (nk, nk) numpy.ndarray
             Integral value.
        """
        if prof3 is None:
            prof3 = prof
        if prof4 is None:
            prof4 = prof2

        if prof34_2pt is None:
            prof34_2pt = prof12_2pt

        self._fix_profile_mass_def(prof)
        self._fix_profile_mass_def(prof2)
        self._fix_profile_mass_def(prof3)
        self._fix_profile_mass_def(prof4)
        self._check_mass_def(prof, prof2, prof3, prof4)
        self._get_ingredients(cosmo, a, get_bf=False)
        uk12 = prof12_2pt.fourier_2pt(
            cosmo, k, self._mass, a, prof, prof2=prof2).T

        if (prof, prof2, prof12_2pt) == (prof3, prof4, prof34_2pt):
            # 4pt approximation of the same profile
            uk34 = uk12
        else:
            uk34 = prof34_2pt.fourier_2pt(
                cosmo, k, self._mass, a, prof3, prof2=prof4).T

        return self._integrate_over_mf(uk12[None, :, :] * uk34[:, None, :])


# TODO: Remove for CCLv3.
def __getattr__(name):
    warn = lambda n, m: warnings.warn(  # noqa
        f"{n} is moved to pyccl.halos.{m}", CCLDeprecationWarning)
    if name in ["halomod_mean_profile_1pt", "halomod_bias_1pt"]:
        from .pk_1pt import __dict__ as mod_dict
        warn(name, "pk_1pt")
        return mod_dict[name]
    elif name in ["halomod_power_spectrum", "halomod_Pk2D"]:
        from .pk_2pt import __dict__ as mod_dict
        warn(name, "pk_2pt")
        return mod_dict[name]
    elif name in ["halomod_trispectrum_1h", "halomod_Tk3D_1h",
                  "halomod_Tk3D_SSC_linear_bias", "halomod_Tk3D_SSC"]:
        from .pk_4pt import __dict__ as mod_dict
        warn(name, "pk_4pt")
        return mod_dict[name]
    return eval(name) if name in locals() else None
