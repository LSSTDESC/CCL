"""
=====================================================
2-point correlators (:mod:`pyccl.halos.profiles_2pt`)
=====================================================

Fourier-space 1-halo 2-point correlators for different halo profiles.
"""

from __future__ import annotations

__all__ = ("Profile2pt", "Profile2ptHOD", "Profile2ptCIB",)

from numbers import Real
from typing import TYPE_CHECKING, Optional, Union

from numpy.typing import NDArray

from .. import CCLObject, mass_def_api, warn_api  # update # TODO: CCLv3 uncomm
from . import HaloProfileHOD, HaloProfileCIB

if TYPE_CHECKING:
    from .. import Cosmology
    from . import HaloProfile


class Profile2pt(CCLObject):
    r"""Base for the 1-halo 2-point correlator between two halo profiles.

    .. math::

        \langle u_1(k) u_2(k) \rangle.

    Parameters
    ----------
    r_corr
        Scale the correlation by :math:`(1 + \rho_{u_1, u_2})`. Useful when the
        individual 1-halo terms are not fully correlated. Example usecases can
        be found in :footcite:t:`Koukoufilippas20` and :footcite:t:`Yan21`.
        The default is equivalent to the product of the Fourier halo profiles.

    References
    ----------
    .. footbibliography::

    Attributes
    ----------
    r_corr
    """
    __repr_attrs__ = __eq_attrs__ = ("r_corr",)

    @warn_api
    def __init__(self, *, r_corr: Real = 0):
        self.r_corr = r_corr

    @warn_api
    # @update(names=["r_corr"])  # TODO: Uncomment in CCLv3.
    # def update_parameters(self) -> None:
    def update_parameters(self, *, r_corr=None) -> None:
        """Update the parameters of the 2-point correlator."""
        if r_corr is not None:
            self.r_corr = r_corr

    @mass_def_api
    @warn_api
    def fourier_2pt(
            self,
            cosmo: Cosmology,
            k: Union[Real, NDArray[Real]],
            M: Union[Real, NDArray[Real]],
            a: Real,
            prof: HaloProfile,
            *,
            prof2: Optional[HaloProfile] = None
    ) -> Union[float, NDArray[float]]:
        r"""Compute the Fourier-space two-point moment between two profiles.

        .. math::

           (1 + \rho_{u_1,u_2}) \langle u_1(k) \rangle \langle u_2(k) \rangle

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        k : array_like (nk,)
            Comoving wavenumber, in units of :math:`\rm Mpc^{-1}`.
        M : array_like (nM,)
            Halo mass in units of :math:`\rm M_{\odot}`.
        a
            Scale factor.
        prof
            First halo profile.
        prof2
            Second halo profile. If None, `prof` is used.

        Returns
        -------
        array_like (nM, nk)
            Second-order Fourier-space moment.
        """
        if prof2 is None:
            prof2 = prof

        uk1 = prof.fourier(cosmo, k, M, a)

        if prof == prof2:
            uk2 = uk1
        else:
            uk2 = prof2.fourier(cosmo, k, M, a)

        return uk1 * uk2 * (1 + self.r_corr)


class Profile2ptHOD(Profile2pt):
    r"""Autocorrelation of HOD profiles.

    .. math::

       \langle n_g^2(k) | M,a \rangle = \bar{N}_c(M,a)
       \left[ 2f_c(a) \, \bar{N}_s(M,a) \, u_{\rm sat}(r | M,a)
             + ( \bar{N}_s(M,a) \, u_{\rm sat}(r | M,a))^2 \right],

    where all quantities are described in the documentation of
    :class:`~HaloProfileHOD`.
    """

    @mass_def_api
    @warn_api
    def fourier_2pt(
            self,
            cosmo: Cosmology,
            k: Union[Real, NDArray[Real]],
            M: Union[Real, NDArray[Real]],
            a: Real,
            prof: HaloProfileHOD,
            *,
            prof2: Optional[HaloProfileHOD] = None
    ) -> Union[float, NDArray[float]]:
        r"""Compute the Fourier-space two-point moment for the HOD profile.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        k : array_like (nk,)
            Comoving wavenumber, in units of :math:`\rm Mpc^{-1}`.
        M : array_like (nM,)
            Halo mass in units of :math:`\rm M_{\odot}`.
        a
            Scale factor.
        prof
            First halo profile.
        prof2
            Second halo profile. If None, `prof` is used.

        Returns
        -------
        array_like (nM, nk)
            Second-order Fourier-space moment.

        Raises
        ------
        ValueError
            If the profiles are not equal and HOD.
        """
        if prof2 is None:
            prof2 = prof

        if prof != prof2:
            raise ValueError("prof and prof2 must be equivalent")
        if not isinstance(prof, HaloProfileHOD):
            raise TypeError("prof and prof2 must be HaloProfileHOD")

        return prof._fourier_variance(cosmo, k, M, a)


class Profile2ptCIB(Profile2pt):
    """Fourier-space 1-halo 2-point correlator for the CIB profile.

    Follows closely the implementation of the HOD 2-point correlator
    (:class:`~Profile2ptHOD`) and Equation 15 of :footcite:t:`McCarthy21`.

    References
    ----------
    .. footbibliography::
    """

    @mass_def_api
    @warn_api
    def fourier_2pt(
            self,
            cosmo: Cosmology,
            k: Union[Real, NDArray[Real]],
            M: Union[Real, NDArray[Real]],
            a: Real,
            prof: HaloProfileCIB,
            *,
            prof2: Optional[HaloProfileCIB] = None
    ) -> Union[float, NDArray[float]]:
        r"""Compute the Fourier-space two-point moment for the HOD profile.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        k : array_like (nk,)
            Comoving wavenumber, in units of :math:`\rm Mpc^{-1}`.
        M : array_like (nM,)
            Halo mass in units of :math:`\rm M_{\odot}`.
        a
            Scale factor.
        prof
            First halo profile.
        prof2
            Second halo profile. If None, `prof` is used.

        Returns
        -------
        array_like (nM, nk)
            Second-order Fourier-space moment.

        Raises
        ------
        ValueError
            If any profile is not CIB.
        """
        if prof2 is None:
            prof2 = prof

        CIB = HaloProfileCIB
        if not (isinstance(prof, CIB) and isinstance(prof2, CIB)):
            raise TypeError("prof and prof2 must be HaloProfileCIB")

        return prof._fourier_variance(cosmo, k, M, a, nu_other=prof2.nu)
