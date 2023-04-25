"""
=====================================================
2-point correlators (:mod:`pyccl.halos.profiles_2pt`)
=====================================================

Fourier-space 1-halo 2-point correlators for different halo profiles.
"""

__all__ = ("Profile2pt", "Profile2ptHOD", "Profile2ptCIB",)

from .. import CCLObject, mass_def_api, warn_api
from . import HaloProfileHOD, HaloProfileCIBShang12


class Profile2pt(CCLObject):
    r"""Base for the 1-halo 2-point correlator between two halo profiles.

    .. math::

        \langle u_1(k) u_2(k) \rangle.

    Parameters
    ----------
    r_corr : int or float
        Scale the correlation by :math:`(1 + \rho_{u_1, u_2})`. Useful when the
        individual 1-halo terms are not fully correlated. Example usecases can
        be found in `Koukoufilippas et al. (2020)
        <https://arxiv.org/abs/1909.09102>`_ and `Yan et al. (2021)
        <https://arxiv.org/abs/2102.07701>`_. The default is :math:`0`, which
        is equivalent to the product of the Fourier halo profiles.
    """
    __repr_attrs__ = __eq_attrs__ = ("r_corr",)

    @warn_api
    def __init__(self, *, r_corr=0):
        self.r_corr = r_corr

    @warn_api
    def update_parameters(self, *, r_corr=None):
        """Update the parameters of the 2-point correlator."""
        if r_corr is not None:
            self.r_corr = r_corr

    @mass_def_api
    @warn_api
    def fourier_2pt(self, cosmo, k, M, a, prof, *, prof2=None):
        r"""Compute the Fourier-space two-point moment between two profiles.

        .. math::

           (1 + \rho_{u_1,u_2}) \langle u_1(k) \rangle \langle u_2(k) \rangle

        Arguments
        ---------
        cosmo : :obj:`~pyccl.Cosmology`
            Cosmological parameters.
        k : int, float or (nk,) array_like
            Comoving wavenumber, in units of :math:`\rm Mpc^{-1}`.
        M : int, float or (nM,) array_like
            Halo mass in units of :math:`\rm M_{\odot}`.
        a : int or float
            Scale factor.
        prof, prof2 : :obj:`~pyccl.halos.HaloProfile`, required, optional
            Halo profiles to correlate. If ``prof2`` is not provided, ``prof``
            is auto-correlated.

        Returns
        -------
        p2pt : float or (nM, nk) numpy.ndarray
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
    :class:`~pyccl.halos.HaloProfileHOD`.
    """

    @mass_def_api
    @warn_api
    def fourier_2pt(self, cosmo, k, M, a, prof, *, prof2=None):
        r"""Compute the Fourier-space two-point moment for the HOD profile.

        Arguments
        ---------
        cosmo : :obj:`~pyccl.Cosmology`
            Cosmological parameters.
        k : int, float or (nk,) array_like
            Comoving wavenumber, in units of :math:`\rm Mpc^{-1}`.
        M : int, float or (nM,) array_like
            Halo mass in units of :math:`\rm M_{\odot}`.
        a : int or float
            Scale factor.
        prof, prof2 : :obj:`~pyccl.halos.HaloProfileHOD`, required, optional
            Halo profiles to correlate. If ``prof2`` is provided it must be the
            equal to ``prof``.

        Returns
        -------
        p2pt : float or (nM, nk) numpy.ndarray
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

    It follows closely the implementation of the HOD 2-point correlator
    (:class:`~pyccl.halos.Profile2ptHOD`) and Equation 15 of `McCarthy &
    Madhavacheril  <https://arxiv.org/abs/2010.16405>`_.
    """

    @mass_def_api
    @warn_api
    def fourier_2pt(self, cosmo, k, M, a, prof, *, prof2=None):
        r"""Compute the Fourier-space two-point moment for the HOD profile.

        Arguments
        ---------
        cosmo : :obj:`~pyccl.Cosmology`
            Cosmological parameters.
        k : int, float or (nk,) array_like
            Comoving wavenumber, in units of :math:`\rm Mpc^{-1}`.
        M : int, float or (nM,) array_like
            Halo mass in units of :math:`\rm M_{\odot}`.
        a : int or float
            Scale factor.
        prof, prof2 : :obj:`~pyccl.halos.HaloProfileCIB`, required, optional
            Halo profiles to correlate. If ``prof2`` is not provided, ``prof``
            is auto-correlated.

        Returns
        -------
        p2pt : float or (nM, nk) numpy.ndarray
            Second-order Fourier-space moment.

        Raises
        ------
        ValueError
            If any profile is not CIB.
        """
        if prof2 is None:
            prof2 = prof

        Shang12 = HaloProfileCIBShang12
        if not (isinstance(prof, Shang12) and isinstance(prof2, Shang12)):
            raise TypeError("prof and prof2 must be HaloProfileCIB")

        return prof._fourier_variance(cosmo, k, M, a, nu_other=prof2.nu)
