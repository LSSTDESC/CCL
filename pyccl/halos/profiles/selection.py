__all__ = ("HaloProfileSelection",)

import numpy as np

from . import HaloProfile


class HaloProfileSelection(HaloProfile):
    """ Point-like profile describing the number of selected halos
    (e.g. clusters) as a function of halo mass and scale factor.

    .. math::
       \\langle N(k)|M,a\\rangle = S(M,a)\\,\\left[1-f_{\\rm off}+
       f_{\\rm off}\\,e^{-k^2\\sigma_{\\rm off}^2(M,a)/2}\\right],

    where :math:`S(M,a)` is the probability that a halo of mass
    :math:`M` at scale factor :math:`a` is selected. Mass-observable
    scatter can be included by integrating the observable-mass relation
    over the observable bin in :math:`S`. The term in brackets accounts
    for miscentering: a fraction :math:`f_{\\rm off}` of the selected halos
    has its centre displaced from the halo centre in the plane of the sky,
    following a 2D Gaussian (i.e. a Rayleigh distribution in radius) with
    dispersion :math:`\\sigma_{\\rm off}=r_{\\rm off}\\,R_\\Delta(M)`, where
    :math:`R_\\Delta(M)` is the comoving halo radius (see e.g.
    `Johnston et al. 2007 <https://arxiv.org/abs/0709.1159>`_).

    The profile is normalized by the mean number density of selected halos,
    :math:`\\bar{n}(a)=\\int dM\\,n(M,a)\\,S(M,a)`.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        selection (:obj:`callable`): selection function with signature
            ``selection(M, a)``, returning an array with the shape of ``M``.
        f_off (:obj:`float`): fraction of miscentered halos. Default 0.
        r_off (:obj:`float`): dispersion of the miscentering offsets in
            units of the comoving halo radius. Default 0.
        is_number_counts (:obj:`bool`): set to ``True`` if this profile
            represents number counts normalized by their mean within the
            survey window. Default ``True``.
    """
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "selection", "f_off",
                                     "r_off", "precision_fftlog",)

    def __init__(self, *, mass_def, selection, f_off=0., r_off=0.,
                 is_number_counts=True):
        self.selection = selection
        self.f_off = f_off
        self.r_off = r_off
        super().__init__(mass_def=mass_def,
                         is_number_counts=is_number_counts)

    def get_normalization(self, cosmo, a, *, hmc):
        """Returns the normalization of this profile, which is the
        mean number density of selected halos.

        .. note:: This integral includes the contribution assigned by
                  :class:`~pyccl.halos.halo_model.HMCalculator` to halos
                  below its minimum mass, and therefore only agrees with
                  :meth:`~pyccl.halos.halo_model.HMCalculator.number_counts`
                  if the selection vanishes at that mass.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology
                object.
            a (:obj:`float`): scale factor.
            hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo
                model calculator object.

        Returns:
            :obj:`float`: normalization factor of this profile.
        """
        return hmc.integrate_over_massfunc(lambda M: self._S(M, a), cosmo, a)

    def _S(self, M, a):
        return np.reshape(self.selection(M, a), np.shape(M))

    def _fourier_offset(self, cosmo, k, M, a):
        # Fourier transform of the distribution of centres, shape (N_M, N_k)
        if self.f_off == 0:
            return np.ones([len(M), len(k)])
        R_M = self.mass_def.get_radius(cosmo, M, a) / a
        sigma = self.r_off * R_M
        return 1 - self.f_off + self.f_off * np.exp(
            -0.5 * (k[None, :] * sigma[:, None])**2)

    def _fourier(self, cosmo, k, M, a):
        M_use = np.atleast_1d(M)
        k_use = np.atleast_1d(k)

        prof = (self._S(M_use, a)[:, None] *
                self._fourier_offset(cosmo, k_use, M_use, a))

        if np.ndim(k) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof
