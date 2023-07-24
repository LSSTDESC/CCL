__all__ = ("BaryonsvanDaalen19",)

import numpy as np

from .. import Pk2D
from . import Baryons


class BaryonsvanDaalen19(Baryons):
    """The "van Daalen+ 2019" model boost factor for baryons.

    .. note:: First presented in van Daalen et al., arXiv:1906.00968.

              The boost factor is applied multiplicatively so that
              :math:`P_{\\rm bar.}(k, a) = P_{\\rm DMO}(k, a)\\,
              f_{\\rm vD19}(k, a)`.

              Notice the model has only been tested at z=0 and is valid for
              :math:`k\\leq 1 \\,h/{\\rm Mpc}.

    Args:
        fbar500c (:obj:`float`): the fraction of baryons in a halo within
        an overdensity of 500 times the critical density, given in units
        of the ratio of :math:`\\Omega_b` to :math:`\\Omega_m`.
        Default to 0.7 which is approximately compatible with observations.
        See Figure 16 of the paper.
    """
    name = 'vanDaalen19'
    __repr_attrs__ = __eq_attrs__ = ("fbar500c",)

    def __init__(self, fbar500c=0.7):
        self.fbar500c = fbar500c

    def boost_factor(self, cosmo, k, a):
        """The vd19 model boost factor for baryons.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`): Cosmological parameters.
            k (:obj:`float` or `array`): Wavenumber (in :math:`{\\rm Mpc}^{-1}`).
            a (:obj:`float` or `array`): Scale factor.

        Returns:
            :obj:`float` or `array`: Correction factor to apply to
                the power spectrum.
        """ # noqa
        a_use, k_use = map(np.atleast_1d, [a, k])
        # [k]=1/Mpc [k_use]=h/Mpc
        a_use, k_use = a_use[:, None], k_use[None, :]/cosmo['h']

        avD19 = 2.215
        bvD19 = 0.1276
        cvD19 = 1.309
        dvD19 = -5.99
        evD19 = -0.5107
        tildefbar500c = self.fbar500c
        numf = (2**avD19 +
                2**bvD19*(cvD19*tildefbar500c)**(bvD19 -
                                                 avD19))
        expf = np.exp(dvD19*tildefbar500c+evD19)
        denf1 = (cvD19*tildefbar500c)**(bvD19-avD19)
        denf = k_use**(-avD19)+k_use**(-bvD19)*denf1
        fka = 1-numf/denf*expf

        if np.ndim(k) == 0:
            fka = np.squeeze(fka, axis=-1)
        if np.ndim(a) == 0:
            fka = np.squeeze(fka, axis=0)
        return fka

    def update_parameters(self, fbar500c=None):
        """Update van Daalen 2019 parameters. All parameters set to
        ``None`` will be left untouched.

        Args:
            fbar500c (:obj:`float`): baryonic fraction in halos
                within 500 times the critical density.
        """
        if fbar500c is not None:
            self.fbar500c = fbar500c

    def _include_baryonic_effects(self, cosmo, pk):
        # Applies boost factor
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        k_arr = np.exp(lk_arr)
        fka = self.boost_factor(cosmo, k_arr, a_arr)
        pk_arr *= fka

        if pk.psp.is_log:
            np.log(pk_arr, out=pk_arr)  # in-place log

        return Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                    is_logp=pk.psp.is_log,
                    extrap_order_lok=pk.extrap_order_lok,
                    extrap_order_hik=pk.extrap_order_hik)
