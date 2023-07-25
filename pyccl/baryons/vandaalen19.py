__all__ = ("BaryonsvanDaalen19",)

import numpy as np

from .. import Pk2D
from . import Baryons


class BaryonsvanDaalen19(Baryons):
    """The "van Daalen+ 2019" model boost factor for baryons.

    .. note:: First presented in 
              `van Daalen et al., <https://arxiv.org/abs/1906.00968>`_. See the
              `DESC Note <https://github.com/LSSTDESC/CCL/blob/master/doc\
/0000-ccl_note/main.pdf>`_
              for details.

              The boost factor is applied multiplicatively so that
              :math:`P_{\\rm bar.}(k, a) = P_{\\rm DMO}(k, a)\\,
              f_{\\rm BCM}(k, a)`.

              Notice the model has only been tested at z=0 and is valid for
              :math:`k\\leq 1 \\,h/{\\rm Mpc}`.

    Args:
        fbar (:obj:`float`): the fraction of baryons in a halo within
            an overdensity of X times the critical density, given in units
            of the ratio of :math:`\\Omega_b` to :math:`\\Omega_m`.
            Default to 0.7 which is approximately compatible with observations.
            See Figure 16 of the paper.
        mass_def (:obj:`string`): whether the mass definition corresponds to
            X=500 or 200 critical. Options are "500c" or "200c".

    """
    name = 'vanDaalen19'
    __repr_attrs__ = __eq_attrs__ = ("fbar","mass_def",)

    def __init__(self, fbar=0.7,mass_def='500c'):
        self.fbar = fbar
        self.mass_def = mass_def 

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

        if(self.mass_def=='500c'):
            avD19 = 2.215
            bvD19 = 0.1276
            cvD19 = 1.309
            dvD19 = -5.99
            evD19 = -0.5107
        else:
            avD19 = 2.111
            bvD19 = 0.0038
            cvD19 = 1.371
            dvD19 = -5.816
            evD19 = -0.4005
            
        numf = (2**avD19 +
                    2**bvD19*(cvD19*self.fbar)**(bvD19 -avD19))
        expf = np.exp(dvD19*self.fbar+evD19)
        denf1 = (cvD19*self.fbar)**(bvD19-avD19)
        denf = k_use**(-avD19)+k_use**(-bvD19)*denf1
        fka = 1-numf/denf*expf
            
        if np.ndim(k) == 0:
            fka = np.squeeze(fka, axis=-1)
        if np.ndim(a) == 0:
            fka = np.squeeze(fka, axis=0)
        return fka

    def update_parameters(self, fbar=None, mass_def=None):
        """Update van Daalen 2019 parameters. All parameters set to
        ``None`` will be left untouched.

        Args:
            fbar (:obj:`float`): baryonic fraction in halos
                within X times the critical density.

            mass_def (:obj:`string`): mass definition: whether 500 ("500c") 
                or 200 critical ("200c").

        """
        if fbar is not None:
            self.fbar = fbar
        if mass_def is not None:
            self.mass_def = mass_def

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
