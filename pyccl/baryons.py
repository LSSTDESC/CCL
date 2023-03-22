from . import ccllib as lib
from .base import CCLAutoreprObject
from .pyutils import check
from .pk2d import Pk2D
from .base import unlock_instance
import numpy as np
from abc import abstractmethod


class Baryons(CCLAutoreprObject):
    """`BaryonicEffect` obects are used to include the imprint of baryons
    on the non-linear matter power spectrum. Their main ingredient is a
    method `include_baryonic_effects` that takes in a `ccl.Pk2D` and
    returns another `ccl.Pk2D` object that now accounts for baryonic
    effects.
    """
    name = 'base'

    @abstractmethod
    def _include_baryonic_effects(self, cosmo, pk, in_place=False):
        """Apply baryonic effects to a given power spectrum.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
            pk (:class:`~pyccl.pk2d.Pk2D`): power spectrum.
            in_place (:obj:`bool`): if True, `pk` itself is modified,
                instead of returning a new `ccl.Pk2D` object.

        Returns:
            :obj:`~pyccl.pk2d.Pk2D` object or `None` (if `in_place`
            is `True`.
        """

    def include_baryonic_effects(self, cosmo, pk, in_place=False):
        """Apply baryonic effects to a given power spectrum.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
            pk (:class:`~pyccl.pk2d.Pk2D`): power spectrum.
            in_place (:obj:`bool`): if True, `pk` itself is modified,
                instead of returning a new `ccl.Pk2D` object.

        Returns:
            :obj:`~pyccl.pk2d.Pk2D` object or `None` (if `in_place`
            is `True`.
        """
        return self._include_baryonic_effects(cosmo, pk,
                                              in_place=in_place)

    @unlock_instance(mutate=True, argv=2)
    def _new_pk(self, cosmo, pk, a_arr, lk_arr, pk_arr, logp, in_place):
        # Returns a new Pk2D, either in place or from scratch.
        if in_place:
            lib.f2d_t_free(pk.psp)
            status = 0
            pk.psp, status = lib.set_pk2d_new_from_arrays(
                lk_arr, a_arr, pk_arr.flatten(),
                int(pk.extrap_order_lok),
                int(pk.extrap_order_hik),
                int(logp), status)
            check(status, cosmo)
            return None
        else:
            new = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk_arr,
                       is_logp=logp,
                       extrap_order_lok=pk.extrap_order_lok,
                       extrap_order_hik=pk.extrap_order_hik)
            return new

    @classmethod
    def _subclasses(cls):
        # This helper returns a set of all subclasses
        return set(cls.__subclasses__()).union(
            [sub for cl in cls.__subclasses__() for sub in cl._subclasses()])

    @classmethod
    def from_name(cls, name):
        """
        Obtain `Baryons` subclass with name `name`.
        """
        models = {p.name: p for p in cls._subclasses()}
        return models[name]


class BaryonsSchneider15(Baryons):
    """The BCM model boost factor for baryons.

    .. note:: BCM stands for the "baryonic correction model" of Schneider &
              Teyssier (2015; https://arxiv.org/abs/1510.06034). See the
              `DESC Note <https://github.com/LSSTDESC/CCL/blob/master/doc\
/0000-ccl_note/main.pdf>`_
              for details.

    .. note:: The boost factor is applied multiplicatively so that
              :math:`P_{\\rm corrected}(k, a) = P(k, a)\\, f_{\\rm bcm}(k, a)`.

    Args:
        log10Mc (:obj:`float`): logarithmic mass scale of hot
            gas suppression. Defaults to log10(1.2E14).
        eta_b (:obj:`float`): ratio of escape to ejection radii (see
            Teyssier et al. 2015). Defaults to 0.5.
        k_s (:obj:`float`): Characteristic scale (wavenumber) of
            the stellar component. Defaults to 55.0.
    """
    name = 'Schneider15'
    __repr_attrs__ = ("log10Mc", "eta_b", "k_s")

    def __init__(self, log10Mc=np.log10(1.2E14), eta_b=0.5, k_s=55.0):
        self.log10Mc = log10Mc
        self.eta_b = eta_b
        self.k_s = k_s

    def boost_factor(self, cosmo, k, a):
        """The BCM model boost factor for baryons.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
            k (float or array_like): Wavenumber; Mpc^-1.
            a (float or array_like): Scale factor.

        Returns:
            float or array_like: Correction factor to apply to
                the power spectrum.
        """
        a_use, k_use = map(np.atleast_1d, [a, k])
        a_use, k_use = a_use[:, None], k_use[None, :]

        z = 1/a_use - 1
        kh = k_use / cosmo['h']
        b0 = 0.105*self.log10Mc - 1.27
        bfunc = b0 / (1. + (z/2.3)**2.5)
        kg = 0.7 * (1-bfunc)**4 * self.eta_b**(-1.6)
        gf = bfunc / (1 + (kh/kg)**3) + 1. - bfunc
        scomp = 1 + (kh / self.k_s)**2
        fka = gf * scomp

        if np.ndim(k) == 0:
            fka = np.squeeze(fka, axis=-1)
        if np.ndim(a) == 0:
            fka = np.squeeze(fka, axis=0)
        return fka

    def update_parameters(self, log10Mc=None, eta_b=None, k_s=None):
        """Update BCM parameters.

        Args:
            log10Mc (:obj:`float`): logarithmic mass scale of hot
                gas suppression. Defaults to 14.08.
            eta_b (:obj:`float`): ratio of escape to ejection radii (see
                Teyssier et al. 2015). Defaults to 0.5.
            k_s (:obj:`float`): Characteristic scale (wavenumber) of
                the stellar component. Defaults to 55.0.
        """
        if log10Mc is not None:
            self.log10Mc = log10Mc
        if eta_b is not None:
            self.eta_b = eta_b
        if k_s is not None:
            self.k_s = k_s

    @unlock_instance(mutate=True, argv=2)
    def _include_baryonic_effects(self, cosmo, pk, in_place=False):
        # Applies boost factor
        a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
        k_arr = np.exp(lk_arr)
        fka = self.boost_factor(cosmo, k_arr, a_arr)
        pk_arr *= fka

        if pk.psp.is_log:
            np.log(pk_arr, out=pk_arr)  # in-place log

        return self._new_pk(cosmo, pk, a_arr, lk_arr, pk_arr,
                            pk.psp.is_log, in_place)
