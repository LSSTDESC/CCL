from .. import ccllib as lib
from ..base import CCLAutoreprObject, unlock_instance
from ..pyutils import check
from ..pk2d import Pk2D
from abc import abstractmethod


__all__ = ("Baryons",)


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
