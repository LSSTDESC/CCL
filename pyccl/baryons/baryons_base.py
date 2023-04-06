from ..base import CCLAutoreprObject
from abc import abstractmethod


__all__ = ("Baryons",)


class Baryons(CCLAutoreprObject):
    """`BaryonicEffect` obects are used to include the imprint of baryons
    on the non-linear matter power spectrum. Their main ingredient is a
    method `include_baryonic_effects` that takes in a `ccl.Pk2D` and
    returns another `ccl.Pk2D` object that now accounts for baryonic
    effects.
    """

    @abstractmethod
    def _include_baryonic_effects(self, cosmo, pk):
        """Apply baryonic effects to a given power spectrum.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
            pk (:class:`~pyccl.pk2d.Pk2D`): power spectrum.

        Returns:
            :obj:`~pyccl.pk2d.Pk2D` object.
        """

    def include_baryonic_effects(self, cosmo, pk):
        """Apply baryonic effects to a given power spectrum.

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): Cosmological parameters.
            pk (:class:`~pyccl.pk2d.Pk2D`): power spectrum.

        Returns:
            :obj:`~pyccl.pk2d.Pk2D` object or `None`.
        """
        return self._include_baryonic_effects(cosmo, pk)

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
