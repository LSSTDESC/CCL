__all__ = ("Baryons",)

from abc import abstractmethod

from .. import CCLAutoRepr, CCLNamedClass


class Baryons(CCLAutoRepr, CCLNamedClass):
    """:class:`Baryons` objects are used to include the effects of
    baryons on the non-linear matter power spectrum. Their main ingredient
    is a method :meth:`include_baryonic_effects` that takes in a
    :class:`~pyccl.pk2d.Pk2D` and returns another :class:`~pyccl.pk2d.Pk2D`
    object that now accounts for baryonic effects (according to the model
    implemented in the corresponding :class:`Baryons` object).
    """

    @abstractmethod
    def _include_baryonic_effects(self, cosmo, pk):
        """Apply baryonic effects to a given power spectrum.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`):
                Cosmological parameters.
            pk (:class:`~pyccl.pk2d.Pk2D`): power spectrum.

        Returns:
            :obj:`~pyccl.pk2d.Pk2D` object.
        """

    def include_baryonic_effects(self, cosmo, pk):
        """Apply baryonic effects to a given power spectrum.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`):
                Cosmological parameters.
            pk (:class:`~pyccl.pk2d.Pk2D`): power spectrum.

        Returns:
            :obj:`~pyccl.pk2d.Pk2D` or :obj:`None`.
        """
        return self._include_baryonic_effects(cosmo, pk)
