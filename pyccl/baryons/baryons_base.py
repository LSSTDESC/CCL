"""Abstract base class for models that add baryonic effects to power spectra.
"""
__all__ = ("Baryons",)

from abc import abstractmethod
from typing import TYPE_CHECKING

from .. import CCLNamedClass

if TYPE_CHECKING:
    from .. import Cosmology, Pk2D


class Baryons(CCLNamedClass):
    r"""Base class for models that add the imprint of baryons to a power
    spectrum (usually the non-linear matter power).

    Implementation
    --------------
    - Subclasses must define :meth:`_include_baryonic_effects` implementing
      the specific baryon model by operating on a :obj:`~pyccl.pk2d.Pk2D`
      power spectrum. The signature must be the same as
      :meth:`include_baryonic_effects`.
    - Subclasses must define a ``name`` class attribute, used to instantiate
      from the base class.
    - Subclasses may define a ``__repr_attrs__`` class attribute for automatic
      creation of :meth:`__repr__`.
    - Subclasses may include :meth:`update_parameters` to update the model
      parameters.
    """

    @abstractmethod
    def _include_baryonic_effects(self, cosmo, pk):
        ...

    def include_baryonic_effects(
            self,
            cosmo: "Cosmology",
            pk: "Pk2D"
    ) -> "Pk2D":
        """Apply baryonic effects to a given power spectrum.

        Arguments
        ---------
        cosmo : :obj:`~pyccl.Cosmology`
            Cosmological parameters.
        pk : :obj:`~pyccl.Pk2D`
            Power spectrum.

        Returns
        -------
        pk_bar : :obj:`~pyccl.Pk2D`
            Power spectrum that includes baryonic effects.
        """
        return self._include_baryonic_effects(cosmo, pk)

    _include_baryonic_effects.__doc__ = include_baryonic_effects.__doc__
