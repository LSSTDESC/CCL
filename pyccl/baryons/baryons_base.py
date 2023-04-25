"""
========================================
Baryons (:mod:`pyccl.base.baryons_base`)
========================================

Abstract base class for models that add baryonic effects to power spectra.
"""

from __future__ import annotations

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
    * Subclasses inherit from ``CCLNamedClass``. Refer to the docs for further
      information.
    * Subclasses must define :meth:`_include_baryonic_effects` implementing
      the specific baryon model by operating on a :obj:`~pyccl.pk2d.Pk2D`
      power spectrum. The signature must be the same as
      :meth:`include_baryonic_effects`.
    """

    @abstractmethod
    def _include_baryonic_effects(self, cosmo, pk):
        ...

    def include_baryonic_effects(
            self,
            cosmo: Cosmology,
            pk: Pk2D
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
