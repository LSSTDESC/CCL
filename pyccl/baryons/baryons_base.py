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
    """

    @abstractmethod
    def _include_baryonic_effects(self, cosmo: Cosmology, pk: Pk2D) -> Pk2D:
        """Model-specific implementation.
        See :meth:`Baryons.include_baryonic_effects`.

        :meta public:
        """

    def include_baryonic_effects(self, cosmo: Cosmology, pk: Pk2D) -> Pk2D:
        """Apply baryonic effects to a given power spectrum.

        Arguments
        ---------
        cosmo
            Cosmological parameters.
        pk
            Power spectrum.

        Returns
        -------

            Power spectrum that includes baryonic effects.
        """
        return self._include_baryonic_effects(cosmo, pk)

    include_baryonic_effects.__doc__ += _include_baryonic_effects.__doc__
