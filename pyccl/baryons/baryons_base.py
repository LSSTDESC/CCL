__all__ = ("Baryons",)

from abc import abstractmethod

from .. import CCLAutoRepr, CCLNamedClass


class Baryons(CCLAutoRepr, CCLNamedClass):
    r"""Base class for models that add the imprint of baryons to a power
    spectrum (usually the non-linear matter power).

    Implementation
    --------------
    Subclasses must define a :meth:`_include_baryonic_effects` which implements
    the specific baryon model by operating on a :obj:`~pyccl.pk2d.Pk2D`
    power spectrum.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = "include_baryonic_effects"
        if (func := getattr(cls, name, False)) and not func.__doc__:
            # Inject docstring to subclasses if they don't have one.
            func.__doc__ = Baryons.include_baryonic_effects.__doc__

    @abstractmethod
    def _include_baryonic_effects(self, cosmo, pk):
        ...

    def include_baryonic_effects(self, cosmo, pk):
        """Apply baryonic effects to a given power spectrum.

        Arguments
        ---------
        cosmo : :class:`~pyccl.core.Cosmology`
            Cosmological parameters.
        pk : :class:`~pyccl.pk2d.Pk2D`
            Power spectrum.

        Returns
        -------
        pk_bar : :obj:`~pyccl.pk2d.Pk2D` object
            Power spectrum that includes baryonic effects.
        """
        return self._include_baryonic_effects(cosmo, pk)

    _include_baryonic_effects.__doc__ = include_baryonic_effects.__doc__
