from ... import ccllib as lib
from .parameters_base import CCLParameters


__all__ = ("PhysicalConstants",)


class PhysicalConstants(CCLParameters, instance=lib.cvar.constants,
                        freeze=True):
    """Instances of this class hold the physical constants."""
