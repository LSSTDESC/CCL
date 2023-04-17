from ... import ccllib as lib
from .parameters_base import CCLParameters


__all__ = ("SplineParams",)


class SplineParams(CCLParameters, instance=lib.cvar.user_spline_params):
    """Instances of this class hold the spline parameters."""
