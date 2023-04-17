from ... import ccllib as lib
from .parameters_base import CCLParameters


__all__ = ("GSLParams",)


class GSLParams(CCLParameters, instance=lib.cvar.user_gsl_params):
    """Instances of this class hold the gsl parameters."""
