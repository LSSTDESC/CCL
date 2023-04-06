from ... import ccllib as lib
from .parameters_base import Parameters


__all__ = ("SplineParams",)


class SplineParams(Parameters, instance=lib.cvar.user_spline_params):
    """Instances of this class hold the spline parameters."""

    # Scale factor spline parameters
    A_SPLINE_NA = 250
    A_SPLINE_MIN = 0.1
    A_SPLINE_MINLOG_PK = 0.01
    A_SPLINE_MIN_PK = 0.1
    A_SPLINE_MINLOG_SM = 0.1
    A_SPLINE_MAX = 1.0
    A_SPLINE_MINLOG = 0.000_1
    A_SPLINE_NLOG = 250

    # Mass splines
    LOGM_SPLINE_DELTA = 0.025
    LOGM_SPLINE_NM = 50
    LOGM_SPLINE_MIN = 6
    LOGM_SPLINE_MAX = 17

    # Power spectrum a- and k-splines
    A_SPLINE_NA_SM = 13
    A_SPLINE_NLOG_SM = 6
    A_SPLINE_NA_PK = 40
    A_SPLINE_NLOG_PK = 11

    # k-splines and integrals
    K_MAX_SPLINE = 50
    K_MAX = 1e3
    K_MIN = 5e-5
    DLOGK_INTEGRATION = 0.025
    DCHI_INTEGRATION = 5.
    N_K = 167
    N_K_3DCORR = 100_000

    # Correlation function parameters
    ELL_MIN_CORR = 0.01
    ELL_MAX_CORR = 60_000
    N_ELL_CORR = 5_000

    def __setattr__(self, key, value):
        if (key, value) == ("A_SPLINE_MAX", 1.0):
            return  # Setting `A_SPLINE_MAX` to its default value; do nothing.
        super().__setattr__(key, value)
