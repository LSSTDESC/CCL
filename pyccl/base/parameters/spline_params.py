__all__ = ("SplineParams", "spline_params",)

from ... import ccllib as lib
from . import Parameters


class SplineParams(Parameters, instance=lib.cvar.user_spline_params):
    """Instances of this class hold the spline parameters."""

    # Scale factor spline parameters
    A_SPLINE_NA = 250
    A_SPLINE_MIN = 0.1
    A_SPLINE_MINLOG_PK = 0.01
    A_SPLINE_MIN_PK = 0.1
    A_SPLINE_MINLOG_SM = 0.01
    A_SPLINE_MIN_SM = 0.1
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
    DCHI_INTEGRATION = 5
    N_K = 167
    N_K_3DCOR = 100_000

    # Correlation function parameters
    ELL_MIN_CORR = 0.01
    ELL_MAX_CORR = 60_000
    N_ELL_CORR = 5_000

    # Interpolation types
    A_SPLINE_TYPE = None
    K_SPLINE_TYPE = None
    M_SPLINE_TYPE = None
    PNL_SPLINE_TYPE = None
    PLIN_SPLINE_TYPE = None
    CORR_SPLINE_TYPE = None

    def __setattr__(self, key, value):
        if key == "A_SPLINE_MAX" and value != 1.0:
            raise ValueError("A_SPLINE_MAX is fixed to 1.")
        if "SPLINE_TYPE" in key and value is not None:
            raise ValueError("Spline types are fixed constants.")
        super().__setattr__(key, value)


spline_params = SplineParams()
