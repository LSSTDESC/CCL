__all__ = ("SplineParams", "spline_params",)

from numbers import Real

from ... import ccllib as lib
from . import Parameters


class SplineParams(Parameters, instance=lib.cvar.user_spline_params):
    r"""Instances of this class hold the spline parameters.

    Descriptions of the parameters start with a key indicating which splines
    the parameter is used in:

        * SF - Scale factor splines.
        * PS - Power spectrum splines.
        * SM - :math:`\sigma(M)` splines.
        * LMB - Limber integration.
        * CORR - Correlation functions.
        * SPLT - Spline types (all fixed to `Akima
          <https://en.wikipedia.org/wiki/Akima_spline>`_).
    """
    A_SPLINE_MINLOG: Real = 0.000_1
    """(SF) Minimum scale factor."""

    A_SPLINE_MIN: Real = 0.1
    """(SF) Transition scale factor between logarithmically- and linearly-\
    spaced spline points.
    """

    A_SPLINE_MAX: Real = 1.0
    """(SF) Maximum scale factor."""

    A_SPLINE_NLOG: int = 250
    """(SF) Number of logarithmically-spaced bins between `A_SPLINE_MINLOG`
    and `A_SPLINE_MIN.`
    """

    A_SPLINE_NA: int = 250
    """(SF) Number of linearly-spaced bins between `A_SPLINE_MIN` and
    `A_SPLINE_MAX`.
    """

    A_SPLINE_MINLOG_PK: Real = 0.01
    """(PS) Minimum scale factor."""

    A_SPLINE_MIN_PK: Real = 0.1
    """(PS) Transition scale factor between logarithmically- and linearly-\
    spaced spline points.
    """

    K_MIN: Real = 5e-5
    r"""(PS) Minimum wavenumber (:math:`\rm Mpc^{-1}`) of analytic models."""

    K_MAX: Real = 1e3
    r"""(PS) Maximum wavenumber (:math:`\rm Mpc^{-1}`) of analytic models."""

    K_MAX_SPLINE: Real = 50
    r"""(PS) Maximum wavenumber (:math:`\rm Mpc^{-1}`) of numerical models."""

    A_SPLINE_NA_PK: int = 40
    """(PS) Number of linearly-spaced bins between `A_SPLINE_MIN_PK` and
    `A_SPLINE_MAX`.
    """

    A_SPLINE_NLOG_PK: int = 11
    """(PS) Number of logarithmically-spaced bins between `A_SPLINE_MINLOG_PK`
    and `A_SPLINE_MIN_PK`.
    """

    N_K: int = 167
    """(PS) Number of spline nodes per dex."""

    A_SPLINE_MINLOG_SM: Real = 0.01
    """(SM) Minimum scale factor."""

    A_SPLINE_MIN_SM: Real = 0.1
    """(SM) Transition scale factor between logarithmically- and linearly-\
    spaced spline points.
    """

    A_SPLINE_NLOG_SM: int = 6
    """(SM) Number of logarithmically-spaced bins between `A_SPLINE_MINLOG_SM`
    and `A_SPLINE_MIN_SM`.
    """

    A_SPLINE_NA_SM: int = 13
    """(SM) Number of linearly-spaced bins between `A_SPLINE_MIN_SM` and
    `A_SPLINE_MAX`
    """

    LOGM_SPLINE_MIN: Real = 6
    """(SM) Base-10 logarithm of the minimum halo mass."""

    LOGM_SPLINE_MAX: Real = 17
    """(SM) Base-10 logarithm of the maximum halo mass."""

    LOGM_SPLINE_NM: int = 50
    """(SM) Number of logarithmically-spaced values in mass."""

    LOGM_SPLINE_DELTA: Real = 0.025
    """(SM) Step in base-10 logarithmic units for computing of the finite
    difference derivatives.
    """

    DLOGK_INTEGRATION: Real = 0.025
    """(LMB) Step in base-10 logarithmic units of the angular wavenumber."""

    DCHI_INTEGRATION: Real = 5
    """(LMB) Step in base-10 logarithmic units of the comoving distance."""

    N_K_3DCOR: int = 100_000  # TODO: Remove in CCLv3.
    """(CORR) Number of spline points in wavenumber per dex.

    .. deprecated:: 2.8.0

        Use `N_K_3DCORR` instead.
    """

    N_K_3DCORR = N_K_3DCOR
    """(CORR) Number of spline points in wavenumber per dex."""

    ELL_MIN_CORR: Real = 0.01
    """(CORR) Minimum angular wavenumber."""

    ELL_MAX_CORR: Real = 60_000
    """(CORR) Maximum angular wavenumber."""

    N_ELL_CORR: int = 5_000
    """(CORR) Number of logarithmically-spaced bins in angular wavenumber
    between `ELL_MIN_CORR` and `ELL_MAX_CORR`.
    """

    A_SPLINE_TYPE: None = None
    """(SPLT) Scale factor spline interpolation type."""

    K_SPLINE_TYPE: None = None
    """(SPLT) Power spectrum spline interpolation type."""

    M_SPLINE_TYPE: None = None
    r"""(SPLT) :math:`\sigma(M)` spline interpolation type."""

    PLIN_SPLINE_TYPE: None = None
    """(SPLT) Linear power spectrum spline interpolation type."""

    PNL_SPLINE_TYPE: None = None
    """(SPLT) Non-linear power spectrum spline interpolation type."""

    CORR_SPLINE_TYPE: None = None
    """(SPLT) Correlation spline interpolation type."""

    def __setattr__(self, name, value):
        if name == "A_SPLINE_MAX" and value != 1.0:
            raise ValueError("A_SPLINE_MAX is fixed to 1.")
        if "SPLINE_TYPE" in name and value is not None:
            raise ValueError("Spline types are fixed constants.")
        super().__setattr__(name, value)


spline_params = SplineParams()
