__all__ = ("GSLParams", "gsl_params",)

from numbers import Real

from ... import ccllib as lib
from . import Parameters


class GSLParams(Parameters, instance=lib.cvar.user_gsl_params):
    """Instances of this class hold the GSL parameters."""
    GSL_INTEG_GAUSS41: int = 4
    """Key for the number of Gauss-Kronrod points used in QAG integration.
    `<https://www.gnu.org/software/gsl/doc/html/integration.html>`_
    """

    GSL_EPSREL: None = 1e-4
    """Default relative precision."""

    GSL_N_ITERATION: int = 1_000
    """Default number of iterations for integration and root-finding."""

    N_ITERATION: int = GSL_N_ITERATION
    """General number of iterations for integration and root-finding."""

    GSL_INTEGRATION_GAUSS_KRONROD_POINTS: int = GSL_INTEG_GAUSS41
    """Key for the default number of Gauss-Kronrod points in QAG
    integration.
    """

    GSL_EPSREL_SIGMAR: Real = 1e-5
    """Relative precision in sigma_R calculations."""

    GSL_EPSREL_KNL: Real = 1e-5
    """Relative precision in k_NL calculations."""

    GSL_EPSREL_DIST: Real = 1e-6
    """Relative precision in distance calculations."""

    GSL_EPSREL_GROWTH: Real = 1e-6
    """Relative precision in growth calculations."""

    GSL_EPSREL_DNDZ: Real = 1e-6
    """Relative precision in dNdz calculations."""

    INTEGRATION_LIMBER_EPSREL: int = GSL_EPSREL
    """Relative precision of Limber integration."""

    INTEGRATION_GAUSS_KRONROD_POINTS: int = \
        GSL_INTEGRATION_GAUSS_KRONROD_POINTS
    """Key for the number of Gauss-Kronrod points in QAG general
    integration.
    """

    INTEGRATION_EPSREL: Real = GSL_EPSREL
    """Relative precision of general integration."""

    INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS: int = \
        GSL_INTEGRATION_GAUSS_KRONROD_POINTS
    """Key for the number of Gauss-Kronrod points in QAG Limber integration."""

    INTEGRATION_LIMBER_EPSREL: Real = GSL_EPSREL
    """Relative precision of Limber integration."""

    INTEGRATION_DISTANCE_EPSREL: Real = GSL_EPSREL_DIST
    """Relative precision of distance integrals."""

    INTEGRATION_SIGMAR_EPSREL: Real = GSL_EPSREL_SIGMAR
    r"""Relative precision of :math:`\sigma(R)` integrals."""

    INTEGRATION_KNL_EPSREL: Real = GSL_EPSREL_KNL
    r"""Relative precision of :math:`k_{\rm NL}` integrals."""

    ROOT_EPSREL: Real = GSL_EPSREL
    """Relative precision of root-finding."""

    ROOT_N_ITERATION: int = GSL_N_ITERATION
    """Maximum number of iterations in root-finding."""

    ODE_GROWTH_EPSREL: Real = GSL_EPSREL_GROWTH
    """Relative precision of ODEs."""

    EPS_SCALEFAC_GROWTH: Real = 1e-6
    """Scale factor precision in the growth ODEs."""

    # TODO: Remove the HM params in CCLv3.
    HM_MMIN: Real = 1e7
    """Minimum mass for the halo model.

    .. deprecated:: 2.8.0

        This parameter is not used and will be removed in the next major
        relaese.
    """

    HM_MMAX: Real = 1e17
    """Maximum mass for the halo model.

    .. deprecated:: 2.8.0

        This parameter is not used and will be removed in the next major
        relaese.
    """

    HM_EPSABS: Real = 0
    """Absolute precision of halo model integrations.

    .. deprecated:: 2.8.0

        This parameter is not used and will be removed in the next major
        relaese.
    """

    HM_EPSREL: Real = 1e-4
    """Relative precision of halo model integrations.

    .. deprecated:: 2.8.0

        This parameter is not used and will be removed in the next major
        relaese.
    """

    HM_LIMIT: int = 1_000
    """Size of the GSL workspace for halo model integrations.

    .. deprecated:: 2.8.0

        This parameter is not used and will be removed in the next major
        relaese.
    """

    HM_INT_METHOD: int = GSL_INTEG_GAUSS41
    """Key for the number of Gauss-Kronrod points in QAG halo model
    integration.

    .. deprecated:: 2.8.0

        This parameter is not used and will be removed in the next major
        relaese.
    """

    NZ_NORM_SPLINE_INTEGRATION: bool = True
    r"""Whether to use spline integration in the normalization of :math:`n(z)`
    in tracers.
    """

    LENSING_KERNEL_SPLINE_INTEGRATION: bool = True
    """Whether to integrate the lensing kernel using spline integration."""


gsl_params = GSLParams()
