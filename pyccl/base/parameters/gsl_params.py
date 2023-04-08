from ... import ccllib as lib
from .parameters_base import Parameters


__all__ = ("GSLParams",)


class GSLParams(Parameters, instance=lib.cvar.user_gsl_params):
    """Instances of this class hold the GSL parameters."""

    # Key for the number of Gauss-Kronrod points used in QAG integration.
    # https://www.gnu.org/software/gsl/doc/html/integration.html
    GSL_INTEG_GAUSS41 = 4

    # Default relative precision.
    GSL_EPSREL = 1e-4
    # Default number of iterations for integration and root-finding.
    GSL_N_ITERATION = 1_000

    N_ITERATION = GSL_N_ITERATION
    # Default number of Gauss-Kronrod points in QAG integration.
    GSL_INTEGRATION_GAUSS_KRONROD_POINTS = GSL_INTEG_GAUSS41
    # Relative precision in sigma_R calculations.
    GSL_EPSREL_SIGMAR = 1e-5
    # Relative precision in k_NL calculations.
    GSL_EPSREL_KNL = 1e-5
    # Relative precision in distance calculations.
    GSL_EPSREL_DIST = 1e-6
    # Relative precision in growth calculations.
    GSL_EPSREL_GROWTH = 1e-6
    # Relative precision in dNdz calculations.
    GSL_EPSREL_DNDZ = 1e-6

    # General parameters.
    INTEGRATION_LIMBER_EPSREL = GSL_N_ITERATION

    # Integration.
    INTEGRATION_GAUSS_KRONROD_POINTS = GSL_INTEGRATION_GAUSS_KRONROD_POINTS
    INTEGRATION_EPSREL = GSL_EPSREL
    # Limber integration.
    INTEGRATION_LIMBER_GAUSS_KRONROD_POINTS = \
        GSL_INTEGRATION_GAUSS_KRONROD_POINTS
    INTEGRATION_LIMBER_EPSREL = GSL_EPSREL
    # Distance integrals.
    INTEGRATION_DISTANCE_EPSREL = GSL_EPSREL_DIST
    # sigma_R integral.
    INTEGRATION_SIGMAR_EPSREL = GSL_EPSREL_SIGMAR
    # kNL integral.
    INTEGRATION_KNL_EPSREL = GSL_EPSREL_KNL

    # Root finding.
    ROOT_EPSREL = GSL_EPSREL
    ROOT_N_ITERATION = GSL_N_ITERATION

    # ODE.
    ODE_GROWTH_EPSREL = GSL_EPSREL_GROWTH
    # Growth.
    EPS_SCALEFAC_GROWTH = 1e-6

    # Halo model.
    HM_MMIN = 1e7
    HM_MMAX = 1e17
    HM_EPSABS = 0.
    HM_EPSREL = 1e-4
    HM_LIMIT = 1_000
    HM_INT_METHOD = GSL_INTEG_GAUSS41

    # Flags for spline integration.
    NZ_NORM_SPLINE_INTEGRATION = True
    LENSING_KERNEL_SPLINE_INTEGRATION = True
