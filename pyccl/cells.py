__all__ = ("angular_cl",)

import warnings

import numpy as np

from . import DEFAULT_POWER_SPECTRUM, CCLWarning, check, lib, warn_api
from .pyutils import integ_types


@warn_api(pairs=[("cltracer1", "tracer1"), ("cltracer2", "tracer2")])
def angular_cl(cosmo, tracer1, tracer2, ell, *,
               p_of_k_a=DEFAULT_POWER_SPECTRUM,
               l_limber=-1., limber_integration_method='qag_quad'):
    """Calculate the angular (cross-)power spectrum for a pair of tracers.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        tracer1 (:class:`~pyccl.tracers.Tracer`): a Tracer object,
            of any kind.
        tracer2 (:class:`~pyccl.tracers.Tracer`): a second Tracer object.
        ell (float or array_like): Angular multipole(s) at which to evaluate
            the angular power spectrum.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or `None`): 3D Power
            spectrum to project. If a string, it must correspond to one of
            the non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).
        l_limber (float): Angular wavenumber beyond which Limber's
            approximation will be used. Defaults to -1.
        limber_integration_method (string): integration method to be used
            for the Limber integrals. Possibilities: ``'qag_quad'`` (GSL's
            `qag` method backed up by `quad` when it fails) and ``'spline'``
            (the integrand is splined and then integrated numerically).

    Returns:
        float or array_like: Angular (cross-)power spectrum values, \
            :math:`C_\\ell`, for the pair of tracers, as a function of \
            :math:`\\ell`.
    """
    if cosmo['Omega_k'] != 0:
        warnings.warn(
            "CCL does not properly use the hyperspherical Bessel functions "
            "when computing angular power spectra in non-flat cosmologies!",
            category=CCLWarning)
    if limber_integration_method not in integ_types:
        raise ValueError(
            f"Unknown integration method {limber_integration_method}.")

    # we need the distances for the integrals
    cosmo.compute_distances()

    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = cosmo_in.parse_pk2d(p_of_k_a, is_linear=False)

    # Create tracer colections
    status = 0
    clt1, status = lib.cl_tracer_collection_t_new(status)
    clt2, status = lib.cl_tracer_collection_t_new(status)
    for t in tracer1._trc:
        status = lib.add_cl_tracer_to_collection(clt1, t, status)
    for t in tracer2._trc:
        status = lib.add_cl_tracer_to_collection(clt2, t, status)

    ell_use = np.atleast_1d(ell)

    # Check the values of ell are monotonically increasing
    if not (np.diff(ell_use) > 0).all():
        raise ValueError("ell values must be monotonically increasing")

    # Return Cl values, according to whether ell is an array or not
    cl, status = lib.angular_cl_vec(
        cosmo, clt1, clt2, psp, l_limber,
        ell_use, integ_types[limber_integration_method],
        ell_use.size, status)
    if np.ndim(ell) == 0:
        cl = cl[0]

    # Free up tracer collections
    lib.cl_tracer_collection_t_free(clt1)
    lib.cl_tracer_collection_t_free(clt2)

    check(status, cosmo=cosmo_in)
    return cl
