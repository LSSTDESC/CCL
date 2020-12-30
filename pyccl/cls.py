import warnings

import numpy as np

from .errors import CCLWarning
from . import ccllib as lib
from .pyutils import check, integ_types
from .pk2d import parse_pk2d

# Define symbolic 'None' type for arrays, to allow proper handling by swig
# wrapper
NoneArr = np.array([])


def angular_cl(cosmo, cltracer1, cltracer2, ell, p_of_k_a=None,
               l_limber=-1., limber_integration_method='qag_quad'):
    """Calculate the angular (cross-)power spectrum for a pair of tracers.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        cltracer1 (:class:`~pyccl.tracers.Tracer`): a `Tracer` object,
            of any kind.
        cltracer2 (:class:`~pyccl.tracers.Tracer`): a second `Tracer` object,
            of any kind.
        ell (float or array_like): Angular wavenumber(s) at which to evaluate
            the angular power spectrum.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to project. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`). If `None`, the non-linear matter
            power spectrum stored in `cosmo` will be used.
        l_limber (float) : Angular wavenumber beyond which Limber's
            approximation will be used. Defaults to -1.
        limber_integration_method (string) : integration method to be used
            for the Limber integrals. Possibilities: 'qag_quad' (GSL's `qag`
            method backed up by `quad` when it fails) and 'spline' (the
            integrand is splined and then integrated analytically).

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

    if limber_integration_method not in ['qag_quad', 'spline']:
        raise ValueError("Integration method %s not supported" %
                         limber_integration_method)

    # we need the distances for the integrals
    cosmo.compute_distances()

    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = parse_pk2d(cosmo_in, p_of_k_a)

    # Create tracer colections
    status = 0
    clt1, status = lib.cl_tracer_collection_t_new(status)
    clt2, status = lib.cl_tracer_collection_t_new(status)
    for t in cltracer1._trc:
        status = lib.add_cl_tracer_to_collection(clt1, t, status)
    for t in cltracer2._trc:
        status = lib.add_cl_tracer_to_collection(clt2, t, status)

    ell_use = np.atleast_1d(ell)

    # Check the values of ell are monotonically increasing
    if not (ell_use[:-1] < ell_use[1:]).all():
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
