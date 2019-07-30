from . import ccllib as lib
from .pyutils import check
from .pk2d import Pk2D
import numpy as np

# Define symbolic 'None' type for arrays, to allow proper handling by swig
# wrapper
NoneArr = np.array([])


def angular_cl(cosmo, cltracer1, cltracer2, ell, p_of_k_a=None,
               l_limber=-1.):
    """Calculate the angular (cross-)power spectrum for a pair of tracers.

    Args:
        cosmo (:obj:`Cosmology`): A Cosmology object.
        cltracer1, cltracer2 (:obj:`Tracer`): Tracer objects, of any kind.
        ell (float or array_like): Angular wavenumber(s) at which to evaluate
            the angular power spectrum.
        p_of_k_a (:obj:`Pk2D` or None): 3D Power spectrum to project. If None,
            the non-linear matter power spectrum will be used.
        l_limber (float) : Angular wavenumber beyond which Limber's
            approximation will be used. Defaults to -1.

    Returns:
        float or array_like: Angular (cross-)power spectrum values,
            :math:`C_\\ell`, for the pair of tracers, as a function of
            :math:`\\ell`.
    """
    if not cosmo.has_power():
        cosmo.compute_power()

    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    if p_of_k_a is not None:
        if isinstance(p_of_k_a, Pk2D):
            psp = p_of_k_a.psp
        else:
            raise ValueError("p_of_k_a must be either a "
                             "pyccl.Pk2D object or None")
    else:
        psp = None

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
        ell_use, ell_use.size, status)
    if np.isscalar(ell):
        cl = cl[0]

    # Free up tracer collections
    lib.cl_tracer_collection_t_free(clt1)
    lib.cl_tracer_collection_t_free(clt2)

    check(status, cosmo=cosmo_in)
    return cl
