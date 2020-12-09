import numpy as np

from . import ccllib as lib
from .pyutils import check, integ_types
from .tk3d import Tk3D

# Define symbolic 'None' type for arrays, to allow proper handling by swig
# wrapper
NoneArr = np.array([])


def angular_cl_cov_cNG(cosmo, cltracer1, cltracer2, ell, tkka, fsky=1.,
                       cltracer3=None, cltracer4=None, ell2=None,
                       integration_method='qag_quad'):
    """Calculate the connected non-Gaussian covariance for a pair of
    power spectra :math:`C_{\\ell_1}^{ab}` and :math:`C_{\\ell_2}^{cd}`,
    between two pairs of tracers (:math:`(a,b)` and :math:`(c,d)`).

    Specifically, it computes:

    .. math::
        {\\rm Cov}_{\\rm cNG}(\\ell_1,\\ell_2)=
        \\int \\frac{d\\chi}{\\chi^6}
        \\tilde{\\Delta}^a_{\\ell_1}(\\chi)
        \\tilde{\\Delta}^b_{\\ell_1}(\\chi)
        \\tilde{\\Delta}^c_{\\ell_2}(\\chi)
        \\tilde{\\Delta}^d_{\\ell_2}(\\chi)\\,
        \\bar{T}_{abcd}\\left[\\frac{\\ell_1+1/2}{\\chi},
                              \\frac{\\ell_2+1/2}{\\chi}, a(\\chi)\\right]

    where :math:`\\Delta^x_\\ell(\\chi)` is the transfer function for tracer
    :math:`x` (see Eq. 39 in the CCL note), and
    :math:`\\bar{T}_{abcd}(k_1,k_2,a)` is the isotropized connected
    trispectrum of the four tracers (see the documentation of the
    :class:`~pyccl.tk3d.Tk3D` class for details).

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        cltracer1 (:class:`~pyccl.tracers.Tracer`): a `Tracer` object,
            of any kind.
        cltracer2 (:class:`~pyccl.tracers.Tracer`): a second `Tracer` object,
            of any kind.
        ell (float or array_like): Angular wavenumber(s) at which to evaluate
            the first dimension of the angular power spectrum covariance.
        tkka (:class:`~pyccl.tk3d.Tk3D` or None): 3D connected trispectrum.
        fsky (float) sky fraction.
        cltracer3 (:class:`~pyccl.tracers.Tracer`): a `Tracer` object,
            of any kind. If `None`, `cltracer1` will be used instead.
        cltracer4 (:class:`~pyccl.tracers.Tracer`): a `Tracer` object,
            of any kind. If `None`, `cltracer1` will be used instead.
        ell2 (float or array_like): Angular wavenumber(s) at which to evaluate
            the second dimension of the angular power spectrum covariance. If
            `None`, `ell` will be used instead.
        integration_method (string) : integration method to be used
            for the Limber integrals. Possibilities: 'qag_quad' (GSL's `qag`
            method backed up by `quad` when it fails) and 'spline' (the
            integrand is splined and then integrated analytically).

    Returns:
        float or array_like: 2D array containing the connected non-Gaussian \
            Angular power spectrum covariance \
            :math:`Cov_{\\rm cNG}(\\ell_1,\\ell_2)`, for the \
            four input tracers, as a function of :math:`\\ell_1` and \
            :math:`\\ell_2`. The ordering is such that \
            `out[i2, i1] = Cov(ell2[i2], ell[i1])`.
    """
    if integration_method not in ['qag_quad', 'spline']:
        raise ValueError("Integration method %s not supported" %
                         integration_method)

    # we need the distances for the integrals
    cosmo.compute_distances()

    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    if isinstance(tkka, Tk3D):
        tsp = tkka.tsp
    else:
        raise ValueError("tkka must be a pyccl.Tk3D")

    # Create tracer colections
    status = 0
    clt1, status = lib.cl_tracer_collection_t_new(status)
    for t in cltracer1._trc:
        status = lib.add_cl_tracer_to_collection(clt1, t, status)
    clt2, status = lib.cl_tracer_collection_t_new(status)
    for t in cltracer2._trc:
        status = lib.add_cl_tracer_to_collection(clt2, t, status)
    if cltracer3 is None:
        clt3 = clt1
    else:
        clt3, status = lib.cl_tracer_collection_t_new(status)
        for t in cltracer3._trc:
            status = lib.add_cl_tracer_to_collection(clt3, t, status)
    if cltracer4 is None:
        clt4 = clt2
    else:
        clt4, status = lib.cl_tracer_collection_t_new(status)
        for t in cltracer4._trc:
            status = lib.add_cl_tracer_to_collection(clt4, t, status)

    ell1_use = np.atleast_1d(ell)
    if ell2 is None:
        ell2 = ell
    ell2_use = np.atleast_1d(ell2)

    cov, status = lib.angular_cov_vec(
        cosmo, clt1, clt2, clt3, clt4, tsp,
        ell1_use, ell2_use, integ_types[integration_method],
        6, 1./(4*np.pi*fsky), ell1_use.size*ell2_use.size, status)

    cov = cov.reshape([ell2_use.size, ell1_use.size])
    if np.ndim(ell2) == 0:
        cov = np.squeeze(cov, axis=0)
    if np.ndim(ell) == 0:
        cov = np.squeeze(cov, axis=-1)

    # Free up tracer collections
    lib.cl_tracer_collection_t_free(clt1)
    lib.cl_tracer_collection_t_free(clt2)
    if cltracer3 is not None:
        lib.cl_tracer_collection_t_free(clt3)
    if cltracer4 is not None:
        lib.cl_tracer_collection_t_free(clt4)

    check(status, cosmo=cosmo_in)
    return cov
