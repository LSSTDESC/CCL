"""
==========================================
Angular power spectra (:mod:`pyccl.cells`)
==========================================

Computations of angular power spectra.
"""
from __future__ import annotations

__all__ = ("angular_cl",)

import warnings
from numbers import Real
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray

from . import DEFAULT_POWER_SPECTRUM, CCLWarning, lib, warn_api
from .pyutils import integ_types

if TYPE_CHECKING:
    from . import Cosmology, Pk2D, Tracer


@warn_api(pairs=[("cltracer1", "tracer1"), ("cltracer2", "tracer2")])
def angular_cl(
        cosmo: Cosmology,
        tracer1: Tracer,
        tracer2: Tracer,
        ell: Union[Real, NDArray[Real]],
        *,
        p_of_k_a: Union[str, Pk2D] = DEFAULT_POWER_SPECTRUM,
        l_limber: Real = -1,
        limber_integration_method: str = 'qag_quad'
) -> Union[float, NDArray[float]]:
    r"""Angular (cross-)power spectrum for a pair of tracers.

    Currently uses the Limber approximation :footcite:p:`Limber53`:

    .. math::

        C_{uv}(\ell) = \int {\rm d}\chi \frac{W_u(\chi) W_v(\chi)}{\chi^2} \,
        P_{UV}\left( k = \frac{\ell + 1/2}{\chi}, z(\chi) \right),

    where :math:`(u, v)` are the correlated quantities, :math:`W(\chi)` are the
    associated radial kernels, and :math:`P(k, z)` is the 3-D power spectrum of
    :math:`u` and :math:`v`:

    .. math::

        \langle U(\mathbf{k}) V^*(\mathbf{k'}) \rangle = (2\pi)^3
        \delta(\mathbf{k} - \mathbf{k'}) \, P_{UV}(k).

    Arguments
    ---------
    cosmo
        Cosmological parameters.
    tracer1, tracer2
        Tracer.
    ell : array_like (nell,)
        Multipoles at which the angular power spectrum is evaluated.
    p_of_k_a
        3-D power spectrum to project.
    l_limber
        Cutoff wavenumber (in :math:`\rm Mpc^{-1}`) for Limber integration.
    limber_integration_method
        Integration method. Available options in
        :class:`~pyccl.pyutils.IntegrationMethods`.

    Returns
    -------
    array_like (nell,)
        Angular power spectrum, :math:`C(\ell)`.

    References
    ----------
    .. footbibliography::
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

    cosmo_in.check(status)
    return cl
