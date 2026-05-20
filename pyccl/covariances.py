__all__ = ("angular_cl_cov_cNG", "sigma2_B_disc", "sigma2_B_from_mask",
           "angular_cl_cov_SSC",)

import numpy as np

from . import DEFAULT_POWER_SPECTRUM, check, lib
from .pyutils import _check_array_params, integ_types


def angular_cl_cov_cNG(cosmo, tracer1, tracer2, *, ell, t_of_kk_a,
                       tracer3=None, tracer4=None, ell2=None,
                       fsky=1., integration_method='qag_quad'):
    """Calculate the connected non-Gaussian covariance for a pair of
    angular power spectra :math:`C_{\\ell_1}^{ab}` and :math:`C_{\\ell_2}^{cd}`,
    between two pairs of tracers (:math:`(a,b)` and :math:`(c,d)`).

    Specifically, it computes:

    .. math::
        {\\rm Cov}_{\\rm cNG}(\\ell_1,\\ell_2)=\\frac{1}{4\\pi f_{\\rm sky}}
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
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        tracer1 (:class:`~pyccl.tracers.Tracer`): a Tracer object.
        tracer2 (:class:`~pyccl.tracers.Tracer`): a second Tracer object.
        ell (:obj:`float` or `array`): Angular wavenumber(s) at which to evaluate
            the first dimension of the angular power spectrum covariance.
        t_of_kk_a (:class:`~pyccl.tk3d.Tk3D`): 3D connected
            trispectrum.
        tracer3 (:class:`~pyccl.tracers.Tracer`): a Tracer object.
            If ``None``, ``tracer1`` will be used instead.
        tracer4 (:class:`~pyccl.tracers.Tracer`): a Tracer object.
            If ``None``, ``tracer2`` will be used instead.
        ell2 (:obj:`float` or `array`): Angular wavenumber(s) at which to evaluate
            the second dimension of the angular power spectrum covariance. If
            ``None``, ``ell`` will be used instead.
        fsky (:obj:`float`): sky fraction.
        integration_method (:obj:`str`) : integration method to be used
            for the Limber integrals. Possibilities: ``'qag_quad'`` (GSL's
            `qag` method backed up by `quad` when it fails) and ``'spline'``
            (the integrand is splined and then integrated analytically).

    Returns:
        (:obj:`float` or `array`): 2D array containing the connected non-Gaussian \
            Angular power spectrum covariance \
            :math:`{\\rm Cov}_{\\rm cNG}(\\ell_1,\\ell_2)`, for the \
            four input tracers, as a function of :math:`\\ell_1` and \
            :math:`\\ell_2`. The ordering is such that \
            ``out[i2, i1] = Cov(ell2[i2], ell[i1])``.
    """ # noqa
    if integration_method not in integ_types:
        raise ValueError(f"Unknown integration method {integration_method}.")

    # we need the distances for the integrals
    cosmo.compute_distances()

    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    tsp = t_of_kk_a.tsp

    # Create tracer colections
    status = 0
    clt1, status = lib.cl_tracer_collection_t_new(status)
    for t in tracer1._trc:
        status = lib.add_cl_tracer_to_collection(clt1, t, status)
    clt2, status = lib.cl_tracer_collection_t_new(status)
    for t in tracer2._trc:
        status = lib.add_cl_tracer_to_collection(clt2, t, status)
    if tracer3 is None:
        clt3 = clt1
    else:
        clt3, status = lib.cl_tracer_collection_t_new(status)
        for t in tracer3._trc:
            status = lib.add_cl_tracer_to_collection(clt3, t, status)
    if tracer4 is None:
        clt4 = clt2
    else:
        clt4, status = lib.cl_tracer_collection_t_new(status)
        for t in tracer4._trc:
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
    if tracer3 is not None:
        lib.cl_tracer_collection_t_free(clt3)
    if tracer4 is not None:
        lib.cl_tracer_collection_t_free(clt4)

    check(status, cosmo=cosmo_in)
    return cov


def sigma2_B_disc(cosmo, a_arr=None, *, fsky=1.,
                  p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """Returns the variance of the projected linear density field
    over a circular disc covering a sky fraction `fsky` as a function
    of scale factor. This is given by

    .. math::
        \\sigma^2_B(z) = \\int_0^\\infty \\frac{k\\,dk}{2\\pi}
            P_L(k,z)\\,\\left[\\frac{2J_1(k R(z))}{k R(z)}\\right]^2,

    where :math:`R(z)` is the corresponding radial aperture as a
    function of redshift. This quantity can be used to compute the
    super-sample covariance (see :func:`angular_cl_cov_SSC`).

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        a_arr (:obj:`float`, `array` or :obj:`None`): an array of scale factor
            values at which to evaluate the projected variance. If
            ``None``, a default sampling will be used.
        fsky (:obj:`float`): sky fraction.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D` or :obj:`str`): Linear
            power spectrum to use. Defaults to the
            internal linear power spectrum from ``cosmo``.

    Returns:
        Tuple containing

        - a_arr (`array`): an array of scale factor values at which the
          projected variance has been evaluated. Only returned if ``a_arr``
          is ``None`` on input.
        - sigma2_B (:obj:`float` or `array`): projected variance.
    """
    full_output = a_arr is None

    if full_output:
        a_arr = cosmo.get_pk_spline_a()
    else:
        ndim = np.ndim(a_arr)
        a_arr = np.atleast_1d(a_arr)

    chi_arr = cosmo.comoving_radial_distance(a_arr)
    R_arr = chi_arr * np.arccos(1-2*fsky)
    psp = cosmo.parse_pk2d(p_of_k_a, is_linear=True)

    status = 0
    s2B_arr, status = lib.sigma2b_vec(cosmo.cosmo, a_arr, R_arr, psp,
                                      len(a_arr), status)
    check(status, cosmo=cosmo)

    if full_output:
        return a_arr, s2B_arr
    if ndim == 0:
        return s2B_arr[0]
    return s2B_arr


def sigma2_B_from_mask(cosmo, a_arr=None, *, mask_wl=None,
                       p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """ Returns the variance of the projected linear density field, given the
    angular power spectrum of the footprint mask and scale factor. This is
    given by

    .. math::
        \\sigma^2_B(z) = \\frac{1}{\\chi^2(z)}\\sum_\\ell
            P_L\\left(\\frac{\\ell+\\frac{1}{2}}{\\chi(z)},z\\right)\\,
            \\sum_{m=-\\ell}^\\ell W^A_{\\ell m} {W^B}^*_{\\ell m},

    where :math:`W^A_{\\ell m}` and :math:`W^B_{\\ell m}` are the spherical
    harmonics decomposition of the footprint masks of fields `A` and `B`,
    normalized by their area. This quantity can be used to compute the
    super-sample covariance (see :func:`angular_cl_cov_SSC`).

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        a_arr (:obj:`float`, `array` or :obj:`None`): an array of scale factor
            values at which to evaluate the projected variance.
        mask_wl (`array`): Array with the angular power spectrum of the
            masks. The power spectrum should be given at integer multipoles,
            starting at :math:`\\ell=0`. The power spectrum is normalized as
            :math:`{\\tt mask\\_wl}=\\sum_m W^A_{\\ell m} {W^B}^*_{\\ell m}`.
            It is the responsibility of the user to the provide the mask power
            out to sufficiently high ell for their required precision.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D` or :obj:`str`): Linear
            power spectrum to use. Defaults to the
            internal linear power spectrum from `cosmo`.

    Returns:
        Tuple containing

        - a_arr (`array`): an array of scale factor values at which the
          projected variance has been evaluated. Only returned if ``a_arr`` is
          ``None`` on input.
        - sigma2_B (:obj:`float` or `array`): projected variance.
    """
    full_output = a_arr is None

    if full_output:
        a_arr = cosmo.get_pk_spline_a()
    else:
        ndim = np.ndim(a_arr)
        a_arr = np.atleast_1d(a_arr)

    if p_of_k_a == DEFAULT_POWER_SPECTRUM:
        cosmo.compute_linear_power()
        p_of_k_a = cosmo.get_linear_power()

    ell = np.arange(mask_wl.size)

    sigma2_B = np.zeros(a_arr.size)
    for i in range(sigma2_B.size):
        if 1-a_arr[i] < 1e-6:
            # For a=1, the integral becomes independent of the footprint in
            # the flat-sky approximation. So we are just using the method
            # for the disc geometry here
            sigma2_B[i] = sigma2_B_disc(cosmo=cosmo, a_arr=a_arr[i],
                                        p_of_k_a=p_of_k_a)
        else:
            chi = cosmo.comoving_angular_distance(a=a_arr)
            k = (ell+0.5)/chi[i]
            pk = p_of_k_a(k, a_arr[i], cosmo)
            # See eq. E.10 of 2007.01844
            sigma2_B[i] = np.sum(pk * mask_wl)/chi[i]**2

    if full_output:
        return a_arr, sigma2_B
    if ndim == 0:
        return sigma2_B[0]
    return sigma2_B


def angular_cl_cov_SSC(cosmo, tracer1, tracer2, *, ell, t_of_kk_a,
                       tracer3=None, tracer4=None, ell2=None,
                       sigma2_B=None, fsky=1.,
                       integration_method='qag_quad'):
    """Calculate the super-sample contribution to the connected
    non-Gaussian covariance for a pair of power spectra
    :math:`C_{\\ell_1}^{ab}` and :math:`C_{\\ell_2}^{cd}`,
    between two pairs of tracers (:math:`(a,b)` and :math:`(c,d)`).

    Specifically, it computes:

    .. math::
        {\\rm Cov}_{\\rm SSC}(\\ell_1,\\ell_2)=
        \\int \\frac{d\\chi}{\\chi^4}
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
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        tracer1 (:class:`~pyccl.tracers.Tracer`): a Tracer object.
        tracer2 (:class:`~pyccl.tracers.Tracer`): a second Tracer object.
        ell (:obj:`float` or `array`): Angular wavenumber(s) at which to evaluate
            the first dimension of the angular power spectrum covariance.
        t_of_kk_a (:class:`~pyccl.tk3d.Tk3D`): 3D connected
            trispectrum.
        tracer3 (:class:`~pyccl.tracers.Tracer`): a Tracer object.
            If ``None``, ``tracer1`` will be used instead.
        tracer4 (:class:`~pyccl.tracers.Tracer`): a Tracer object.
            If ``None``, ``tracer2`` will be used instead.
        ell2 (:obj:`float` or `array`): Angular wavenumber(s) at which to evaluate
            the second dimension of the angular power spectrum covariance. If
            ``None``, ``ell`` will be used instead.
        sigma2_B (:obj:`tuple` or :obj:`None`): A tuple of arrays
            (a, sigma2_B(a)) containing the variance of the projected matter
            overdensity over the footprint as a function of the scale factor.
            If ``None``, a compact circular footprint will be assumed covering
            a sky fraction ``fsky``.
        fsky (:obj:`float`): sky fraction.
        integration_method (:obj:`str`) : integration method to be used
            for the Limber integrals. Possibilities: ``'qag_quad'`` (GSL's
            `qag` method backed up by `quad` when it fails) and ``'spline'``
            (the integrand is splined and then integrated analytically).

    Returns:
        (:obj:`float` or `array`): 2D array containing the super-sample \
            Angular power spectrum covariance \
            :math:`{\\rm Cov}_{\\rm SSC}(\\ell_1,\\ell_2)`, for the \
            four input tracers, as a function of :math:`\\ell_1` and \
            :math:`\\ell_2`. The ordering is such that \
            ``out[i2, i1] = Cov(ell2[i2], ell[i1])``.
    """ # noqa
    if integration_method not in integ_types:
        raise ValueError(f"Unknown integration method {integration_method}.")

    # we need the distances for the integrals
    cosmo.compute_distances()

    # Access ccl_cosmology object
    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    tsp = t_of_kk_a.tsp

    # Create tracer colections
    status = 0
    clt1, status = lib.cl_tracer_collection_t_new(status)
    for t in tracer1._trc:
        status = lib.add_cl_tracer_to_collection(clt1, t, status)
    clt2, status = lib.cl_tracer_collection_t_new(status)
    for t in tracer2._trc:
        status = lib.add_cl_tracer_to_collection(clt2, t, status)
    if tracer3 is None:
        clt3 = clt1
    else:
        clt3, status = lib.cl_tracer_collection_t_new(status)
        for t in tracer3._trc:
            status = lib.add_cl_tracer_to_collection(clt3, t, status)
    if tracer4 is None:
        clt4 = clt2
    else:
        clt4, status = lib.cl_tracer_collection_t_new(status)
        for t in tracer4._trc:
            status = lib.add_cl_tracer_to_collection(clt4, t, status)

    ell1_use = np.atleast_1d(ell)
    if ell2 is None:
        ell2 = ell
    ell2_use = np.atleast_1d(ell2)

    if sigma2_B is None:
        a_arr, s2b_arr = sigma2_B_disc(cosmo_in, fsky=fsky)
    else:
        a_arr, s2b_arr = _check_array_params(sigma2_B, 'sigma2_B')
    cov, status = lib.angular_cov_ssc_vec(
        cosmo, clt1, clt2, clt3, clt4, tsp, a_arr, s2b_arr,
        ell1_use, ell2_use, integ_types[integration_method],
        4, 1., ell1_use.size*ell2_use.size, status)

    cov = cov.reshape([ell2_use.size, ell1_use.size])
    if np.ndim(ell2) == 0:
        cov = np.squeeze(cov, axis=0)
    if np.ndim(ell) == 0:
        cov = np.squeeze(cov, axis=-1)

    # Free up tracer collections
    lib.cl_tracer_collection_t_free(clt1)
    lib.cl_tracer_collection_t_free(clt2)
    if tracer3 is not None:
        lib.cl_tracer_collection_t_free(clt3)
    if tracer4 is not None:
        lib.cl_tracer_collection_t_free(clt4)

    check(status, cosmo=cosmo_in)
    return cov
