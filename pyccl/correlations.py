__all__ = ("CorrelationMethods", "CorrelationTypes", "correlation",
           "correlation_3d", "correlation_multipole", "correlation_3dRsd",
           "correlation_3dRsd_avgmu", "correlation_pi_sigma",)

from enum import Enum
import numpy as np
from . import DEFAULT_POWER_SPECTRUM, check, lib


class CorrelationMethods(Enum):
    """Choices of algorithms used to compute correlation functions:

    - 'Bessel' is a direct integration using Bessel functions.
    - 'FFTLog' is fast using a fast Fourier transform.
    - 'Legendre' uses a sum over Legendre polynomials.
    """
    FFTLOG = "fftlog"
    BESSEL = "bessel"
    LEGENDRE = "legendre"


class CorrelationTypes(Enum):
    """Correlation function types.
    """
    NN = "NN"
    NG = "NG"
    GG_PLUS = "GG+"
    GG_MINUS = "GG-"


correlation_methods = {
    'fftlog': lib.CCL_CORR_FFTLOG,
    'bessel': lib.CCL_CORR_BESSEL,
    'legendre': lib.CCL_CORR_LGNDRE,
}

correlation_types = {
    'NN': lib.CCL_CORR_GG,
    'NG': lib.CCL_CORR_GL,
    'GG+': lib.CCL_CORR_LP,
    'GG-': lib.CCL_CORR_LM,
}


def correlation(cosmo, *, ell, C_ell, theta, type='NN', method='fftlog'):
    r"""Compute the angular correlation function.

    .. math::

        \xi^{ab}_\pm(\theta) =
        \sum_\ell\frac{2\ell+1}{4\pi}\,(\pm1)^{s_b}\,
        C^{ab\pm}_\ell\,d^\ell_{s_a,\pm s_b}(\theta)

    where :math:`\theta` is the angle between the two fields :math:`a` and
    :math:`b` with spins :math:`s_a` and :math:`s_b` after alignement of their
    tangential coordinate. :math:`d^\ell_{mm'}` are the Wigner-d matrices and
    we have defined the power spectra

    .. math::
        C^{ab\pm}_\ell \equiv
        (C^{a_Eb_E}_\ell \pm C^{a_Bb_B}_\ell)+i
        (C^{a_Bb_E}_\ell \mp C^{a_Eb_B}_\ell),

    which reduces to the :math:`EE` power spectrum when all :math:`B`-modes
    are 0.

    The different spin combinations are:

        * :math:`s_a=s_b=0` e.g. galaxy-galaxy, galaxy-:math:`\kappa`
          and :math:`\kappa`-:math:`\kappa`
        * :math:`s_a=2`, :math:`s_b=0` e.g. galaxy-shear, and :math:`\kappa`-shear
        * :math:`s_a=s_b=2` e.g. shear-shear.

    .. note::
        For scales smaller than :math:`\sim 0.1^{\circ}`, the input power
        spectrum should be sampled to sufficienly high :math:`\ell` to ensure
        the Hankel transform is well-behaved. The following spline parameters,
        related to ``FFTLog``-sampling may also be modified for accuracy:

            * ``ccl.spline_params.ELL_MIN_CORR``
            * ``ccl.spline_params.ELL_MAX_CORR``
            * ``ccl.spline_params.N_ELL_CORR``.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        ell (array): Multipoles corresponding to the input angular power
                          spectrum.
        C_ell (array): Input angular power spectrum.
        theta (:obj:`float` or `array`): Angular separation(s) at which to
            calculate the angular correlation function (in degrees).
        type (:obj:`str`): Type of correlation function. Choices: ``'NN'`` (0x0),
            ``'NG'`` (0x2), ``'GG+'`` (2x2, :math:`\xi_+`),
            ``'GG-'`` (2x2, :math:`\xi_-`), where numbers refer to the spins
            of the two quantities being cross-correlated (see Section 2.4.2 of
            the CCL paper). The naming system roughly follows the nomenclature
            used in `TreeCorr
            <https://rmjarvis.github.io/TreeCorr/_build/html/correlation2.html>`_.
        method (:obj:`str`): Method to compute the correlation function.
            Choices: ``'Bessel'`` (direct integration over Bessel function),
            ``'FFTLog'`` (fast integration with FFTLog), ``'Legendre'``
            (brute-force sum over Legendre polynomials).

    Returns:
        (:obj:`float` or `array`): Value(s) of the correlation function at the
        input angular separations.
    """ # noqa
    cosmo_in = cosmo
    cosmo = cosmo.cosmo
    status = 0
    method = method.lower()

    if type not in correlation_types:
        raise ValueError(f"Invalid correlation type {type}.")

    if method not in correlation_methods.keys():
        raise ValueError(f"Invalid correlation method {method}.")

    # Convert scalar input into an array
    if scalar := isinstance(theta, (int, float)):
        theta = np.array([theta, ])

    if np.all(np.array(C_ell) == 0):
        # short-cut and also avoid integration errors
        wth = np.zeros_like(theta)
    else:
        # Call correlation function
        wth, status = lib.correlation_vec(cosmo, ell, C_ell, theta,
                                          correlation_types[type],
                                          correlation_methods[method],
                                          len(theta), status)
    check(status, cosmo_in)
    if scalar:
        return wth[0]
    return wth


def correlation_3d(cosmo, *, r, a, p_of_k_a=DEFAULT_POWER_SPECTRUM):
    r"""Compute the 3D correlation function:

    .. math::
        \xi(r)\equiv\frac{1}{2\pi^2}\int dk\,k^2\,P(k)\,j_0(kr).

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        r (:obj:`float` or `array`): distance(s) at which to calculate the 3D
            correlation function (in Mpc).
        a (:obj:`float`): scale factor.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).

    Returns:
        Value(s) of the correlation function at the input distance(s).
    """ # noqa
    cosmo.compute_nonlin_power()

    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = cosmo_in.parse_pk2d(p_of_k_a)

    status = 0

    # Convert scalar input into an array
    if scalar := isinstance(r, (int, float)):
        r = np.array([r, ])

    # Call 3D correlation function
    xi, status = lib.correlation_3d_vec(cosmo, psp, a, r,
                                        len(r), status)
    check(status, cosmo_in)
    if scalar:
        return xi[0]
    return xi


def correlation_multipole(cosmo, *, r, a, beta, ell,
                          p_of_k_a=DEFAULT_POWER_SPECTRUM):
    r"""Compute the correlation function multipoles:

    .. math::
        \xi_\ell(r)\equiv\frac{i^\ell}{2\pi^2}\int dk\,k^2\,P(k)\,j_\ell(kr).

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        r (:obj:`float` or `array`): distance(s) at which to calculate the 3D
            correlation function (in Mpc).
        a (:obj:`float`): scale factor.
        beta (:obj:`float`): growth rate divided by galaxy bias.
        ell (:obj:`int`) : the desired multipole
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).

    Returns:
        Value(s) of the correlation function at the input distance(s).
    """ # noqa
    cosmo.compute_nonlin_power()

    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = cosmo_in.parse_pk2d(p_of_k_a)

    status = 0

    # Convert scalar input into an array
    if scalar := isinstance(r, (int, float)):
        r = np.array([r, ])

    # Call 3D correlation function
    xis, status = lib.correlation_multipole_vec(cosmo, psp, a, beta, ell, r,
                                                len(r), status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis


def correlation_3dRsd(cosmo, *, r, a, mu, beta,
                      p_of_k_a=DEFAULT_POWER_SPECTRUM, use_spline=True):
    r"""
    Compute the 3D correlation function with linear RSDs using
    multipoles:

    .. math::
        \xi(r,\mu) = \sum_{\ell\in\{0,2,4\}}\xi_\ell(r)\,P_\ell(\mu)

    where :math:`P_\ell(\mu)` are the Legendre polynomials, and
    :math:`\xi_\ell(r)` are the correlation function multipoles.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        r (:obj:`float` or `array`): distance(s) at which to calculate the
            3D correlation function (in Mpc).
        a (:obj:`float`): scale factor.
        mu (:obj:`float`): cosine of the angle at which to calculate the 3D
            correlation function.
        beta (:obj:`float`): growth rate divided by galaxy bias.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).
        use_spline (:obj:`bool`): switch that determines whether the RSD correlation
            function is calculated using global splines of multipoles.

    Returns:
        Value(s) of the correlation function at the input distance(s) & angle.
    """ # noqa
    cosmo.compute_nonlin_power()

    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = cosmo_in.parse_pk2d(p_of_k_a)

    status = 0

    # Convert scalar input into an array
    if scalar := isinstance(r, (int, float)):
        r = np.array([r, ])

    # Call 3D correlation function
    xis, status = lib.correlation_3dRsd_vec(cosmo, psp, a, mu, beta, r,
                                            len(r), int(use_spline), status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis


def correlation_3dRsd_avgmu(cosmo, *, r, a, beta,
                            p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """
    Compute the 3D correlation function averaged over angles with
    RSDs. Equivalent to calling :func:`correlation_multipole`
    with ``ell=0``.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        r (:obj:`float` or `array`): distance(s) at which to calculate the 3D
            correlation function (in Mpc).
        a (:obj:`float`): scale factor.
        beta (:obj:`float`): growth rate divided by galaxy bias.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).

    Returns:
        Value(s) of the correlation function at the input distance(s) & angle.
    """ # noqa
    cosmo.compute_nonlin_power()

    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = cosmo_in.parse_pk2d(p_of_k_a)

    status = 0

    # Convert scalar input into an array
    if scalar := isinstance(r, (int, float)):
        r = np.array([r, ])

    # Call 3D correlation function
    xis, status = lib.correlation_3dRsd_avgmu_vec(cosmo, psp, a, beta, r,
                                                  len(r), status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis


def correlation_pi_sigma(cosmo, *, pi, sigma, a, beta,
                         use_spline=True, p_of_k_a=DEFAULT_POWER_SPECTRUM):
    r"""
    Compute the 3D correlation in :math:`(\pi,\sigma)` space. This is
    just

    .. math::
        \xi(\pi,\sigma) = \xi(r=\sqrt{\pi^2+\sigma^2},\mu=\pi/r).

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        pi (:obj:`float`): distance times cosine of the angle (in Mpc).
        sigma (:obj:`float` or `array`): distance(s) times sine of the angle
            (in Mpc).
        a (:obj:`float`): scale factor.
        beta (:obj:`float`): growth rate divided by galaxy bias.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).
        use_spline (:obj:`bool`): switch that determines whether the RSD correlation
            function is calculated using global splines of multipoles.

    Returns:
        Value(s) of the correlation function at the input pi and sigma.
    """ # noqa
    cosmo.compute_nonlin_power()

    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = cosmo_in.parse_pk2d(p_of_k_a)

    status = 0

    # Convert scalar input into an array
    if scalar := isinstance(sigma, (int, float)):
        sigma = np.array([sigma, ])

    # Call 3D correlation function
    xis, status = lib.correlation_pi_sigma_vec(cosmo, psp, a, beta, pi, sigma,
                                               len(sigma), int(use_spline),
                                               status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis
