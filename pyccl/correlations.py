"""Correlation function computations.

Choices of algorithms used to compute correlation functions:
    'Bessel' is a direct integration using Bessel functions.
    'FFTLog' is fast using a fast Fourier transform.
    'Legendre' uses a sum over Legendre polynomials.
"""

from . import ccllib as lib
from . import constants as const
from . import DEFAULT_POWER_SPECTRUM
from .pyutils import check
from .pk2d import parse_pk2d
from .base import warn_api
from .errors import CCLDeprecationWarning
import numpy as np
import warnings

correlation_methods = {
    'fftlog': const.CCL_CORR_FFTLOG,
    'bessel': const.CCL_CORR_BESSEL,
    'legendre': const.CCL_CORR_LGNDRE,
}

correlation_types = {
    'NN': const.CCL_CORR_GG,
    'NG': const.CCL_CORR_GL,
    'GG+': const.CCL_CORR_LP,
    'GG-': const.CCL_CORR_LM,
}


@warn_api
def correlation(cosmo, *, ell, C_ell, theta, type='NN', corr_type=None,
                method='fftlog'):
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

       C^{ab\pm}_\ell \equiv\ left(C^{a_Eb_E}_\ell\pm C^{a_Bb_B}_\ell\right)
       + i\left(C^{a_Bb_E}_\ell\mp C^{a_Eb_B}_\ell\right)

    which reduces to the :math:`EE` power spectrum when all :math:`B`-modes
    are 0.

    The different spin combinaisons are:

      -  :math:`s_a=s_b=0` e.g. galaxy-galaxy, galaxy-:math:`\kappa`
         and :math:`\kappa`-:math:`\kappa`

      - :math:`s_a=2`, `s_b=0` e.g. galaxy-shear, and :math:`\kappa`-shear

      - :math:`s_a=s_b=2` e.g. shear-shear.

    .. note::

        For scales smaller than :math:`\sim 0.1^{\circ}`, the input power
        spectrum should be sampled to sufficienly high :math:`\ell` to ensure
        the Hankel transform is well-behaved. The following spline parameters,
        related to ``FFTLog``-sampling may also be modified for accuracy:
            - ``ccl.spline_params.ELL_MIN_CORR``
            - ``ccl.spline_params.ELL_MAX_CORR``
            - ``ccl.spline_params.N_ELL_CORR``.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        ell (array_like): Multipoles corresponding to the input angular power
                          spectrum.
        C_ell (array_like): Input angular power spectrum.
        theta (float or array_like): Angular separation(s) at which to
                                     calculate the angular correlation
                                     function (in degrees).
        type (string): Type of correlation function. Choices:
                       'NN' (0x0), 'NG' (0x2),
                       'GG+' (2x2, xi+),
                       'GG-' (2x2, xi-), where numbers refer to the
                       spins of the two quantities being cross-correlated
                       (see Section 2.4.2 of the CCL paper).
        method (string, optional): Method to compute the correlation function.
                                   Choices: 'Bessel' (direct integration over
                                   Bessel function), 'FFTLog' (fast
                                   integration with FFTLog), 'Legendre'
                                   (brute-force sum over Legendre polynomials).
        corr_type (string): (deprecated, please use `type`)
                            Type of correlation function. Choices:
                            'gg' (0x0), 'gl' (0x2),
                            'l+' (2x2, xi+),
                            'l-' (2x2, xi-), where the numbers refer to the
                            spins of the two quantities being cross-correlated
                            (see Section 2.4.2 of the CCL paper).

    Returns:
        float or array_like: Value(s) of the correlation function at the \
            input angular separations.
    """
    cosmo_in = cosmo
    cosmo = cosmo.cosmo
    status = 0

    if corr_type is not None:
        # Convert to lower case
        corr_type = corr_type.lower()
        if corr_type == 'gg':
            type = 'NN'
        elif corr_type == 'gl':
            type = 'NG'
        elif corr_type == 'l+':
            type = 'GG+'
        elif corr_type == 'l-':
            type = 'GG-'
        else:
            raise ValueError("Unknown corr_type " + corr_type)
        warnings.warn("corr_type is deprecated. Use type = {}".format(type),
                      CCLDeprecationWarning)
    method = method.lower()

    if type not in correlation_types.keys():
        raise ValueError("'%s' is not a valid correlation type." % type)

    if method not in correlation_methods.keys():
        raise ValueError("'%s' is not a valid correlation method." % method)

    # Convert scalar input into an array
    scalar = False
    if isinstance(theta, (int, float)):
        scalar = True
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


@warn_api(reorder=['a', 'r'])
def correlation_3d(cosmo, *, r, a, p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """Compute the 3D correlation function.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        r (float or array_like): distance(s) at which to calculate the 3D
            correlation function (in Mpc).
        a (float): scale factor.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).

    Returns:
        Value(s) of the correlation function at the input distance(s).
    """
    cosmo.compute_nonlin_power()

    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = parse_pk2d(cosmo_in, p_of_k_a)

    status = 0

    # Convert scalar input into an array
    scalar = False
    if isinstance(r, (int, float)):
        scalar = True
        r = np.array([r, ])

    # Call 3D correlation function
    xi, status = lib.correlation_3d_vec(cosmo, psp, a, r,
                                        len(r), status)
    check(status, cosmo_in)
    if scalar:
        return xi[0]
    return xi


@warn_api(pairs=[('s', 'r'), ('l', 'ell')],
          reorder=['a', 'beta', 'ell', 'r'])
def correlation_multipole(cosmo, *, r, a, beta, ell,
                          p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """Compute the correlation multipoles.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        r (float or array_like): distance(s) at which to calculate the 3DRsd
            correlation function (in Mpc).
        a (float): scale factor.
        beta (float): growth rate divided by galaxy bias.
        ell (int) : the desired multipole
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).

    Returns:
        Value(s) of the correlation function at the input distance(s).
    """
    cosmo.compute_nonlin_power()

    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = parse_pk2d(cosmo_in, p_of_k_a)

    status = 0

    # Convert scalar input into an array
    scalar = False
    if isinstance(r, (int, float)):
        scalar = True
        r = np.array([r, ])

    # Call 3D correlation function
    xis, status = lib.correlation_multipole_vec(cosmo, psp, a, beta, ell, r,
                                                len(r), status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis


@warn_api(pairs=[('s', 'r')],
          reorder=['a', 'r', 'mu', 'beta', 'use_spline', 'p_of_k_a'])
def correlation_3dRsd(cosmo, *, r, a, mu, beta,
                      p_of_k_a=DEFAULT_POWER_SPECTRUM, use_spline=True):
    """
    Compute the 3DRsd correlation function using linear approximation
    with multipoles.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        r (float or array_like): distance(s) at which to calculate the
            3DRsd correlation function (in Mpc).
        a (float): scale factor.
        mu (float): cosine of the angle at which to calculate the 3DRsd
            correlation function (in Radian).
        beta (float): growth rate divided by galaxy bias.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).
        use_spline: switch that determines whether the RSD correlation
            function is calculated using global splines of multipoles.

    Returns:
        Value(s) of the correlation function at the input distance(s) & angle.
    """
    cosmo.compute_nonlin_power()

    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = parse_pk2d(cosmo_in, p_of_k_a)

    status = 0

    # Convert scalar input into an array
    scalar = False
    if isinstance(r, (int, float)):
        scalar = True
        r = np.array([r, ])

    # Call 3D correlation function
    xis, status = lib.correlation_3dRsd_vec(cosmo, psp, a, mu, beta, r,
                                            len(r), int(use_spline), status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis


@warn_api(pairs=[('s', 'r')], reorder=['a', 'r'])
def correlation_3dRsd_avgmu(cosmo, *, r, a, beta,
                            p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """
    Compute the 3DRsd correlation function averaged over mu at constant s.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        r (float or array_like): distance(s) at which to calculate the 3DRsd
            correlation function (in Mpc).
        a (float): scale factor.
        beta (float): growth rate divided by galaxy bias.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).

    Returns:
        Value(s) of the correlation function at the input distance(s) & angle.
    """
    cosmo.compute_nonlin_power()

    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = parse_pk2d(cosmo_in, p_of_k_a)

    status = 0

    # Convert scalar input into an array
    scalar = False
    if isinstance(r, (int, float)):
        scalar = True
        r = np.array([r, ])

    # Call 3D correlation function
    xis, status = lib.correlation_3dRsd_avgmu_vec(cosmo, psp, a, beta, r,
                                                  len(r), status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis


@warn_api(pairs=[("sig", "sigma")],
          reorder=['a', 'beta', 'pi', 'sigma'])
def correlation_pi_sigma(cosmo, *, pi, sigma, a, beta,
                         use_spline=True, p_of_k_a=DEFAULT_POWER_SPECTRUM):
    """
    Compute the 3DRsd correlation in pi-sigma space.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        pi (float): distance times cosine of the angle (in Mpc).
        sigma (float or array-like): distance(s) times sine of the angle
            (in Mpc).
        a (float): scale factor.
        beta (float): growth rate divided by galaxy bias.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).
        use_spline: switch that determines whether the RSD correlation
            function is calculated using global splines of multipoles.

    Returns:
        Value(s) of the correlation function at the input pi and sigma.
    """
    cosmo.compute_nonlin_power()

    cosmo_in = cosmo
    cosmo = cosmo.cosmo

    psp = parse_pk2d(cosmo_in, p_of_k_a)

    status = 0

    # Convert scalar input into an array
    scalar = False
    if isinstance(sigma, (int, float)):
        scalar = True
        sigma = np.array([sigma, ])

    # Call 3D correlation function
    xis, status = lib.correlation_pi_sigma_vec(cosmo, psp, a, beta, pi, sigma,
                                               len(sigma), int(use_spline),
                                               status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis
