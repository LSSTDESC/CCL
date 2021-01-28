"""Correlation functon computations.

Choices of algorithms used to compute correlation functions:
    'Bessel' is a direct integration using Bessel functions.
    'FFTLog' is fast using a fast Fourier transform.
    'Legendre' uses a sum over Legendre polynomials.
"""

from . import ccllib as lib
from . import constants as const
from .core import check
from .pk2d import parse_pk2d
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


def correlation(cosmo, ell, C_ell, theta, type='NN', corr_type=None,
                method='fftlog'):
    """Compute the angular correlation function.

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
    from .errors import CCLWarning
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
                      CCLWarning)
    method = method.lower()

    if type not in correlation_types.keys():
        raise ValueError("'%s' is not a valid correlation type." % type)

    if method not in correlation_methods.keys():
        raise ValueError("'%s' is not a valid correlation method." % method)

    # Convert scalar input into an array
    scalar = False
    if isinstance(theta, float) or isinstance(theta, int):
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


def correlation_3d(cosmo, a, r, p_of_k_a=None):
    """Compute the 3D correlation function.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        a (float): scale factor.
        r (float or array_like): distance(s) at which to calculate the 3D
                                 correlation function (in Mpc).
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`). If `None`, the non-linear matter
            power spectrum stored in `cosmo` will be used.

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
    if isinstance(r, float) or isinstance(r, int):
        scalar = True
        r = np.array([r, ])

    # Call 3D correlation function
    xi, status = lib.correlation_3d_vec(cosmo, psp, a, r,
                                        len(r), status)
    check(status, cosmo_in)
    if scalar:
        return xi[0]
    return xi


def correlation_multipole(cosmo, a, beta, l, s, p_of_k_a=None):
    """Compute the correlation multipoles.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        a (float): scale factor.
        beta (float): growth rate divided by galaxy bias.
        l (int) : the desired multipole
        s (float or array_like): distance(s) at which to calculate the 3DRsd
                                 correlation function (in Mpc).
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`). If `None`, the non-linear matter
            power spectrum stored in `cosmo` will be used.

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
    if isinstance(s, float) or isinstance(s, int):
        scalar = True
        s = np.array([s, ])

    # Call 3D correlation function
    xis, status = lib.correlation_multipole_vec(cosmo, psp, a, beta, l, s,
                                                len(s), status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis


def correlation_3dRsd(cosmo, a, s, mu, beta, use_spline=True, p_of_k_a=None):
    """
    Compute the 3DRsd correlation function using linear approximation
    with multipoles.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        a (float): scale factor.
        s (float or array_like): distance(s) at which to calculate the
                                 3DRsd correlation function (in Mpc).
        mu (float): cosine of the angle at which to calculate the 3DRsd
                    correlation function (in Radian).
        beta (float): growth rate divided by galaxy bias.
        use_spline: switch that determines whether the RSD correlation
                    function is calculated using global splines of multipoles.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`). If `None`, the non-linear matter
            power spectrum stored in `cosmo` will be used.

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
    if isinstance(s, float) or isinstance(s, int):
        scalar = True
        s = np.array([s, ])

    # Call 3D correlation function
    xis, status = lib.correlation_3dRsd_vec(cosmo, psp, a, mu, beta, s,
                                            len(s), int(use_spline), status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis


def correlation_3dRsd_avgmu(cosmo, a, s, beta, p_of_k_a=None):
    """
    Compute the 3DRsd correlation function averaged over mu at constant s.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        a (float): scale factor.
        s (float or array_like): distance(s) at which to calculate the 3DRsd
                                 correlation function (in Mpc).
        beta (float): growth rate divided by galaxy bias.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`). If `None`, the non-linear matter
            power spectrum stored in `cosmo` will be used.

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
    if isinstance(s, float) or isinstance(s, int):
        scalar = True
        s = np.array([s, ])

    # Call 3D correlation function
    xis, status = lib.correlation_3dRsd_avgmu_vec(cosmo, psp, a, beta, s,
                                                  len(s), status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis


def correlation_pi_sigma(cosmo, a, beta, pi, sig,
                         use_spline=True, p_of_k_a=None):
    """
    Compute the 3DRsd correlation in pi-sigma space.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        a (float): scale factor.
        pi (float): distance times cosine of the angle (in Mpc).
        sig (float or array-like): distance(s) times sine of the angle
                                   (in Mpc).
        beta (float): growth rate divided by galaxy bias.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, `str` or None): 3D Power spectrum
            to integrate. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`). If `None`, the non-linear matter
            power spectrum stored in `cosmo` will be used.

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
    if isinstance(sig, float) or isinstance(sig, int):
        scalar = True
        sig = np.array([sig, ])

    # Call 3D correlation function
    xis, status = lib.correlation_pi_sigma_vec(cosmo, psp, a, beta, pi, sig,
                                               len(sig), int(use_spline),
                                               status)
    check(status, cosmo_in)
    if scalar:
        return xis[0]
    return xis
