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
from .pyutils import resample_array, _fftlog_transform
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

    .. note::

        For scales smaller than :math:`\\sim 0.1^{\\circ}`, the input power
        spectrum should be sampled to sufficienly high :math:`\\ell` to ensure
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
    if isinstance(s, (int, float)):
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
    if isinstance(s, (int, float)):
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
    if isinstance(s, (int, float)):
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
    if isinstance(sig, (int, float)):
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


def correlation_ab(cosmo, r_p: np.ndarray, z: np.ndarray,
                   dndz: np.ndarray = None, dndz2: np.ndarray = None,
                   p_of_k_a=None,
                   type: str = 'gg',
                   precision_fftlog: dict = None):
    """Computes :math:`w_{ab}`.
    .. math::
        \\w_{ab}(r_p)=\\int dz\\,(W)(z)
        \\int\\frac{dk k}{2\\pi^2}\\,P(k,a)\\,J_n(k r_p)

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            r_p (float or array-like): Projected radial separation where the
                correlation function will be evaluated.
            z (array-like): Redshift values where the redshift distribution
                has been evaluated. If float, the window function is set to
                a delta function.
            dndz (array-like): Redshift distribution to be used when computing
                the window function (see e.g. Eq. (3.11) in (1811.09598)).
            dndz2 (array-like or None): Redshift distribution corresponding to
                sample `b` of tracers to be correlated. If None, tracers `a`
                and `b` are the same. Default value: None.
            p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): 3D Power spectrum to
                integrate.
            type (string): Type of `ab` correlation. This changes the Bessel
                function in the integral above. Choices: 'gg' (`J_0`),
                'g+' (`J_2`), '++' (`J_0+J_4`).
            precision_fftlog (dict): Dictionary containing the precision
                parameters used by FFTLog to compute Hankel transforms. The
                dictionary should contain the keys: `padding_lo_fftlog`,
                `padding_lo_fftlog`, `padding_lo_extra`, `padding_hi_fftlog`,
                `padding_hi_extra`, `n_per_decade`, `extrapol`,
                `plaw_fourier`. Default values are 0.01, 0.1, 10., 10.,
                100, 'linx_liny' and -1.5. For more information look at
                `pyccl.halos.profiles.HaloProfile.update_precision_fftlog`.

        Returns:
            array-like: Value(s) of the correlation function at the input
            projected radial separation(s).
    """
    # TODO: add functionality for p_of_k_a=None.
    # TODO: add some unit tests for Value Errors.
    if dndz2 is None:
        dndz2 = dndz
    if dndz is None and np.iterable(z):
        raise ValueError('Cannot have iterable redshift array while '
                         'dndz is None.')
    a = 1/(1+z)
    a = np.atleast_1d(a)
    if type == 'gg':
        xi = np.array([_fftlog_wrap(r_p, a_, cosmo, p_of_k_a, n=0,
                                    precision_fftlog=precision_fftlog)
                      for a_ in a])
    elif type == 'g+':
        xi = np.array([_fftlog_wrap(r_p, a_, cosmo, p_of_k_a, n=2,
                                    precision_fftlog=precision_fftlog)
                      for a_ in a])
    elif type == '++':
        xi = np.array([(_fftlog_wrap(r_p, a_, cosmo, p_of_k_a, n=0,
                                     precision_fftlog=precision_fftlog) +
                       _fftlog_wrap(r_p, a_, cosmo, p_of_k_a, n=4,
                                    precision_fftlog=precision_fftlog))
                      for a_ in a])
    else:
        raise ValueError('Correlation type not recognised. Accepted'
                         'values: "gg", "g+", "++".')
    if np.iterable(z):
        wz = dndz*dndz2*(1+z)**2/(
            cosmo.comoving_radial_distance(a)**2. *
            _compute_dchi_da_num(a, cosmo))
        wz[np.isnan(wz)] = 0  # For values at z=0
        wz /= np.trapz(wz, z)
        if np.ndim(xi) == 2:
            wz = wz.reshape((len(z), 1))
        w = np.trapz(wz*xi, z, axis=0)
    else:
        w = xi[0]
    return w


def _fftlog_wrap(r_p: np.ndarray, a: float, cosmo, p_of_k_a=None,
                 precision_fftlog: dict = None,
                 n=0, large_padding=True):
    '''Computes the integral
       .. math::
          \\xi(r_p,a)=\\int\\frac{dk k}{2\\pi^2}\\,P(k,a)\\,J_n(k r_p)
       Adapted from pyccl.halos.profiles.HaloProfile._fftlog_wrap.
    '''
    # Note: The _fftlog_transform solves for
    # f(r) = \int 4 pi k^2 f(k) j_l(kr) dk
    # To compute the integral shown above we use _fftlog_transform
    # and set fk = rp^1/2 P(k,a:float)/(2^5/2 pi^5/2 k^1/2).
    if precision_fftlog is None:
        precision_fftlog = {'padding_lo_fftlog': 0.01,
                            'padding_lo_extra': 0.1,
                            'padding_hi_fftlog': 10.,
                            'padding_hi_extra': 10.,
                            'n_per_decade': 100,
                            'extrapol': 'linx_liny',
                            'plaw_fourier': -1.5}
    else:
        for key in ['padding_lo_fftlog', 'padding_lo_extra',
                    'padding_hi_fftlog', 'padding_hi_extra', 'n_per_decade',
                    'extrapol', 'plaw_fourier']:
            if key not in precision_fftlog.keys():
                raise ValueError('Error.')
    l = n - 1 / 2

    r_use = np.atleast_1d(r_p)
    lr_use = np.log(r_use)

    # k-range to be used with FFTLog and its sampling.
    if large_padding:
        k_min = precision_fftlog['padding_lo_fftlog'] * np.amin(r_use)
        k_max = precision_fftlog['padding_hi_fftlog'] * np.amax(r_use)
    else:
        k_min = precision_fftlog['padding_lo_extra'] * np.amin(r_use)
        k_max = precision_fftlog['padding_hi_extra'] * np.amax(r_use)
    n_k = (int(np.log10(k_max / k_min)) *
           precision_fftlog['n_per_decade'])
    k_arr = np.geomspace(k_min, k_max, n_k)  # Array to be used by FFTLog.

    # What needs to go in _fftlog_transform to get out \xi_gg.
    fk = p_of_k_a.eval(k_arr, a, cosmo)/((2*np.pi)**(5./2)*k_arr**(1/2))
    r_arr, xi_fourier = _fftlog_transform(k_arr, fk, 3, l,
                                          precision_fftlog['plaw_fourier'])
    lr_arr = np.log(r_arr)

    # Resample into input k values.
    xi_out = resample_array(lr_arr, xi_fourier, lr_use,
                            precision_fftlog['extrapol'],
                            precision_fftlog['extrapol'],
                            0, 0)
    # Multiply to get the correct output.
    xi_out *= (2*np.pi)**3*np.sqrt(r_use)

    if np.ndim(r_p) == 0:
        xi_out = np.squeeze(xi_out, axis=-1)
    return xi_out


def _compute_dchi_da_num(a, cosmo, a_step=0.01):
    a_use = np.atleast_1d(a)
    # If the a-array has values higher than 1, throw an error.
    if len(a_use[a_use > 1]) > 0:
        raise ValueError('Cannot accept scale factor larger than 1.')
    # If the a-array has values that go beyond 1 when the step size is added
    # reduce the step size until that doesn't happen anymore and
    # throw a warning.
    if len(a_use[(a_use + a_step > 1) & (a_use < 1)] > 0):
        while len(a_use[(a_use + a_step > 1) & (a_use < 1)] > 0):
            a_step /= 2.
        warnings.warn('Step size causes scale factor > 1. Using step=%f'
                      % a_step)
    dxda = np.empty(a_use.shape)
    # Compute for all values of a that do not go above 1.
    dxda[a_use < 1] = np.array([
        (cosmo.comoving_radial_distance(a + a_step) -
         cosmo.comoving_radial_distance(a - a_step)) / (2 * a_step)
        for a in a_use[a_use < 1]])
    # For the values that go above 1,
    # use a constant value that is very close to 1.
    dxda[a_use == 1] = (cosmo.comoving_radial_distance(0.9999 + 0.00001) -
                        cosmo.comoving_radial_distance(
        0.9999 - 0.00001)) / (2 * 0.00001)
    return dxda
