__all__ = ("CorrelationMethods", "CorrelationTypes", "correlation",
           "correlation_3d", "correlation_multipole", "correlation_3dRsd",
           "correlation_3dRsd_avgmu", "correlation_pi_sigma",
           "correlation_ab")

from enum import Enum
import numpy as np
from .pyutils import resample_array, _fftlog_transform
from . import DEFAULT_POWER_SPECTRUM, check, lib, physical_constants


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
        raise ValueError(f"Invalud correlation type {type}.")

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


def correlation_ab(cosmo, *, r_p: np.ndarray, z: np.ndarray,
                   dndz: np.ndarray = None, dndz2: np.ndarray = None,
                   p_of_k_a=None,
                   type: str = 'gg',
                   precision_fftlog: dict = None):
    """
    Computes :math:`w_{ab}`.
    .. math::
        \\w_{ab}(r_p)=\\int dz\\,(W)(z)
        \\int\\frac{dk k}{2\\pi^2}\\,P(k,a)\\,J_n(k r_p)

        Args:
            cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
            r_p (float or array-like): Projected radial separation where the
                correlation function will be evaluated.
            z (float or array-like): Redshift values where the redshift 
                distribution has been evaluated. If float, the window function
                is set to a delta function.
            dndz (array-like): Redshift distribution to be used when computing
                the window function (see e.g. Eq. (3.11) in (1811.09598)).
            dndz2 (array-like or None): Redshift distribution corresponding to
                sample `b` of tracers to be correlated. If None, tracers `a`
                and `b` are the same. Default value: None.
            p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`): 3D Power spectrum
                to integrate. If a string, it must correspond to one of the
                non-linear power spectra stored in `cosmo` (e.g.
                `'delta_matter:delta_matter'`).
            type (string): Type of `ab` correlation. This changes the Bessel
                function in the integral above. Choices: 'gg' (`J_0`),
                'g+' (`J_2`), '++' (`J_0+J_4`).
            precision_fftlog (dict): Dictionary containing the precision
                parameters used by FFTLog to compute Hankel transforms. The
                dictionary should contain the keys:
                `padding_lo_fftlog`, `padding_hi_fftlog`,
                `n_per_decade`, `extrapol`, `plaw_fourier`.
                Default values are 0.01, 0.1, 10., 10.,
                100, 'linx_liny' and -1.5. For more information look at
                `pyccl.halos.profiles.HaloProfile.update_precision_fftlog`.

        Returns:
            array-like: Value(s) of the correlation function at the input
            projected radial separation(s).
    """ # noqa
    cosmo.compute_nonlin_power()
    cosmo_in = cosmo
    psp = cosmo_in.parse_pk(p_of_k_a)

    if dndz2 is None:
        dndz2 = dndz
    if dndz is None and np.iterable(z):
        raise ValueError('Cannot have iterable redshift array while '
                         'dndz is None.')
    a = 1/(1+z)
    a = np.atleast_1d(a)
    if type == 'gg':
        xi = np.array([_fftlog_wrap(r_p, a_, psp, n=0,
                                    precision_fftlog=precision_fftlog)
                      for a_ in a])
    elif type == 'g+':
        xi = np.array([_fftlog_wrap(r_p, a_, psp, n=2,
                                    precision_fftlog=precision_fftlog)
                      for a_ in a])
    elif type == '++':
        xi = np.array([(_fftlog_wrap(r_p, a_, psp, n=0,
                                     precision_fftlog=precision_fftlog) +
                       _fftlog_wrap(r_p, a_, psp, n=4,
                                    precision_fftlog=precision_fftlog))
                      for a_ in a])
    else:
        raise ValueError('Correlation type not recognised. Accepted'
                         'values: "gg", "g+", "++".')
    if np.iterable(z):
        speed_of_light_kmps = physical_constants.CLIGHT * 1e-3
        dchi_dz = speed_of_light_kmps/(cosmo_in.h_over_h0(a)*cosmo_in['H0'])
        denominator = cosmo_in.comoving_radial_distance(a) ** 2. * dchi_dz
        wz = np.divide(dndz * dndz2,
                       denominator,
                       where=denominator > 0)
        wz /= np.trapz(wz, z)
        if np.ndim(xi) == 2:
            wz = wz.reshape((len(z), 1))
        w = np.trapz(wz*xi, z, axis=0)
    else:
        w = xi[0]
    return w


def _fftlog_wrap(r_p: np.ndarray, a: float, p_of_k_a,
                 precision_fftlog: dict = None,
                 n=0):
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
                            'padding_hi_fftlog': 10.,
                            'n_per_decade': 100,
                            'extrapol': 'linx_liny',
                            'plaw_fourier': -1.5}
    else:
        for key in ['padding_lo_fftlog', 'padding_hi_fftlog', 'n_per_decade',
                    'extrapol', 'plaw_fourier']:
            if key not in precision_fftlog.keys():
                raise ValueError('Error.')
    l = n - 1 / 2

    r_use = np.atleast_1d(r_p)
    lr_use = np.log(r_use)

    # k-range to be used with FFTLog and its sampling.
    k_min = precision_fftlog['padding_lo_fftlog'] * np.amin(r_use)
    k_max = precision_fftlog['padding_hi_fftlog'] * np.amax(r_use)
    n_k = (int(np.log10(k_max / k_min)) *
           precision_fftlog['n_per_decade'])
    k_arr = np.geomspace(k_min, k_max, n_k)  # Array to be used by FFTLog.

    # What needs to go in _fftlog_transform to get out \xi_gg.
    fk = p_of_k_a(k_arr, a)/((2*np.pi)**(5./2)*k_arr**(1/2))
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
