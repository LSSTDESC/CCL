"""Written by Paul Rogozenski (paulrogozenski@arizona.edu),
 implementing the FKEM non-limber integration method of the N5K challenge
 detailed in this paper: https://arxiv.org/pdf/1911.11947.pdf .
We utilize a modified generalized version of FFTLog
 (https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf)
 to compute integrals over spherical bessel functions
"""
__all__ = ("_nonlimber_FKEM",)


import numpy as np
from .. import lib, check
from ..pyutils import integ_types
from pyccl.pyutils import _fftlog_transform_general
import pyccl as ccl
from .. import CCLWarning, warnings


def _get_general_params(b):
    """Get the parameters for the generalized FFTLog transform
    Args:
        b (int): Bessel function derivative order
            (corresponds to CCL bessel_deriv_type)
    Returns:
        nu (float): bias parameter for FFTLog
        deriv (float): Derivative order of Bessel Function
            for integrand
        plaw (float): Power-law index of k-factor in integrand
    """
    # nu set to 1.01 to avoid singularity at nu=1, as in FFTLog paper
    nu = 1.01
    deriv = 0.0
    plaw = 0.0
    if b < 0:
        plaw = -2.0
    if b <= 0:
        deriv = 0
    else:
        deriv = b
    return nu, deriv, plaw


def _get_k_common(ks_1, ks_2):
    """Get the common k array for the two tracers to perform the final
    chi-integral integration over.
    Args:
        ks_1 (array): List of wavenumbers at which to evaluate
            the chi integrands for tracer 1.
        ks_2 (array): List of wavenumbers at which to evaluate
            the chi integrands for tracer 2.
    Returns:
        k_common (array): Common wavenumbers at which to evaluate the
            chi-integral integration for all tracers.
    """
    all_ks = [np.asarray(k, dtype=float) for k in (*ks_1, *ks_2)]

    k_mins = [k[np.isfinite(k) & (k > 0.0)].min() for k in all_ks]
    k_maxs = [k[np.isfinite(k) & (k > 0.0)].max() for k in all_ks]

    k_min = max(k_mins)   # overlap lower bound
    k_max = min(k_maxs)   # overlap upper bound

    n_k = max(len(k) for k in all_ks)
    return np.logspace(np.log10(k_min), np.log10(k_max), n_k)


def _pad_chi_grid(chi):
    """Extend an FFTLog grid without changing its interior sampling."""
    dlnchi = np.log(chi[-1] / chi[0]) / (len(chi) - 1)
    # A decade on each side extends the low-k coverage and separates the
    # periodic copies of the kernel. Bound the cost for narrow input ranges.
    npad = min(len(chi), int(np.ceil(np.log(10.) / dlnchi)))
    lower = chi[0] * np.exp(dlnchi * np.arange(-npad, 0))
    upper = chi[-1] * np.exp(dlnchi * np.arange(1, npad + 1))
    return np.concatenate((lower, chi, upper)), npad


def _chi_integrands(cosmo, clt,
                    Nchi, chi_min, chi_max,
                    ell, k_low,
                    chi_logspace_arr, status):
    """
    Computes the chi integrands for FKEM using FFTLog
    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        clt (:class:`~pyccl.tracers.TracerCollection`): TracerCollection
            object for tracer.
        Nchi (int): Number of values of the comoving distance
            over which FKEM will evaluate the radial kernels.
        chi_min (float): Minimum comoving distance used by FKEM to sample
            the radial kernels.
        chi_max (float): Maximum comoving distance used by FKEM to sample
            the radial kernels.
        ell (float): Angular multipole at which to evaluate the integrand.
        k_low (float): large-scale wavenumber for scale-dependence
            kernel approximation.
        chi_logspace_arr (array): Array of comoving distances in log-space
            over which FKEM evaluates the radial kernels.
        status (int): Status flag for error checking.
    Returns:
        k (array): Wavenumbers at which to evaluate the full integral.
        fks (array): Chi integrands for each tracer.
        transfers (array): Transfer functions for each tracer.
    """

    kernels, chis = clt.get_kernel()
    bessels = clt.get_bessel_derivative()
    a_arr = ccl.scale_factor_of_chi(cosmo, chi_logspace_arr)
    growfac_arr = ccl.growth_factor(cosmo, a_arr)
    avg_as = clt.get_avg_weighted_a()

    chi_fft, npad = _pad_chi_grid(chi_logspace_arr)
    fks = np.zeros((len(kernels), len(chi_fft)))
    transfers = np.zeros_like(fks)
    ks = []
    for i in range(len(kernels)):
        k, fk = clt._get_fkem_fft(
            clt._trc[i], Nchi, chi_min, chi_max, ell, cosmo,
        )

        if (k is None) or (fk is None):
            transfer_low = np.array(
                clt.get_transfer(np.log(k_low), a_arr)
            )
            transfer_avg = clt.get_transfer(np.log(k_low), avg_as[i])
            # check no zeros in transfer functions
            if np.any(transfer_avg == 0.0):
                raise ZeroDivisionError(
                    "Zero transfer function encountered in FKEM chi "
                    "integrand calculation. "
                    "Setting integrand to zero."
                )

            kernel = np.interp(
                chi_logspace_arr, chis[i], kernels[i], left=0., right=0.
            )
            # transfer function approximation for the case
            # when it's inseperable in k and a
            # exact for seperable transfer functions
            fchi_arr = (
                kernel
                * chi_logspace_arr
                * growfac_arr
                * transfer_low[i]
                / transfer_avg[i]
            )
            # calls to fftlog to perform integration over chi integrals
            nu, deriv, plaw = _get_general_params(bessels[i])
            k, fk = _fftlog_transform_general(
                chi_fft,
                np.pad(fchi_arr.flatten(), npad),
                float(ell),
                nu,
                1,
                float(deriv),
                float(plaw),
            )
            check(status, cosmo=cosmo)

        clt._set_fkem_fft(
            clt._trc[i], cosmo, Nchi, chi_min, chi_max, ell, k, fk,
        )
        ks.append(k)
        fks[i] = fk
        transfers[i] = np.array(clt.get_transfer(np.log(k), avg_as[i]))[i]

    return ks, fks, transfers


def _auto_limber_transition_ell(clt1, clt2, cosmo, psp_lin, psp_nonlin,
                                cls_nonlimber_lin, ell,
                                limber_max_error, status):
    """
    Helper function for _nonlimber_FKEM to determine whether the
    Limber transition ell/threshold has been reached for all
    combinations of tracers, not just the total cl,
    to ensure that the non-Limber calculation is accurate up to the
    specified limber_max_error threshold.
    Args:
        clt1 (:class:`~pyccl.tracers.TracerCollection`):
            TracerCollection object for tracer 1.
        clt2 (:class:`~pyccl.tracers.TracerCollection`):
            TracerCollection object for tracer 2.
        cosmo (:class:`~pyccl.core.Cosmology`):
            A Cosmology object.
        psp_lin (:class:`~pyccl.pk2d.Pk2D`):
            Linear power spectrum to use for growth factor scaling.
        psp_nonlin (:class:`~pyccl.pk2d.Pk2D`):
            Non-linear power spectrum to project.
        cls_nonlimber_lin (array):
            Linear non-Limber spectra for each tracer-component pair.
        ell (float):
            Current ell value.
        limber_max_error (float):
            Maximum fractional error for Limber integration.
        status (int):
            Status flag for error checking.
    Returns:
        status (int):
            Updated status flag for error checking.
        is_limber (bool):
            Whether the Limber transition ell/threshold
            has been reached for all combinations of tracers.
    """
    for i in range(len(clt1._trc)):
        for j in range(len(clt2._trc)):
            ti = clt1._trc[i]
            tj = clt2._trc[j]
            t1_temp, status = lib.cl_tracer_collection_t_new(status)
            check(status)
            t2_temp, status = lib.cl_tracer_collection_t_new(status)
            check(status)
            status = lib.add_cl_tracer_to_collection(t1_temp, ti, status)
            check(status)
            status = lib.add_cl_tracer_to_collection(t2_temp, tj, status)
            check(status)

            cl_limber_lin_temp, status = lib.angular_cl_vec_limber(
                cosmo.cosmo,
                t1_temp,
                t2_temp,
                psp_lin,
                [ell],
                integ_types["qag_quad"],
                1,
                status,
            )
            check(status, cosmo=cosmo)
            cl_limber_nonlin_temp, status = lib.angular_cl_vec_limber(
                cosmo.cosmo,
                t1_temp,
                t2_temp,
                psp_nonlin,
                [ell],
                integ_types["qag_quad"],
                1,
                status,
            )
            check(status, cosmo=cosmo)

            cl_temp = (
                cl_limber_nonlin_temp[-1]
                - cl_limber_lin_temp[-1]
                + cls_nonlimber_lin[i, j]
            )
            thresh = cl_temp / cl_limber_nonlin_temp[-1] - 1.0

            lib.cl_tracer_collection_t_free(t1_temp)
            lib.cl_tracer_collection_t_free(t2_temp)

            if np.abs(thresh) >= limber_max_error:
                return status, False
    return status, True


def _nonlimber_FKEM(
        cosmo, clt1, clt2, p_of_k_a,
        ls, l_limber, **params):
    """Performs the FKEM non-Limber integration method
        for angular power spectra.
    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
        clt1 (:class:`~pyccl.tracers.TracerCollection`): TracerCollection
            object for tracer 1.
        clt2 (:class:`~pyccl.tracers.TracerCollection`): TracerCollection
            object for tracer 2.
        p_of_k_a (str or :class:`~pyccl.pk2d.Pk2D`): 3D Power spectrum
            to project. If a string, it must correspond to one of the
            non-linear power spectra stored in `cosmo`
            (e.g. 'delta_matter:delta_matter').
        ls (array): Angular multipole(s) at which to evaluate
            the angular power spectrum.
        l_limber (int or str): Maximum ell for non-Limber calculation.
            If 'auto', the non-Limber integrator
            will determine when to transition to the Limber approximation
            based on `limber_max_error` in `params`.
        params: Additional parameters for the FKEM method:
            - chi_min: Minimum comoving distance used by FKEM to sample
              the tracer radial kernels. Must be greater than zero.
            - Nchi: Number of values of the comoving distance over which
              FKEM will interpolate the radial kernels. Default is
              two times the maximum number of chi samples across the tracers.
            - pk_linear: Linear power spectrum to use for growth factor
              scaling. If a string, it must correspond to one of the
              linear power spectra stored in `cosmo`
              (e.g. 'delta_matter:delta_matter').
            - limber_max_error: Maximum fractional error for
                Limber integration.
    Returns:
        l_limber: Maximum ell for non-Limber calculation.
        cells (numpy.ndarray): Calculated angular power spectra.
        status (int): Error status. 0 if there were no errors.
    """

    kpow = 3
    k_low = 1.0e-5
    cells = []
    kernels_t1, chis_t1 = clt1.get_kernel()
    kernels_t2, chis_t2 = clt2.get_kernel()
    fll_t1 = clt1.get_f_ell(ls)
    fll_t2 = clt2.get_f_ell(ls)
    status = 0

    p_of_k_a_lin = params['pk_linear']
    limber_max_error = params['limber_max_error']
    Nchi = params['Nchi']
    chi_min = params['chi_min']
    if (not (isinstance(p_of_k_a, str) and isinstance(p_of_k_a_lin, str)) and
       not (isinstance(p_of_k_a, ccl.Pk2D)
            and isinstance(p_of_k_a_lin, ccl.Pk2D)
            )):
        warnings.warn(
            "p_of_k_a and p_of_k_a_lin must be of the same "
            "type: a str in cosmo or a Pk2D object. "
            "Defaulting to Limber calculation. ",
            category=CCLWarning, importance='high')
        return -1, np.array([]), status

    # check valid inputs, set defaults if necessary
    if Nchi is None or not isinstance(Nchi, int) or Nchi <= 0:
        warnings.warn("Nchi must be a positive integer. "
                      "Setting to match tracer with"
                      " large chi samples x 2.",
                      category=CCLWarning,
                      importance='high'
                      )
        # We put factor of 2 to help reduce spikes when comparing
        # the decomposed and direct integration approaches to order 1e-6
        # for at least a simple Gaussian density kernel.
        # May need to go higher for different cases.
        Nchi = 2 * max(max(len(i) for i in chis_t1),
                       max(len(i) for i in chis_t2))
    if chi_min is None or not isinstance(chi_min, (float)) or chi_min <= 0.0:
        warnings.warn("chi_min must be greater than zero."
                      "Setting to default 1e-6 Mpc.",
                      category=CCLWarning,
                      importance='high'
                      )
        chi_min = 1.0e-6
    if limber_max_error is None or not isinstance(limber_max_error, (float)) \
            or limber_max_error <= 0.0:
        warnings.warn("limber_max_error must be greater than zero."
                      "Setting to default 0.01.",
                      category=CCLWarning,
                      importance='high'
                      )
        limber_max_error = 0.01

    if l_limber is None \
            or (not isinstance(l_limber, str)
                and not isinstance(l_limber, int)) \
            or ((isinstance(l_limber, str) and l_limber != 'auto')
                or (isinstance(l_limber, int) and l_limber <= 0)):
        warnings.warn("l_limber must be a positive integer or 'auto'. "
                      "Setting to 'auto'.",
                      category=CCLWarning,
                      importance='high'
                      )
        l_limber = 'auto'

    psp_lin = cosmo.parse_pk2d(p_of_k_a_lin, is_linear=True)
    psp_nonlin = cosmo.parse_pk2d(p_of_k_a, is_linear=False)

    t1, status = lib.cl_tracer_collection_t_new(status)
    check(status)
    t2, status = lib.cl_tracer_collection_t_new(status)
    check(status)
    for t in clt1._trc:
        status = lib.add_cl_tracer_to_collection(t1, t, status)
        check(status)
    for t in clt2._trc:
        status = lib.add_cl_tracer_to_collection(t2, t, status)
        check(status)
    if isinstance(p_of_k_a_lin, ccl.Pk2D):
        pk = p_of_k_a_lin
    else:
        pk = cosmo.get_linear_power(name=p_of_k_a_lin)

    max_chis_t1 = np.max([np.max(i) for i in chis_t1])
    max_chis_t2 = np.max([np.max(i) for i in chis_t2])
    chi_max = np.max([max_chis_t1, max_chis_t2])

    chi_logspace_arr = np.logspace(
        np.log10(chi_min), np.log10(chi_max), num=Nchi, endpoint=True
    )

    for el in range(len(ls)):
        ell = ls[el]
        status = 0
        cl_limber_lin, status = lib.angular_cl_vec_limber(
            cosmo.cosmo,
            t1,
            t2,
            psp_lin,
            [ell],
            integ_types["qag_quad"],
            1,
            status,
        )
        check(status, cosmo=cosmo)
        cl_limber_nonlin, status = lib.angular_cl_vec_limber(
            cosmo.cosmo,
            t1,
            t2,
            psp_nonlin,
            [ell],
            integ_types["qag_quad"],
            1,
            status,
        )
        check(status, cosmo=cosmo)

        # chi-integral integrand splines
        ks, fks_1, transfers_t1 = _chi_integrands(
            cosmo, clt1,
            Nchi, chi_min, chi_max,
            ell, k_low,
            chi_logspace_arr, status
        )

        if clt1 != clt2:
            ks_2, fks_2, transfers_t2 = _chi_integrands(
                cosmo, clt2,
                Nchi, chi_min, chi_max,
                ell, k_low,
                chi_logspace_arr, status
            )
        else:
            fks_2 = fks_1
            transfers_t2 = transfers_t1
            ks_2 = ks
        # Each component pair has its own overlap and quadrature. This keeps
        # adding tracer components from changing another pair's integration.
        cls_nonlimber_lin = np.zeros((len(ks), len(ks_2)))
        for i, k1 in enumerate(ks):
            for j, k2 in enumerate(ks_2):
                k = _get_k_common([k1], [k2])
                dlnk = np.log(k[-1] / k[0]) / (len(k) - 1)
                integrand = (
                    np.interp(k, k1, fks_1[i])
                    * np.interp(k, k1, transfers_t1[i])
                    * np.interp(k, k2, fks_2[j])
                    * np.interp(k, k2, transfers_t2[j])
                    * k**kpow * pk(k, 1.0, cosmo)
                )
                cls_nonlimber_lin[i, j] = (
                    np.sum(integrand) * dlnk * 2.0 / np.pi
                    * fll_t1[i, el] * fll_t2[j, el]
                )
        # append the final cl calculation to the returned array
        # see whether the Limber transition ell/threshold has been reached
        # and check whether to continue to higher ells
        cells.append(
            cl_limber_nonlin[-1] - cl_limber_lin[-1]
            + np.sum(cls_nonlimber_lin)
        )

        if (type(l_limber) is not str and ell >= l_limber):
            l_limber = ell
            break
        if l_limber == "auto":
            # need to check each individual combination of tracers
            # to ensure all have reached the limber threshold,
            # not just the total cl
            # if any of the individual combinations of tracers
            # have not reached the limber threshold,
            # we need to continue to higher ells
            # this is because the total cl can be dominated
            # by one combination of tracers that has reached
            # the limber threshold
            status, is_limber = _auto_limber_transition_ell(
                clt1, clt2, cosmo, psp_lin,
                psp_nonlin, cls_nonlimber_lin, ell,
                limber_max_error, status
            )
            if is_limber:
                l_limber = ell
                break
    if type(l_limber) is str:
        l_limber = ls[-1]
    if False in np.isfinite(cells):
        status = 1
    lib.cl_tracer_collection_t_free(t1)
    lib.cl_tracer_collection_t_free(t2)
    return l_limber, np.array(cells), status
