__all__ = ("_nonlimber_FKEM",)
"""Written by Paul Rogozenski (paulrogozenski@arizona.edu),
 implementing the FKEM non-limber integration method of the N5K challenge
 detailed in this paper: https://arxiv.org/pdf/1911.11947.pdf .
We utilize a modified generalized version of FFTLog
 (https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf)
 to compute integrals over spherical bessel functions
"""
import numpy as np
from . import lib, check
from .pyutils import integ_types
from scipy.interpolate import interp1d
from pyccl.pyutils import _fftlog_transform_general
import pyccl as ccl
from . import CCLWarning, warnings


def _get_general_params(b):
    nu = 1.51
    nu2 = 0.51
    deriv = 0.0
    plaw = 0.0
    best_nu = nu
    if b < 0:
        plaw = -2.0
        best_nu = nu2

    if b <= 0:
        deriv = 0
    else:
        deriv = b

    return best_nu, deriv, plaw


def _nonlimber_FKEM(
        cosmo, clt1, clt2, p_of_k_a,
        ls, l_limber, **params):
    """clt1, clt2 are lists of tracer in a tracer object
    cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
    psp non-linear power spectrum
    l_limber max ell for non-limber calculation
    ell_use ells at which we calculate the non-limber integrals
    """

    kpow = 3
    k_low = 1.0e-5
    cells = np.zeros(len(ls))
    kernels_t1, chis_t1 = clt1.get_kernel()
    bessels_t1 = clt1.get_bessel_derivative()
    kernels_t2, chis_t2 = clt2.get_kernel()
    bessels_t2 = clt2.get_bessel_derivative()
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
    min_chis_t1 = np.min([np.min(i) for i in chis_t1])
    min_chis_t2 = np.min([np.min(i) for i in chis_t2])
    max_chis_t1 = np.max([np.max(i) for i in chis_t1])
    max_chis_t2 = np.max([np.max(i) for i in chis_t2])
    if chi_min is None:
        chi_min = np.min([min_chis_t1, min_chis_t2])
    chi_max = np.max([max_chis_t1, max_chis_t2])
    if Nchi is None:
        Nchi = min(min(len(i) for i in chis_t1),
                   min(len(i) for i in chis_t2))
    """zero chi_min will result in a divide-by-zero error.
    If it is zero, we set it to something very small
    """
    if chi_min == 0.0:
        chi_min = 1.0e-6
    chi_logspace_arr = np.logspace(
        np.log10(chi_min), np.log10(chi_max), num=Nchi, endpoint=True
    )
    dlnr = np.log(chi_max / chi_min) / (Nchi - 1.0)
    cells = []

    a_arr = ccl.scale_factor_of_chi(cosmo, chi_logspace_arr)
    growfac_arr = ccl.growth_factor(cosmo, a_arr)
    avg_a1s = clt1.get_avg_weighted_a()
    avg_a2s = clt2.get_avg_weighted_a()
    for el in range(len(ls)):
        ell = ls[el]
        cls_nonlimber_lin = 0.0
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

        """transfer function approximation for the case
        when it's inseperable in k and a
        exact for seperable transfer functions
        """

        # chi-integral integrand splines

        fks_1 = np.zeros((len(kernels_t1), Nchi))
        transfers_t1 = np.zeros((len(kernels_t1), Nchi))
        for i in range(len(kernels_t1)):
            k, fk1 = clt1._get_fkem_fft(
                clt1._trc[i], Nchi, chi_min, chi_max, ell
            )

            if (k is None) or (fk1 is None):
                transfer_t1_low = np.array(
                    clt1.get_transfer(np.log(k_low), a_arr)
                )
                transfer_t1_avg = clt1.get_transfer(np.log(k_low), avg_a1s[i])

                fchi1_interp = interp1d(
                    chis_t1[i], kernels_t1[i], fill_value="extrapolate"
                )
                fchi1_arr = (
                    fchi1_interp(chi_logspace_arr)
                    * chi_logspace_arr
                    * growfac_arr
                    * transfer_t1_low[i]
                    / transfer_t1_avg[i]
                )
                # calls to fftlog to perform integration over chi integrals
                nu, deriv, plaw = _get_general_params(bessels_t1[i])
                k, fk1 = _fftlog_transform_general(
                    chi_logspace_arr,
                    fchi1_arr.flatten(),
                    float(ell),
                    nu,
                    1,
                    float(deriv),
                    float(plaw),
                )
                check(status, cosmo=cosmo)

                clt1._set_fkem_fft(
                    clt1._trc[i], Nchi, chi_min, chi_max, ell, k, fk1
                )
            fks_1[i] = fk1
        transfers_t1 = np.array(clt1.get_transfer(np.log(k), avg_a1s[i]))
        """transfer function approximation for the case
        when it's inseperable in k and a
        exact for seperable transfer functions
        """

        # chi-integral integrand splines

        fks_2 = np.zeros((len(kernels_t2), Nchi))
        transfers_t2 = np.zeros((len(kernels_t2), Nchi))
        if clt1 != clt2:
            for j in range(len(kernels_t2)):
                k, fk2 = clt2._get_fkem_fft(
                    clt2._trc[j], Nchi, chi_min, chi_max, ell
                )
                if ((k is None) or (fk2 is None)):
                    transfer_t2_low = np.array(
                        clt2.get_transfer(np.log(k_low), a_arr)
                    )
                    transfer_t2_avg = clt2.get_transfer(
                        np.log(k_low), avg_a2s[j]
                    )
                    fchi2_interp = interp1d(
                        chis_t2[j], kernels_t2[j],
                        fill_value="extrapolate"
                    )
                    fchi2_arr = (
                        fchi2_interp(chi_logspace_arr)
                        * chi_logspace_arr
                        * growfac_arr
                        * transfer_t2_low[j]
                        / transfer_t2_avg[j]
                    )
                    # calls to fftlog to perform integration over chi integrals
                    nu2, deriv2, plaw2 = _get_general_params(bessels_t2[j])
                    k, fk2 = _fftlog_transform_general(
                        chi_logspace_arr,
                        fchi2_arr.flatten(),
                        float(ell),
                        nu2,
                        1,
                        float(deriv2),
                        float(plaw2),
                    )
                    check(status, cosmo=cosmo)

                    clt2._set_fkem_fft(
                        clt2._trc[j], Nchi, chi_min, chi_max, ell, k, fk2
                    )
                fks_2[j] = fk2
            transfers_t2 = np.array(clt2.get_transfer(np.log(k), avg_a2s[j]))
        else:
            fks_2 = fks_1
            transfers_t2 = transfers_t1

        for i in range(len(kernels_t1)):
            for j in range(len(kernels_t2)):
                cls_nonlimber_lin += (
                    np.sum(
                        fks_1[i]
                        * transfers_t1[i]
                        * fks_2[j]
                        * transfers_t2[j]
                        * k**kpow
                        * pk(k, 1.0, cosmo)
                    )
                    * dlnr
                    * 2.0
                    / np.pi
                    * fll_t1[i][el]
                    * fll_t2[j][el]
                )

        # append the final cl calculation to the returned array
        # and check whether to continue to higher ells
        cells.append(
            cl_limber_nonlin[-1] - cl_limber_lin[-1] + cls_nonlimber_lin
        )
        if (
            np.abs(cells[-1] / cl_limber_nonlin[-1] - 1) < limber_max_error
            and l_limber == "auto"
        ) or (type(l_limber) is not str and ell >= l_limber):
            l_limber = ell
            break

    if type(l_limber) is str:
        l_limber = ls[-1]
    if False in np.isfinite(cells):
        status = 1
    return l_limber, cells, status
