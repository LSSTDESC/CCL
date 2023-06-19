__all__ = ("nonlimber_FKEM",)
'''Written by Paul Rogozenski (paulrogozenski@arizona.edu),
 implementing the FKEM non-limber integration method of the N5K challenge
 detailed in this paper: https://arxiv.org/pdf/1911.11947.pdf .
We utilize a modified generalized version of FFTLog
 (https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf)
 to compute integrals over spherical bessel functions
'''
import numpy as np
from . import lib
from .pyutils import integ_types
from scipy.interpolate import interp1d
from pyccl.pyutils import _fftlog_transform_general
import pyccl as ccl


def get_general_params(b):
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


def get_average_a(a_arr, dndz):
    z_arr = 1./(a_arr) - 1
    dz = (z_arr[-1] - z_arr[0])/(len(z_arr) - 1)
    new_arr = []
    for i in range(len(dndz)):
        if dndz[i] != 0:
            new_arr.append(dz * z_arr[i] * dndz[i])
    z_mean = np.sum(np.array(new_arr))
    return 1./(1. + z_mean)


def nonlimber_FKEM(cosmo, clt1, clt2, p_of_k_a, ls,
                   l_limber, limber_max_error):
    '''clt1, clt2 are lists of tracer in a tracer object
    cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
    psp non-linear power spectrum
    l_limber max ell for non-limber calculation
    ell_use ells at which we calculate the non-limber integrals
    '''

    kpow = 3
    k_low = 1.e-3
    cells = np.zeros(len(ls))
    kernels_t1, chis_t1 = clt1.get_kernel()
    bessels_t1 = clt1.get_bessel_derivative()
    kernels_t2, chis_t2 = clt2.get_kernel()
    bessels_t2 = clt2.get_bessel_derivative()
    fll_t1 = clt1.get_f_ell(ls)
    fll_t2 = clt2.get_f_ell(ls)
    psp_lin = cosmo.parse_pk2d(p_of_k_a, is_linear=True)
    psp_nonlin = cosmo.parse_pk2d(p_of_k_a, is_linear=False)
    status = 0
    t1, status = lib.cl_tracer_collection_t_new(status)
    t2, status = lib.cl_tracer_collection_t_new(status)
    for t in clt1._trc:
        status = lib.add_cl_tracer_to_collection(t1, t, status)
    for t in clt2._trc:
        status = lib.add_cl_tracer_to_collection(t2, t, status)
    pk = cosmo.get_linear_power(name=p_of_k_a)

    chi_min = np.max([np.min(chis_t1), np.min(chis_t2)])
    chi_max = np.min([np.max(chis_t1), np.max(chis_t2)])
    Nchi = min(min(len(i) for i in chis_t1), min(len(i) for i in chis_t2))
    '''zero chi_min will result in a divide-by-zero error.
    If it is zero, we set it to something very small
    '''
    if chi_min == 0.0:
        chi_min = 1.e-6
    chi_logspace_arr = np.logspace(np.log10(chi_min), np.log10(chi_max),
                                   num=Nchi, endpoint=True)
    dlnr = np.log(chi_max/chi_min)/(Nchi - 1.)
    a_arr = ccl.scale_factor_of_chi(cosmo, chi_logspace_arr)
    growfac_arr = ccl.growth_factor(cosmo, a_arr)

    '''transfer function approximation for the case
    when it's inseperable in k and a
    exact for seperable transfer functions
    '''
    transfer_t1_low = np.array(clt1.get_transfer(np.log(k_low), a_arr))
    transfer_t2_low = np.array(clt2.get_transfer(np.log(k_low), a_arr))
    avg_a1 = get_average_a(a_arr, clt1.get_dndz(1./a_arr - 1))
    avg_a2 = get_average_a(a_arr, clt2.get_dndz(1./a_arr - 1))
    transfer_t1_avg = clt1.get_transfer(np.log(k_low), avg_a1)
    transfer_t2_avg = clt2.get_transfer(np.log(k_low), avg_a2)

    # chi-integral integrand splines
    fchi1_arr = np.zeros((len(kernels_t1), len(chi_logspace_arr)))
    for i in range(len(kernels_t1)):
        fchi1_interp = interp1d(chis_t1[i], kernels_t1[i],
                                fill_value='extrapolate')
        fchi1_arr[i] = (fchi1_interp(chi_logspace_arr)
                        * chi_logspace_arr*growfac_arr
                        * transfer_t1_low[i]/transfer_t1_avg[i])

    fchi2_arr = np.zeros((len(kernels_t2), len(chi_logspace_arr)))
    for j in range(len(kernels_t2)):
        fchi2_interp = interp1d(chis_t2[j], kernels_t2[j],
                                fill_value='extrapolate')
        fchi2_arr[j] = (fchi2_interp(chi_logspace_arr)
                        * chi_logspace_arr * growfac_arr
                        * transfer_t2_low[j]/transfer_t2_avg[j])

    cells = []
    for el in range(len(ls)):
        ell = ls[el]
        cls_nonlimber_lin = 0.0
        status = 0
        cl_limber_lin, status = (lib.angular_cl_vec_limber(cosmo.cosmo,
                                 t1, t2,
                                 psp_lin,
                                 [ell],
                                 integ_types['qag_quad'],
                                 1, status))
        if status != 0:
            raise ValueError("Error in Limber integrator.")
        cl_limber_nonlin, status = (lib.angular_cl_vec_limber(cosmo.cosmo,
                                    t1, t2,
                                    psp_nonlin,
                                    [ell],
                                    integ_types['qag_quad'],
                                    1, status))
        if status != 0:
            raise ValueError("Error in Limber integrator.")

        for i in range(len(kernels_t1)):
            # calls to fftlog to perform integration over chi integrals
            nu, deriv, plaw = get_general_params(bessels_t1[i])
            k, fk1 = (_fftlog_transform_general(chi_logspace_arr,
                      fchi1_arr[i], float(ell), nu, 1,
                      float(deriv), float(plaw)))
            transfer_t1 = np.array(clt1.get_transfer(np.log(k), avg_a1))[i]
            for j in range(len(kernels_t2)):
                # calls to fftlog to perform integration over chi integrals
                if clt1 != clt2:
                    nu2, deriv2, plaw2 = get_general_params(bessels_t2[j])
                    k, fk2 = (_fftlog_transform_general(chi_logspace_arr,
                              fchi2_arr[j], float(ell), nu2, 1,
                              float(deriv2), float(plaw2)))
                    transfer_t2 = np.array(clt2.get_transfer(np.log(k),
                                                             avg_a2))[j]
                else:
                    fk2 = fk1
                    transfer_t2 = transfer_t1

                # sum contributions of all components of the tracers
                # to calculate non-limber portion of the cl's
                cls_nonlimber_lin += (np.sum(fk1 * transfer_t1
                                      * fk2 * transfer_t2
                                      * k**kpow
                                      * pk(k, 1.0, cosmo))
                                      * dlnr * 2./np.pi
                                      * fll_t1[i][el] * fll_t2[j][el])

        # append the final cl calculation to the returned array
        # and check whether to continue to higher ells
        cells.append(cl_limber_nonlin[-1] - cl_limber_lin[-1]
                     + cls_nonlimber_lin)
        if ((np.abs(cells[-1]/cl_limber_nonlin[-1] - 1) < limber_max_error
                and l_limber == 'auto')
                or (type(l_limber) != str and ell >= l_limber)):

            l_limber = ell
            break

    if type(l_limber) == str:
        l_limber = ls[-1]
    return l_limber, cells, status
