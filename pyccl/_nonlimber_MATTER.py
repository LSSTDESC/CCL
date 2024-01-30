__all__ = ("_nonlimber_MATTER",)
"""Written by Paul Rogozenski (paulrogozenski@arizona.edu),
 implementing the matter non-limber integration method of the N5K challenge
 detailed in this paper:
"""
import warnings
import numpy as np
from . import lib, check
from .pyutils import integ_types
import pyccl as ccl
from . import CCLWarning
import matterlib
from scipy.interpolate import CubicSpline as interp


def _choose_returned_cl(cls_matter, clt1, clt2, noni_1, noni_2):
    parse_str = 'dd'
    if noni_1 == noni_2 and not noni_1:
        parse_str = 'll'
    if noni_1 != noni_2:
        parse_str = 'dl'

    auto_keys = []
    cross_keys = []
    for key in cls_matter[parse_str].keys():
        if key[0] == key[1]:
            auto_keys.append(key)
        else:
            cross_keys.append(key)
    cells = np.zeros(len(cls_matter[parse_str][auto_keys[0]]))
    if clt1 != clt2 and len(cross_keys) > 0:
        cells = cls_matter[parse_str][cross_keys[0]]
    else:
        cells = cls_matter[parse_str][auto_keys[0]]
    return cells


def _get_chi_arrs(cosmo, clt, Nchi_nonintegrated=800, Nchi_integrated=1600):
    t = clt._trc
    threshold = 1e-50
    kernels_t, chis_t = clt.get_kernel()
    bessels_t = clt.get_bessel_derivative()
    Nt = len(bessels_t)
    nonintegrated = False
    chi_mins, chi_maxs = np.zeros((2, Nt), dtype="float64")
    for i, tt in enumerate(t):
        tt_test = kernels_t[i]
        maxtt = np.max(tt_test)
        mask = tt_test > threshold * maxtt
        chi_maxs[i] = chis_t[i][len(mask) - np.argmax(mask[::-1])-1]
        if bessels_t[i] >= 0:
            nonintegrated = True
            chi_mins[i] = chis_t[i][np.argmax(mask)]
        else:
            chi_mins[i] = 1.e-8

    if nonintegrated:
        chi = [np.linspace(chi_mins[i], chi_maxs[i],
               num=Nchi_nonintegrated) for i in range(Nt)]
        kerfac = np.zeros((Nt, Nchi_nonintegrated))
        for i in range(Nt):
            kern = clt.get_kernel(chi[i])
            trans = clt.get_transfer(0.,
                                     ccl.scale_factor_of_chi(cosmo, chi[i]))
            for itr in range(kern.shape[0]):
                kerfac[i] += kern[itr] * trans[itr]

    else:
        chi = [np.linspace(np.min(chi_mins[i]), np.max(chi_maxs[i]),
               num=Nchi_integrated) for i in range(Nt)]
        kerfac = np.zeros((Nt, Nchi_integrated))
        for i in range(Nt):
            kern = clt.get_kernel(chi[i])
            trans = clt.get_transfer(0.,
                                     ccl.scale_factor_of_chi(cosmo, chi[i]))

            for itr in range(kern.shape[0]):
                kerfac[i][1:] += kern[itr][1:] * trans[itr][1:] / chi[i][1:]**2
                # Ill defined for the 0th index, keep it as 0
    if nonintegrated:
        return (nonintegrated, chi, np.zeros((Nt, Nchi_integrated)),
                kerfac, np.zeros((Nt, Nchi_integrated)))
    return (nonintegrated, np.zeros((Nt, Nchi_nonintegrated)),
            chi, np.zeros((Nt, Nchi_nonintegrated)), kerfac)


def _nonlimber_MATTER(
    cosmo, clt1, clt2, p_of_k_a, p_of_k_a_lin, ls, l_limber, limber_max_error
):
    """clt1, clt2 are lists of tracer in a tracer object
    cosmo (:class:`~pyccl.core.Cosmology`): A Cosmology object.
    psp non-linear power spectrum
    l_limber max ell for non-limber calculation
    ell_use ells at which we calculate the non-limber integrals
    """

    # check if we already have a matterlib object.
    # If not, create one. Else, use the predefined version
    status = 0
    if (not (isinstance(p_of_k_a, str) and isinstance(p_of_k_a_lin, str)) and
       not (isinstance(p_of_k_a, ccl.Pk2D)
            and isinstance(p_of_k_a_lin, ccl.Pk2D)
            )):
        warnings.warn(
            "p_of_k_a and p_of_k_a_lin must be of the same "
            "type: a str in cosmo or a Pk2D object. "
            "Defaulting to Limber calculation. ", CCLWarning)
        return -1, np.array([]), status

    if not (isinstance(p_of_k_a, ccl.Pk2D)):
        cosmo.compute_nonlin_power()
        pk = cosmo.get_nonlin_power(p_of_k_a)
    else:
        pk = p_of_k_a

    a_arr, lk_arr, pk_arr = pk.get_spline_arrays()
    lk_arr = np.log10(np.exp(lk_arr))
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

    lmax = l_limber
    verbosity = 0
    sfftcutoff = 100
    stw = 60
    sitw = 480
    st = 200
    st_spline = 60
    l_logstep = 1.1
    l_linstep = 75
    seperability = False
    kmin = 1.e-7
    Nk_fft = 256
    Nchi_nonintegrated = 800
    Nchi_integrated = 1600
    internal_logchi_i_offset = 2e-31
    status = 0

    ma = matterlib.Matter(ma_verbose=verbosity)

    (noni_1, chi_noni_1, chi_i_1, kerfac_noni_1, kerfac_i_1) = _get_chi_arrs(cosmo, clt1, Nchi_nonintegrated, Nchi_integrated)
    (noni_2, chi_noni_2, chi_i_2, kerfac_noni_2, kerfac_i_2) = _get_chi_arrs(cosmo, clt2, Nchi_nonintegrated, Nchi_integrated)

    if noni_1 == noni_2:
        chi_nonintegrated = np.concatenate([chi_noni_1, chi_noni_2])
        chi_integrated = np.concatenate([chi_i_1, chi_i_2])
        kerfac_noni = np.concatenate([kerfac_noni_1, kerfac_noni_2])
        kerfac_i = np.concatenate([kerfac_i_1, kerfac_i_2])
        if noni_1 is False:
            Nk_fft *= 2
        else:
            st *= 4
            st_spline *= 2
    else:
        if noni_1:
            chi_nonintegrated = chi_noni_1
            chi_integrated = chi_i_2
            kerfac_noni = kerfac_noni_1
            kerfac_i = kerfac_i_2
        else:
            chi_nonintegrated = chi_noni_2
            chi_integrated = chi_i_1
            kerfac_noni = kerfac_noni_2
            kerfac_i = kerfac_i_1
    power = pk_arr
    Na_pk = len(a_arr)

    a_pk = a_arr
    pk_growth = ccl.growth_factor(cosmo, a_pk)
    growth = pk_growth
    n_s = cosmo['n_s']

    growth_func = interp(a_pk, pk_growth)
    for i in range(len(chi_nonintegrated)):
        kerfac_noni[i] *= growth_func(ccl.scale_factor_of_chi(cosmo, chi_nonintegrated[i]))
    for i in range(len(chi_integrated)):
        kerfac_i[i] *= growth_func(ccl.scale_factor_of_chi(cosmo, chi_integrated[i]))

    new_k_min = kmin
    Nk_small = int((lk_arr[0] - np.log10(new_k_min))/(lk_arr[1] - lk_arr[0]) + 1)
    assert (Nk_small > 10)
    k = 10 ** lk_arr
    ksmall = np.geomspace(new_k_min, k[0], endpoint=False, num=Nk_small)
    k_all = np.concatenate([ksmall, k])

    chi_pk = ccl.comoving_radial_distance(cosmo, a_pk)
    k_pk = np.geomspace(kmin, k[-1], num=Nk_fft)
    pk = np.empty((Na_pk, Nk_fft))
    deltaksq = np.empty((Na_pk, Nk_fft))
    for i in range(Na_pk):
        pk_all = np.concatenate([(ksmall / k[0]) ** (n_s) * power[i][0], power[i]])
        pk[i] = interp(k_all, pk_all)(k_pk)
        deltaksq[i] = interp(k_all, pk_all)(k_pk) * k_pk**3 / (2. * np.pi ** 2)

    if type(lmax) is not str:
        print(lmax)
        ma.set((chi_nonintegrated, chi_integrated),
               (kerfac_noni, kerfac_i), (a_pk, chi_pk, k_pk, deltaksq), growth,
               lmax=lmax, uses_separability=seperability,
               size_fft_cutoff=sfftcutoff, tw_size=stw,
               integrated_tw_size=sitw,
               l_logstep=l_logstep, l_linstep=l_linstep,
               t_size=st, t_spline_size=st_spline,
               internal_logchi_i_offset=internal_logchi_i_offset
               )
        ma.compute()
        all_ells = ls
        ell_matter = all_ells[all_ells <= lmax]
        cls_matter = ma.matter_cl(ell_matter)
        cells = _choose_returned_cl(cls_matter, clt1, clt2, noni_1, noni_2)
    else:
        for el in range(220, 221, 100):
            lmax = el
            ma.set((chi_nonintegrated, chi_integrated),
                   (kerfac_noni, kerfac_i), (a_pk, chi_pk, k_pk, deltaksq), growth,
                   lmax=lmax, uses_separability=seperability,
                   size_fft_cutoff=sfftcutoff, tw_size=stw,
                   integrated_tw_size=sitw,
                   l_logstep=l_logstep, l_linstep=l_linstep,
                   t_size=st, t_spline_size=st_spline,
                   internal_logchi_i_offset=internal_logchi_i_offset
                   )

            cl_limber_nonlin, status = lib.angular_cl_vec_limber(
                cosmo.cosmo,
                t1,
                t2,
                psp_nonlin,
                ls[ls <= lmax],
                integ_types["qag_quad"],
                len(ls[ls <= lmax]),
                status,
            )
            check(status, cosmo=cosmo)
            ma.compute()
            all_ells = ls
            ell_matter = all_ells[all_ells <= lmax]
            cls_matter = ma.matter_cl(ell_matter)

            cells = _choose_returned_cl(cls_matter, clt1, clt2, noni_1, noni_2)
            if len((cells)[np.abs(cells / cl_limber_nonlin - 1)
                           < (limber_max_error)]) > 0:
                lmax = np.where(np.abs(cells / cl_limber_nonlin - 1)
                                < (limber_max_error))[0][0]
                cells = cells[:(lmax + 1)]
                lmax = ls[lmax]
                break
    ma.clean()
    return lmax, cells, status
