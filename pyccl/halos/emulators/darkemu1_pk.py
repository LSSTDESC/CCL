__all__ = ("darkemu_power_spectrum", "galaxy_bias", "galaxy_bias_DEmuxHOD", )

import warnings
import numpy as np

from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.interpolate import RegularGridInterpolator as rgi

from .. import Profile2pt

from dark_emulator import model_hod
demuhod = model_hod.darkemu_x_hod()


def galaxy_bias(cosmo, hmc, a, prof):
    """ Computes the galaxy bias for a given halo profile."""
    Mh = np.logspace(12.2,15.0,2**5+1) / cosmo['h'] # [Msun]
    lmass = np.log10(Mh)
    NgalM = _Ngal(Mh, 1., profile=prof)
    # hmc._mf: dn/dlogM [Mpc^-3] (comoving) not dn/dM
    hmc._get_ingredients(cosmo, a, get_bf=False)
    hmf = ius(hmc._lmass, hmc._mf)(lmass)
    norm1 = hmc._integrator(hmf * NgalM, lmass)  # integral of dn/dM
    hbias = np.array([demuhod.get_bias_mass(Mi, redshift=1./a - 1)[0,0]
                      for Mi in Mh*cosmo['h']])
    galbias = hmc._integrator(hmf * hbias * NgalM, lmass) / norm1
    # integral of dn/dM
    return galbias


def galaxy_bias_DEmuxHOD(cosmo, a, prof):
    a_use = np.atleast_1d(a).astype(float)
    good_a = (1./a_use - 1. <= 1.48) & (1./a_use - 1. >= 0.)
    if np.any(good_a is False):
        warnings.warn('Dark Emulator I supports 0 <= z <= 1.48. '
                      'The redshifts larger than 1.48 will be truncated.')
    a_use = a_use[good_a]

    # cosmological params for the dark emulator
    h_ = cosmo['h']
    if np.isnan(cosmo['A_s']):
        raise ValueError("Dark Emulator needs A_s as an input parameter.")
    # [omegab0.,omega_c0.,Omega_Lambda0,np.log(As*10**10),ns,w0]
    demuhod.set_cosmology(np.array([cosmo['Omega_b']*h_**2.,
                                 cosmo['Omega_c']*h_**2.,
                                 1-(cosmo['Omega_c']+cosmo['Omega_b']),
                                 np.log(cosmo['A_s']*10**10),
                                 cosmo['n_s'],
                                 cosmo['w0']]))
    # HOD parameters for the dark emulator
    gparam_input = {"logMmin": prof.log10Mmin_0 + np.log10(h_),
                    "sigma_sq": (prof.siglnM_0/np.log(10))**2,
                    "logM1": prof.log10M1_0 + np.log10(h_),
                    "alpha": prof.alpha_0,
                    "kappa": 10**(prof.log10M0_0 - prof.log10Mmin_0),
                    "poff": 0., "Roff": 0.,
                    "sat_dist_type": "NFW",
                    "alpha_inc": 0., "logM_inc": 0.}
    demuhod.set_galaxy(gparam_input)

    galbias = np.array([demuhod._get_effective_bias(redshift=z_) 
                        for z_ in (1./a_use - 1.)])
    if len(galbias) == 1:
        return galbias[0]
    else:
        return galbias


def _compute_logdens(dehod, Mh, redshift):
    Mh = np.atleast_1d(Mh)
    logdens = np.log10([dehod._convert_mass_to_dens(
        Mh[i], redshift, integration=dehod.config["hmf_int_algorithm"])
        for i in range(len(Mh))])
    return logdens


def _compute_p_hh(dehod, ks, Mh, redshift):
    if not dehod.logdens_computed:
        dehod._compute_logdens(redshift)
    dehod._compute_p_hh_spl(redshift)

    points_known = (-dehod.g1.logdens_list, -dehod.g1.logdens_list, dehod.fftlog_2h.k)
    logdens = _compute_logdens(dehod=dehod, Mh=Mh, redshift=redshift)
    grid_d1, grid_d2, grid_k = np.meshgrid(-logdens, -logdens,
                                           ks, indexing='ij')
    points_target = np.stack([grid_d1.ravel(), grid_d2.ravel(),
                              grid_k.ravel()], axis=-1)

    interpolator = rgi(points_known, dehod.p_hh_base, method='cubic', 
                       bounds_error=False, fill_value=None)

    p_hh = interpolator(points_target).reshape(len(logdens), len(logdens), len(ks))
    return p_hh.transpose(2, 0, 1)



# referred from dark_emulator_public/dark_emulator/model_hod/hod_interface.py
def _compute_p_hm(dehod, ks, Mh, redshift):
    if not dehod.logdens_computed:
        dehod._compute_logdens(redshift)
    if not dehod.xi_hm_computed:
        dehod._compute_xi_hm(redshift)

    logdens = _compute_logdens(dehod=dehod, Mh=Mh, redshift=redshift)

    if dehod.do_linear_correction:
        pm_lin = dehod.get_pklin(ks)

    p_hm = np.zeros((len(dehod.logdens), len(dehod.fftlog_1h.k)))
    for i in range(len(dehod.logdens)):
        p_hm[i] = dehod.fftlog_1h.xi2pk(dehod.xi_hm[i],
                                        1.01, N_extrap_high=1024)[1]
        if dehod.do_linear_correction:
            p_hm[i] *= (dehod.pm_lin_k_1h_out_of_range/pm_lin)

    return rbs(dehod.fftlog_1h.k, -dehod.logdens, p_hm.T)(ks, -logdens)


def _Ngal(M, a, profile):
    Nc = profile._Nc(M, a)
    Ns = profile._Ns(M, a)
    fc = profile._fc(a)
    if profile.ns_independent:
        return Nc*fc + Ns
    return Nc*(fc + Ns)


def _I_0_2(hmc, cosmo, k, a, prof, *, prof2=None, prof_2pt, lmass, hmf):
    if prof2 is None:
        prof2 = prof

    M_array = 10**lmass
    hmc._check_mass_def(prof, prof2)
    uk = prof_2pt.fourier_2pt(cosmo, k, M_array, a, prof, prof2=prof2).T
    return hmc._integrator(hmf[None, :] * uk, lmass)


def darkemu_power_spectrum(cosmo, hmc, k, a, prof, *,
                           prof2=None, prof_2pt=None,
                           get_1h=True, get_2h=True,
                           suppress_1h=None):
    """ Computes the halo model power spectrum for two
    quantities defined by their respective halo profiles.
    The halo model power spectrum for two profiles
    :math:`u` and :math:`v` is:
    
    .. math::
        P_{u,v}(k,a) = I^0_2(k,a|u,v) +
        I^1_1(k,a|u)\\,I^1_1(k,a|v)\\,P_{\\rm lin}(k,a)
    
    where :math:`P_{\\rm lin}(k,a)` is the linear matter
    power spectrum, :math:`I^1_1` is defined in the documentation
    of :meth:`~pyccl.halos.halo_model.HMCalculator.I_1_1`, and :math:`I^0_2`
    is defined in the documentation of
    :meth:`~pyccl.halos.halo_model.HMCalculator.I_0_2`.
    
    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo model calculator.
        k (:obj:`float` or `array`): comoving wavenumber in Mpc^-1.
        a (:obj:`float` or `array`): scale factor.
        prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile.
        prof2 (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): a
            second halo profile. If ``None``, ``prof`` will be used as
            ``prof2``.
        prof_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object
            returning the the two-point moment of the two profiles
            being correlated. If ``None``, the default second moment
            will be used, corresponding to the products of the means
            of both profiles.
        get_1h (:obj:`bool`): if ``False``, the 1-halo term (i.e. the first
            term in the first equation above) won't be computed.
        get_2h (:obj:`bool`): if ``False``, the 2-halo term (i.e. the second
            term in the first equation above) won't be computed.
        suppress_1h (:obj:`callable` or :obj:`None`):
            Suppress the 1-halo large scale contribution by a
            time- and scale-dependent function :math:`k_*(a)`,
            defined as in `HMCODE-2020 <https://arxiv.org/abs/2009.01858>`_:
            :math:`1/[1+(k_*(a)/k)^4]`.
            If ``None`` the standard 1-halo term is returned with no damping.
    
    Returns:
        (:obj:`float` or `array`): integral values evaluated at each
        combination of ``k`` and ``a``. The shape of the output will
        be ``(N_a, N_k)`` where ``N_k`` and ``N_a`` are the sizes of
        ``k`` and ``a`` respectively. If ``k`` or ``a`` are scalars, the
        corresponding dimension will be squeezed out on output.
    """ # noqa
    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

    good_a = (1./a_use - 1. <= 1.48) & (1./a_use - 1. >= 0.)
    if np.any(good_a is False):
        warnings.warn('Dark Emulator I supports 0 <= z <= 1.48. '
                      'The redshifts larger than 1.48 will be truncated.')
    a_use = a_use[good_a]

    # cosmological params for the dark emulator
    h_ = cosmo['h']
    if np.isnan(cosmo['A_s']):
        raise ValueError("Dark Emulator needs A_s as an input parameter.")
    # [omegab0.,omega_c0.,Omega_Lambda0,np.log(As*10**10),ns,w0]
    demuhod.set_cosmology(np.array([cosmo['Omega_b']*h_**2.,
                                 cosmo['Omega_c']*h_**2.,
                                 1-(cosmo['Omega_c']+cosmo['Omega_b']),
                                 np.log(cosmo['A_s']*10**10),
                                 cosmo['n_s'],
                                 cosmo['w0']]))
    # HOD parameters for the dark emulator
    gparam_input = {"logMmin": prof.log10Mmin_0 + np.log10(h_),
                    "sigma_sq": (prof.siglnM_0/np.log(10))**2,
                    "logM1": prof.log10M1_0 + np.log10(h_),
                    "alpha": prof.alpha_0,
                    "kappa": 10**(prof.log10M0_0 - prof.log10Mmin_0),
                    "poff": 0., "Roff": 0.,
                    "sat_dist_type": "NFW",
                    "alpha_inc": 0., "logM_inc": 0.}
    demuhod.set_galaxy(gparam_input)

    if suppress_1h is not None:
        if not get_1h:
            raise ValueError("Can't suppress the 1-halo term "
                             "when get_1h is False.")

    if prof2 is None:
        prof2 = prof
    if prof_2pt is None:
        prof_2pt = Profile2pt()

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    M_array = hmc._mass.copy()
    good_m = (M_array <= 1E16 / h_) & (M_array >= 1E12 / h_)
    if np.any(good_m is False):
        warnings.warn('Dark Emulator I supports 1E12 <= Mh [Msun/h] <= 1E16. '
                      'Others will be truncated.')
    M_array = M_array[good_m]
    lmass = np.log10(M_array)
    nM = len(M_array)


    for ia, aa in enumerate(a_use):
        # normalizations
        # norm1 = prof.get_normalization(cosmo, aa, hmc=hmc)
        # redshift dependence is not consistent with the original darkemulator.
        NgalM = _Ngal(M_array, 1., profile=prof)
        # hmc._mf: dn/dlogM [Mpc^-3] (comoving) not dn/dM
        hmc._get_ingredients(cosmo, aa, get_bf=False)
        hmf = ius(hmc._lmass, hmc._mf)(lmass)
        norm1 = hmc._integrator(hmf * NgalM, lmass)  # integral of dn/dM
        if prof2 == prof:
            norm2 = norm1


        # calc profile in fourier space
        u1k = prof.fourier(cosmo, k_use, M_array, aa).T  # (Nk,NM)
        if prof2 == prof:  # pkgg
            u2k = u1k
            
            # (Nk,NM,NM)
            pkhh = _compute_p_hh(dehod=demuhod, ks=k_use/h_,
                                  Mh=M_array*h_, redshift=1./aa-1) / h_**3.

            # integration
            pk_2h_M2_int = list()
            for i in range(nM):
                res = pkhh[:, i] * u1k
                pk_2h_M2_int.append(hmc._integrator(
                    hmf[None, :] * res, lmass))
            res = np.array(pk_2h_M2_int).T * u2k
            pk_2h = hmc._integrator(hmf[None, :] * res, lmass)

        else:
            # (Nk,NM)
            pkhm = _compute_p_hm(demuhod, k_use/h_, M_array*h_,
                                 redshift=1./aa-1) / h_**3.

            # integration
            res = pkhm * u1k
            pk_2h = hmc._integrator(hmf[None, :] * res, lmass)

        if get_1h and (prof2 == prof):
            pk_1h = _I_0_2(hmc, cosmo, k_use, aa, prof,
                           prof2=prof2, prof_2pt=prof_2pt, 
                           lmass=lmass, hmf=hmf)  # 1h term
            
            if suppress_1h is not None:
                # large-scale damping of 1-halo term
                ks = suppress_1h(aa)
                pk_1h *= (k_use / ks)**4 / (1 + (k_use / ks)**4)
        else:
            pk_1h = 0

        # smooth 1h/2h transition region
        if prof2 == prof:
            out[ia] = (pk_1h + pk_2h) / (norm1 * norm2)
        else:
            out[ia] = pk_2h / norm1

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out

