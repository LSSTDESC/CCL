__all__ = ("darkemu_power_spectrum",)

import warnings
import numpy as np
from scipy import integrate
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.interpolate import InterpolatedUnivariateSpline as ius

from ... import Pk2D
from .. import Profile2pt

from dark_emulator import darkemu
from dark_emulator import model_hod
hod = model_hod.darkemu_x_hod()

def _compute_logdens(Mh, redshift):
    Mh = np.atleast_1d(Mh)
    logdens = np.log10([hod._convert_mass_to_dens(
        Mh[i], redshift, integration=hod.config["hmf_int_algorithm"]) for i in range(len(Mh))])
    return logdens

def _compute_p_hh(ks, Mh, redshift): # referred from dark_emulator_public/dark_emulator/model_hod/hod_interface.py
    hod._compute_p_hh_spl(redshift)

    logdens_de = hod.g1.logdens_list
    logdens = _compute_logdens(Mh, redshift)

    p_hh_tmp = np.zeros(
        (len(logdens_de), len(Mh), len(ks)))
    for i in range(len(logdens_de)):
        p_hh_tmp[i] = rbs(-logdens_de, hod.fftlog_2h.k,
                               hod.p_hh_base[i])(-logdens, ks)

    p_hh = np.zeros((len(Mh), len(Mh), len(ks)))
    for i in range(len(Mh)):
        p_hh[:, i] = rbs(-logdens_de, hod.fftlog_2h.k,
                              p_hh_tmp[:, i, :])(-logdens, ks)
    return p_hh.transpose(2, 0, 1)


def _compute_p_hm(ks, Mh, redshift): # referred from dark_emulator_public/dark_emulator/model_hod/hod_interface.py
    if hod.logdens_computed == False:
        hod._compute_logdens(redshift)
    if hod.xi_hm_computed == False:
        hod._compute_xi_hm(redshift)

    logdens = _compute_logdens(Mh, redshift)

    if hod.do_linear_correction:
        pm_lin = hod.get_pklin(ks)

    p_hm = np.zeros((len(logdens), len(ks)))
    for i in range(len(logdens)):
        p_hm[i] = hod.fftlog_1h.xi2pk(hod.xi_hm[i], 1.01, N_extrap_high=1024)[1]
        if hod.do_linear_correction:
            p_hm[i] *= (hod.pm_lin_k_1h_out_of_range/pm_lin)

    return p_hm.T


def darkemu_power_spectrum(demu, cosmo, hmc, k, a, prof, *,
                           prof2=None, prof_2pt=None,
                           p_of_k_a=None,
                           get_1h=True, get_2h=True,
                           suppress_1h=None,
                           extrap_pk=False): # KI add demu
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
        demu (:class:`dark_emulator.darkemu.base_class`): the dark emulator object.
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
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If ``None``,
            the power spectrum stored within `cosmo` will be used.
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
        extrap_pk (:obj:`bool`):
            Whether to extrapolate ``p_of_k_a`` in case ``a`` is out of its
            support. If ```False```, and the queried values are out of bounds,
            an error is raised.
    
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
    if np.any(good_a == False):
        warnings.warn('Dark Emulator I supports 0 <= z <= 1.48. The redshifts rather than 1.48 will be truncated.')
    a_use = a_use[good_a]

    # KI add: cosmology set for the dark emulator
    h_ = cosmo['h']
    if np.isnan(cosmo['A_s']):
        raise ValueError("Dark Emulator needs A_s as input parameters.")
    # [omegab0.,omega_c0.,Omega_Lambda0,np.log(As*10**10),ns,w0]
    demu.set_cosmology(np.array([cosmo['Omega_b']*h_**2.,
                                 cosmo['Omega_c']*h_**2.,
                                 1-(cosmo['Omega_c']+cosmo['Omega_b']),
                                 np.log(cosmo['A_s']*10**10),
                                 cosmo['n_s'],
                                 cosmo['w0']]))
    
    if suppress_1h is not None:
        if not get_1h:
            raise ValueError("Can't suppress the 1-halo term "
                             "when get_1h is False.")
    
    if prof2 is None:
        prof2 = prof
    if prof_2pt is None:
        prof_2pt = Profile2pt()
    demu_phh_mass = demu.get_phh_mass
    demu_phm_mass = demu.get_phm_mass
    
    # pk2d = cosmo.parse_pk(p_of_k_a)
    extrap = cosmo if extrap_pk else None  # extrapolation rule for pk2d
    
    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    M_array = hmc._mass.copy()
    good_m = (M_array <= 1E16 / h_) & (M_array >= 1E12 / h_)
    M_array = M_array[good_m]
    if np.any(good_m == False):
        warnings.warn('Dark Emulator I supports 1E12 <= Mh [Msun/h] <= 1E16. Others will be truncated.')
    nM = len(M_array)
    print('nM', nM)
    lmass = np.log10(M_array)
    for ia, aa in enumerate(a_use):
        
        # normalizations
        norm1 = prof.get_normalization(cosmo, aa, hmc=hmc)
        if prof2 == prof:
            norm2 = norm1
        else:
            norm2 = prof2.get_normalization(cosmo, aa, hmc=hmc)

        # call components: hmf, uk, 
        # hmc._mf: dn/dlogM [Mpc^-3] (comoving) not dn/dM 
        hmc._get_ingredients(cosmo, aa, get_bf=False)
        hmf = ius(hmc._lmass, hmc._mf)(lmass)
        u1k = prof.fourier(cosmo, k_use, M_array, aa).T
        
        # calc pkhx
        if prof2 == prof: # pkgg
            u2k = u1k
        # else:
        #     u2k = prof2.fourier(cosmo, k_use, M_array, aa).T
        
            #########
            # pkhh
            # Dark Emulator accepts k [h/Mpc] and returns Pk [Mpc/h]^3 as units
            pkhh = np.zeros([nk, nM, nM])
            for i in range(nM):
                for j in range(i+1):
                    vals = demu_phh_mass(ks=k_use/h_,
                                         M1=M_array[i]*h_,M2=M_array[j]*h_,
                                         redshift=1./aa-1.) / h_**3.
                    pkhh[:,i,j] = vals
                    pkhh[:,j,i] = vals
            # pkhh = _compute_p_hh(k_use, M_array, redshift=1./aa-1) # (Nk,NM,NM)
            
            # integration
            pk_2h_M2_int = list()
            for i in range(nM):
                pk_2h_M2_int.append(hmc._integrator(
                    pkhh[:,i] * u1k * hmf[None,:], lmass))
            pk_2h = hmc._integrator(np.array(pk_2h_M2_int).T * u2k * hmf[None,:], lmass)
            #########

        else:
            pkhm = np.zeros([nk, nM])
            for i in range(nM):
                pkhm[:,i] = demu_phm_mass(ks=k_use/h_, M=M_array[i]*h_, redshift=1./aa-1.) / h_**3.
            # pkhm = _compute_p_hm(k_use, M_array, redshift=1./aa-1) # (Nk,NM)
            
            pk_2h = hmc._integrator(pkhm * u1k * hmf[None,:], lmass)
    
        if get_1h and (prof2 == prof):
            pk_1h = hmc.I_0_2(cosmo, k_use, aa, prof,
                              prof2=prof2, prof_2pt=prof_2pt)  # 1h term
    
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