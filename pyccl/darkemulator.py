from . import ccllib as lib

from .pyutils import check, _get_spline2d_arrays, _get_spline3d_arrays
import numpy as np

from . import core
import warnings
from .errors import CCLWarning
from .pk2d import Pk2D
from .tk3d import Tk3D
  
from dark_emulator import darkemu
from dark_emulator import model_hod
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from . import halos

def darkemu_Tk3D_SSC(cosmo, prof1, deltah=0.02,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     normprof1=False,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing
    the super-sample covariance trispectrum, given by the tensor
    product of the power spectrum responses associated with the
    two pairs of quantities being correlated. Each response is
    calculated as:

    .. math::
        \\frac{\\partial P_{u,v}(k)}{\\partial\\delta_L} =
        \\left(\\frac{68}{21}-\\frac{d\\log k^3P_L(k)}{d\\log k}\\right)
        P_L(k)I^1_1(k,|u)I^1_1(k,|v)+I^1_2(k|u,v) - (b_{u} + b_{v})
        P_{u,v}(k)

    where the :math:`I^a_b` are defined in the documentation
    of :meth:`~HMCalculator.I_1_1` and  :meth:`~HMCalculator.I_1_2` and
    :math:`b_{u}` and :math:`b_{v}` are the linear halo biases for quantities
    :math:`u` and :math:`v`, respectively (zero if they are not clustering).

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof1 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
    
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`,
            the power spectrum stored within `cosmo` will be used.
        a_arr (array): an array holding values of the scale factor
            at which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        use_log (bool): if `True`, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: SSC effective trispectrum.
    """
    if lk_arr is None:
        status = 0
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status, cosmo=cosmo)
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status, cosmo=cosmo)

    k_use = np.exp(lk_arr)

    # Check inputs
    if not isinstance(prof1, halos.profiles.HaloProfile):
        raise TypeError("prof1 must be of type `HaloProfile`")
    
    h = cosmo["h"]
    k_emu = k_use / h   # [h/Mpc]

    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')
         
    # set cosmology for dark emulator
    emu = darkemu_set_cosmology(cosmo)

    # set h-modified cosmology to take finite differencing
    hp = h + deltah 
    hm = h - deltah 
    cosmo_hp, cosmo_hm = set_hmodified_cosmology(cosmo,deltah)

    emu_p = darkemu_set_cosmology(cosmo_hp)
    emu_m = darkemu_set_cosmology(cosmo_hm)

    # Growth factor                         
    Dp = cosmo_hp.growth_factor_unnorm(a_arr)
    Dm = cosmo_hm.growth_factor_unnorm(a_arr)

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    pk12 = np.zeros([na, nk])
    #dpk34 = np.zeros([na, nk])
    Mfor_hmf = np.linspace(8,17,200)
    Mh = np.linspace(log10Mh_min,log10Mh_max,2**5+1)  # M_sol/h
    M = np.log10(10**Mh/h)
    dM = M[1] - M[0]
    dlogM = dM
    b1_th_tink = np.zeros(len(Mh))
    #b2_th_tink = np.zeros(len(Mh))
    Pth = [0] * len(Mh)
    Pnth_hp = [0] * len(Mh)
    Pnth_hm = [0] * len(Mh)
    Pbin = [0] * len(Mh)
    nths = np.zeros(len(Mh))
    
    mass_def=halos.MassDef200m()
    hmf_DE = halos.MassFuncDarkEmulator(cosmo,mass_def=mass_def) 
    hbf = halos.hbias.HaloBiasTinker10(cosmo,mass_def=mass_def)
        
    if np.any(a_arr < 1/(1+1.48)):
        hmf = halos.MassFuncTinker10(cosmo,mass_def=mass_def) 

        nfw = halos.HaloProfileNFW(halos.ConcentrationDuffy08(mass_def),
                                       fourier_analytic=True)
        hmc = halos.HMCalculator(cosmo, hmf, hbf, mass_def)

        halomod_pk_arr = halos.halomod_power_spectrum(cosmo, hmc, k_use, a_arr,
                                        prof=nfw, prof_2pt=None,
                                        prof2=prof1, p_of_k_a=None,
                                        normprof1=True, normprof2=True,
                                        get_1h=True, get_2h=True,
                                        smooth_transition=None,
                                        supress_1h=None)

        halomod_tk3D, dpk12_halomod = halos.halomod_Tk3D_SSC(cosmo=cosmo, hmc=hmc,
                                              prof1=nfw,
                                              prof2=prof1,
                                              prof12_2pt=None,
                                              normprof1=True, normprof2=True,
                                              lk_arr=np.log(k_use), a_arr=a_arr,
                                              use_log=use_log)

    for ia, aa in enumerate(a_arr):
        z = 1. / aa - 1   # dark emulator is valid for 0 =< z <= 1.48
        if z > 1.48:
            dpk12[ia, :] = dpk12_halomod[ia, :]  
            pk12[ia, :] = halomod_pk_arr[ia, :]
            print("use halo model for z={:.2f}>1.48".format(z))
        else:
            # mass function 
            dndlog10m_emu = ius(Mfor_hmf ,hmf_DE.get_mass_function(cosmo, 10**Mfor_hmf ,aa))  # Mpc^-3  #ius(np.log10(Mlist), dndm_emu * Mlist * np.log(10) * h ** 3)
            
            if Mh[0] < 12.0:  # Msol/h
                    Pth[0] = emu.get_phm_massthreshold(k_emu,10**12,z) * (1/h)**3
                    nths12 = emu.mass_to_dens(10**12,z) * h**3
                    Pnth_hp[0] = emu_p.get_phm(k_emu*(h/hp),np.log10(nths12*(1/hp)**3),z)*(1/hp)**3
                    Pnth_hm[0] = emu_m.get_phm(k_emu*(h/hm),np.log10(nths12*(1/hm)**3),z)*(1/hm)**3
                    Pbin[0] = emu.get_phm_mass(k_emu, 10 ** 12, z) * (1/h)**3
            else:  
                    Pth[0] = emu.get_phm_massthreshold(k_emu,10**Mh[0],z) * (1/h)**3     
                    nths[0] = emu.mass_to_dens(10**Mh[0],z) * h**3
                    Pnth_hp[0] = emu_p.get_phm(k_emu*(h/hp),np.log10(nths[0]*(1/hp)**3),z)*(1/hp)**3
                    Pnth_hm[0] = emu_m.get_phm(k_emu*(h/hm),np.log10(nths[0]*(1/hm)**3),z)*(1/hm)**3
                    Pbin[0] = emu.get_phm_mass(k_emu, 10 ** Mh[0], z) * (1/h)**3
                   
            for m in range(1,len(Mh)):
                if Mh[m] < 12.0:  # Msol/h
                    Pth[m] = Pth[0]
                    Pnth_hp[m] = Pnth_hp[0]
                    Pnth_hm[m] = Pnth_hm[0]
                    Pbin[m] = Pbin[0] 
                else:  
                    Pth[m] = emu.get_phm_massthreshold(k_emu,10**Mh[m],z) * (1/h)**3     
                    nths[m] = emu.mass_to_dens(10**Mh[m],z) * h**3
                    Pnth_hp[m] = emu_p.get_phm(k_emu*(h/hp),np.log10(nths[m]*(1/hp)**3),z)*(1/hp)**3
                    Pnth_hm[m] = emu_m.get_phm(k_emu*(h/hm),np.log10(nths[m]*(1/hm)**3),z)*(1/hm)**3
                    Pbin[m] = emu.get_phm_mass(k_emu, 10 ** Mh[m], z) * (1/h)**3
                  
                    
            
                M1 = np.linspace(M[m], M[-1], 2**5+1)
                dM1 = M[1] - M[0]
                b1_th_tink[m] = integrate.romb(dndlog10m_emu(M1) * hbf.get_halo_bias(cosmo,(10 ** M1), aa), dx = dM1)\
                        /integrate.romb(dndlog10m_emu(M1), dx = dM1)

                
            Nc = prof1._Nc(10 ** M, aa)
            Ns = prof1._Ns(10 ** M, aa)
            fc = prof1._fc(aa)
            Ng =  Nc * (fc + Ns)
            Mps = M + dlogM
            Mms = M - dlogM

            prof_Mp = prof1.fourier(cosmo, k_use, (10 ** Mps), aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):
            prof_Mm = prof1.fourier(cosmo, k_use, (10 ** Mms), aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):
            prof = prof1.fourier(cosmo, k_use,(10 ** M), aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):

            dprof_dlogM = (prof_Mp - prof_Mm) / (2 * dlogM)#*np.log(10))
            nth_mat = np.tile(nths, (len(k_use), 1)).transpose()
            ng = integrate.romb(dndlog10m_emu(M) * Ng, dx = dM, axis = 0)
            bgE = integrate.romb(dndlog10m_emu(M) * Ng * \
            (hbf.get_halo_bias(cosmo,(10 ** M), aa)), dx = dM, axis = 0) / ng

            bgE2 = integrate.romb(dndlog10m_emu(M) * Ng * \
            b2H17(hbf.get_halo_bias(cosmo,(10 ** M), aa)), dx = dM, axis = 0) / ng
            bgL = bgE - 1

            dndlog10m_func_mat = np.tile(dndlog10m_emu(M), (len(k_emu), 1)).transpose()  # M_sol,Mpc^-3
            b1L_th_mat = np.tile(b1_th_tink -1, (len(k_emu), 1)).transpose()
            Pgm = integrate.romb(dprof_dlogM * (nth_mat * np.array(Pth)), \
            dx = dM, axis = 0) / ng   

            dPhm_db_nfix = (26. / 21.) * (np.array(Pnth_hp) - np.array(Pnth_hm)) / \
                        (2 * (np.log(Dp[ia]) - np.log(Dm[ia])))  # Mpc^3

            dnP_hm_db_emu = nth_mat * (dPhm_db_nfix + b1L_th_mat * np.array(Pbin))  # Dless

            # stitching
            k_switch = 0.08  # [h/Mpc]
            kmin = 1e-2  # [h/Mpc]
            dnP_gm_db = integrate.romb(dprof_dlogM * (dnP_hm_db_emu), dx = dM, axis = 0) #Dless
                        
            Pgm_growth = dnP_gm_db / ng - bgL * Pgm  # Dless

            Pgm_d = -1. / 3. *  np.gradient(np.log(Pgm)) / np.gradient(np.log(k_use)) * Pgm #Dless

            dpklin = pk2dlin.eval_dlogpk_dlogk(k_use, aa, cosmo) 

            Pgm_lin = bgE * pk2dlin.eval(k_use, aa, cosmo)
            dPgm_db_lin = (47/21 + bgE2/bgE - bgE -1/3 * dpklin) * \
                           bgE * pk2dlin.eval(k_use, aa, cosmo)
            dPgm_db = dPgm_db_lin * np.exp(-k_emu/k_switch) + \
                     (Pgm_growth + Pgm_d) * (1 - np.exp(-k_emu/k_switch))

            Pgm = Pgm_lin * np.exp(-k_emu/k_switch) + \
                  Pgm * (1 - np.exp(-k_emu/k_switch))

            # use linear theory below kmin
            dPgm_db[k_emu < kmin] = dPgm_db_lin[k_emu < kmin]
            dpk12[ia, :] = dPgm_db

            Pgm[k_emu < kmin] = Pgm_lin[k_emu < kmin]
            pk12[ia, :] = Pgm


    if use_log:
        if np.any(dpk12 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning)
            use_log = False
        else:
            dpk12 = np.log(dpk12)
            
    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                cosmo=cosmo, is_logp=False)
    
    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d, pk2d

def darkemu_pkarr_SSC(cosmo, prof1, deltah=0.02,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     normprof1=False, kmax=2.0,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False, highk_HM=True):
    """ Returns a 2D array with shape `[na,nk]` describing the
    first function :math:`f_1(k,a)` that makes up a factorizable
    trispectrum :math:`T(k_1,k_2,a)=f_1(k_1,a)f_2(k_2,a)` The response is
    calculated as:

    .. math::
        \\frac{\\partial P_{u,v}(k)}{\\partial\\delta_L} =
        \\left(\\frac{68}{21}-\\frac{d\\log k^3P_L(k)}{d\\log k}\\right)
        P_L(k)I^1_1(k,|u)I^1_1(k,|v)+I^1_2(k|u,v) - (b_{u} + b_{v})
        P_{u,v}(k)

    where the :math:`I^a_b` are defined in the documentation
    of :meth:`~HMCalculator.I_1_1` and  :meth:`~HMCalculator.I_1_2` and
    :math:`b_{u}` and :math:`b_{v}` are the linear halo biases for quantities
    :math:`u` and :math:`v`, respectively (zero if they are not clustering).

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof1 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
    
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`,
            the power spectrum stored within `cosmo` will be used.
        a_arr (array): an array holding values of the scale factor
            at which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        use_log (bool): if `True`, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: SSC effective trispectrum.
    """
    if lk_arr is None:
        status = 0
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status, cosmo=cosmo)
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status, cosmo=cosmo)

    k_use = np.exp(lk_arr)

    # Check inputs
    if not isinstance(prof1, halos.profiles.HaloProfile):
        raise TypeError("prof1 must be of type `HaloProfile`")
    
    h = cosmo["h"]
    k_emu = k_use / h   # [h/Mpc]
    Omega_m = cosmo["Omega_b"] + cosmo["Omega_c"] + 0.00064/(h**2)
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')
         
    # set cosmology for dark emulator
    emu = darkemu_set_cosmology(cosmo)

    # set h-modified cosmology to take finite differencing
    hp = h + deltah 
    hm = h - deltah 
    cosmo_hp, cosmo_hm = set_hmodified_cosmology(cosmo,deltah)

    emu_p = darkemu_set_cosmology(cosmo_hp)
    emu_m = darkemu_set_cosmology(cosmo_hm)

    # Growth factor                         
    Dp = cosmo_hp.growth_factor_unnorm(a_arr)
    Dm = cosmo_hm.growth_factor_unnorm(a_arr)

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    pk12 = np.zeros([na, nk])
    #dpk34 = np.zeros([na, nk])
    Mfor_hmf = np.linspace(8,17,200)
    Mh = np.linspace(log10Mh_min,log10Mh_max,2**5+1)  # M_sol/h
    M = np.log10(10**Mh/h)
    dM = M[1] - M[0]
    dlogM = dM
    b1_th_tink = np.zeros(len(Mh))
    #b2_th_tink = np.zeros(len(Mh))
    Pth = [0] * len(Mh)
    Pnth_hp = [0] * len(Mh)
    Pnth_hm = [0] * len(Mh)
    Pbin = [0] * len(Mh)
    nths = np.zeros(len(Mh))
    
    mass_def=halos.MassDef200m()
    #mdef_other=halos.MassDef200m()
    
    hmf_DE = halos.MassFuncDarkEmulator(cosmo,mass_def=mass_def) 
    hbf = halos.hbias.HaloBiasTinker10(cosmo,mass_def=mass_def)
    
    #kmax = 2
    if np.any(a_arr < 1/(1+1.48)) or k_use[-1] > kmax:
        #hmf = halos.MassFuncTinker10(cosmo,mass_def=mass_def) 
        nfw = halos.HaloProfileNFW(halos.ConcentrationDuffy08(mass_def),
                                       fourier_analytic=True)
        
        #nfw = halos.HaloProfileNFW(halos.ConcentrationDiemer15_colossus(mass_def),
        #                               fourier_analytic=True)
        hmc = halos.HMCalculator(cosmo, hmf_DE, hbf, mass_def,log10M_min=np.log10(M[0]),log10M_max=np.log10(M[-1]))

        halomod_pk_arr = halos.halomod_power_spectrum(cosmo, hmc, k_use, a_arr,
                                        prof=nfw, prof_2pt=None,
                                        prof2=prof1, p_of_k_a=None,
                                        normprof1=True, normprof2=True,
                                        get_1h=True, get_2h=True,
                                        smooth_transition=None,
                                        supress_1h=None)

        halomod_tk3D, dpk12_halomod = halos.halomod_Tk3D_SSC(cosmo=cosmo, hmc=hmc,
                                              prof1=nfw,
                                              prof2=prof1,
                                              prof12_2pt=None,
                                              normprof1=True, normprof2=True,
                                              lk_arr=np.log(k_use), a_arr=a_arr,
                                              use_log=use_log)

    for ia, aa in enumerate(a_arr):
        z = 1. / aa - 1   # dark emulator is valid for 0 =< z <= 1.48
        if z > 1.48:
            dpk12[ia, :] = dpk12_halomod[ia, :]  
            pk12[ia, :] = halomod_pk_arr[ia, :]
            print("use halo model for z={:.2f}>1.48".format(z))
        else:
            # mass function 
            dndlog10m_emu = ius(Mfor_hmf ,hmf_DE.get_mass_function(cosmo, 10**Mfor_hmf ,aa))  # Mpc^-3  #ius(np.log10(Mlist), dndm_emu * Mlist * np.log(10) * h ** 3)
            
            if Mh[0] < 12.0:  # Msol/h
                    Pth12 = emu.get_phm_massthreshold(k_emu,10**12,z) * (1/h)**3
                    nths12 = emu.mass_to_dens(10**12,z) * h**3
                    Pnth_hp12 = emu_p.get_phm(k_emu*(h/hp),np.log10(nths12*(1/hp)**3),z)*(1/hp)**3
                    Pnth_hm12 = emu_m.get_phm(k_emu*(h/hm),np.log10(nths12*(1/hm)**3),z)*(1/hm)**3
                    Pbin12 = emu.get_phm_mass(k_emu, 10 ** 12, z) * (1/h)**3
#             else:  
#                     Pth[0] = emu.get_phm_massthreshold(k_emu,10**Mh[0],z) * (1/h)**3     
#                     nths[0] = emu.mass_to_dens(10**Mh[0],z) * h**3
#                     Pnth_hp[0] = emu_p.get_phm(k_emu*(h/hp),np.log10(nths[0]*(1/hp)**3),z)*(1/hp)**3
#                     Pnth_hm[0] = emu_m.get_phm(k_emu*(h/hm),np.log10(nths[0]*(1/hm)**3),z)*(1/hm)**3
#                     Pbin[0] = emu.get_phm_mass(k_emu, 10 ** Mh[0], z) * (1/h)**3
                   
            for m in range(0,len(Mh)):
                if Mh[m] < 12.0:  # Msol/h
                    Pth[m] = Pth12 * hbf.get_halo_bias(cosmo,(10 ** M1), aa)
                    Pnth_hp[m] = Pnth_hp[0]
                    Pnth_hm[m] = Pnth_hm[0]
                    Pbin[m] = Pbin[0] 
                else:  
                    Pth[m] = emu.get_phm_massthreshold(k_emu,10**Mh[m],z) * (1/h)**3     
                    nths[m] = emu.mass_to_dens(10**Mh[m],z) * h**3
                    Pnth_hp[m] = emu_p.get_phm(k_emu*(h/hp),np.log10(nths[m]*(1/hp)**3),z)*(1/hp)**3
                    Pnth_hm[m] = emu_m.get_phm(k_emu*(h/hm),np.log10(nths[m]*(1/hm)**3),z)*(1/hm)**3
                    Pbin[m] = emu.get_phm_mass(k_emu, 10 ** Mh[m], z) * (1/h)**3
                  
                    
            
                M1 = np.linspace(M[m], M[-1], 2**5+1)
                dM1 = M[1] - M[0]
                b1_th_tink[m] = integrate.romb(dndlog10m_emu(M1) * hbf.get_halo_bias(cosmo,(10 ** M1), aa), dx = dM1)\
                        /integrate.romb(dndlog10m_emu(M1), dx = dM1)

                
            Nc = prof1._Nc(10 ** M, aa)
            Ns = prof1._Ns(10 ** M, aa)
            fc = prof1._fc(aa)
            Ng =  Nc * (fc + Ns)
            Mps = M + dlogM
            Mms = M - dlogM

            prof_Mp = prof1.fourier(cosmo, k_use, (10 ** Mps), aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):
            prof_Mm = prof1.fourier(cosmo, k_use, (10 ** Mms), aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):
            prof = prof1.fourier(cosmo, k_use,(10 ** M), aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):
            #uk = prof1._usat_fourier(cosmo, k_use,(10 ** M), aa, mass_def) 
            #rho_cr = 2.775*h**2*1e11  # M_solMpc^-3 (w/o h in units)
            #factor_mat = np.tile(10**M/(Omega_m*rho_cr), (len(k_emu), 1)).transpose()
            
            dprof_dlogM = (prof_Mp - prof_Mm) / (2 * dlogM)#*np.log(10))
            nth_mat = np.tile(nths, (len(k_use), 1)).transpose()
            ng = integrate.romb(dndlog10m_emu(M) * Ng, dx = dM, axis = 0)
            bgE = integrate.romb(dndlog10m_emu(M) * Ng * \
            (hbf.get_halo_bias(cosmo,(10 ** M), aa)), dx = dM, axis = 0) / ng

            bgE2 = integrate.romb(dndlog10m_emu(M) * Ng * \
            b2H17(hbf.get_halo_bias(cosmo,(10 ** M), aa)), dx = dM, axis = 0) / ng
            bgL = bgE - 1

            dndlog10m_func_mat = np.tile(dndlog10m_emu(M), (len(k_emu), 1)).transpose()  # M_sol,Mpc^-3
            b1E_mat = np.tile((hbf.get_halo_bias(cosmo,(10 ** M), aa)), (len(k_emu), 1)).transpose()
            
            b1L_th_mat = np.tile(b1_th_tink -1, (len(k_emu), 1)).transpose()
            Pgm = integrate.romb(dprof_dlogM * (nth_mat * np.array(Pth)), \
            dx = dM, axis = 0) / ng   

            dPhm_db_nfix = (26. / 21.) * (np.array(Pnth_hp) - np.array(Pnth_hm)) / \
                        (2 * (np.log(Dp[ia]) - np.log(Dm[ia])))  # Mpc^3

            dnP_hm_db_emu = nth_mat * (dPhm_db_nfix + b1L_th_mat * np.array(Pbin))  # Dless

            dnP_gm_db = integrate.romb(dprof_dlogM * (dnP_hm_db_emu), dx = dM, axis = 0) #Dless
            
            Pgm_growth = dnP_gm_db / ng - bgL * Pgm  # Dless

            Pgm_d = -1. / 3. *  np.gradient(np.log(Pgm)) / np.gradient(np.log(k_use)) * Pgm #Dless
            
            dPgm_db = (Pgm_growth + Pgm_d)
            
            dpklin = pk2dlin.eval_dlogpk_dlogk(k_use, aa, cosmo) 

            Pgm_lin = bgE * pk2dlin.eval(k_use, aa, cosmo)
            dPgm_db_lin = (47/21 + bgE2/bgE - bgE -1/3 * dpklin) * \
                           bgE * pk2dlin.eval(k_use, aa, cosmo)
            
            # stitching
            k_switch = 0.08  # [h/Mpc]
            kmin = 1e-2  # [h/Mpc]
            
            dPgm_db = dPgm_db_lin * np.exp(-k_emu/k_switch) + \
                    (Pgm_growth + Pgm_d) * (1 - np.exp(-k_emu/k_switch))

            Pgm = Pgm_lin * np.exp(-k_emu/k_switch) + \
                  Pgm * (1 - np.exp(-k_emu/k_switch))

            # use linear theory below kmin
            dPgm_db[k_emu < kmin] = dPgm_db_lin[k_emu < kmin]
            dpk12[ia, :] = dPgm_db

            Pgm[k_emu < kmin] = Pgm_lin[k_emu < kmin]
            pk12[ia, :] = Pgm
            
            # use Halo Model above kmax
            if k_use[-1] > kmax:
                #i12 = integrate.romb(dndlog10m_func_mat * b1E_mat * prof * factor_mat * uk, dx = dM, axis = 0) /ng
                
                #i02 = integrate.romb(dndlog10m_func_mat * prof * factor_mat * uk, dx = dM, axis = 0) /ng 
                #HM_1h_resp = i12 - bgE * i02
                k_HM = 1
                dPgm_db1 = dPgm_db * np.exp(-k_use/k_HM) + \
                           HM_1h_resp * (1 - np.exp(-k_use/k_HM))
                dPgm_db[k_use > kmax] = dPgm_db1[k_use > kmax]

                
    if use_log:
        if np.any(dpk12 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "The negative values are substituted by 1e-5.",
                category=CCLWarning)
            np.where(dpk12 <= 0, 1e-5, dpk12)
            
        dpk12 = np.log(dpk12)
            
    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                cosmo=cosmo, is_logp=False)
    
    return dpk12, pk2d


def halomod_Tk3D_SSC(cosmo, prof1, 
                     normprof1=False,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False):

    if lk_arr is None:
        status = 0
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status, cosmo=cosmo)
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status, cosmo=cosmo)

    # Check inputs
    if not isinstance(prof1, halos.profiles.HaloProfile):
        raise TypeError("prof1 must be of type `HaloProfile`")
    
    k_use = np.exp(lk_arr)
    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    
    mass_def=halos.MassDef200m()
    hbf = halos.hbias.HaloBiasTinker10(cosmo,mass_def=mass_def)
    hmf = halos.MassFuncTinker10(cosmo,mass_def=mass_def)              
    nfw = halos.HaloProfileNFW(halos.ConcentrationDuffy08(mass_def),
                                   fourier_analytic=True)
    hmc = halos.HMCalculator(cosmo, hmf, hbf, mass_def)
    
    halomod_tk3D, dpk12_halomod = halos.halomod_Tk3D_SSC(cosmo=cosmo, hmc=hmc,
                                          prof1=nfw,
                                          prof2=prof1,
                                          prof12_2pt=None,
                                          normprof1=True, normprof2=True,
                                          lk_arr=np.log(k_use), a_arr=a_arr,
                                          use_log=use_log)

    for ia, aa in enumerate(a_arr):
        dpk12[ia, :] = dpk12_halomod[ia, :] #np.sqrt(np.diag(halomod_tk3D.eval(k=k_use, a=aa)))

    if use_log:
        if np.any(dpk12 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning)
            use_log = False
        else:
            dpk12 = np.log(dpk12)

    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d



def b2H17(b1):#H17
    b2 = 0.77 - (2.43 * b1) + ( b1 * b1) 
    return b2


def b2L16(b1):#L16
    b2 = 0.412 - (2.143 * b1) + (0.929 * b1 * b1) + (0.008 * b1 * b1 * b1)
    return b2

def darkemu_set_cosmology(cosmo):
    Omega_c = cosmo["Omega_c"]
    Omega_b = cosmo["Omega_b"]
    h = cosmo["h"]
    n_s = cosmo["n_s"]
    A_s = cosmo["A_s"]

    omega_c = Omega_c * h ** 2
    omega_b = Omega_b * h ** 2
    omega_nu = 0.00064
    Omega_L = 1 - ((omega_c + omega_b + omega_nu) / h **2)

    emu = darkemu.de_interface.base_class()
    
    #Parameters cparam (numpy array) : Cosmological parameters (𝜔𝑏, 𝜔𝑐, Ω𝑑𝑒, ln(10^10 𝐴𝑠), 𝑛𝑠, 𝑤)  
    cparam = np.array([omega_b,omega_c,Omega_L,np.log(10 ** 10 * A_s),n_s,-1.])
    emu.set_cosmology(cparam)

    return emu

def set_hmodified_cosmology(cosmo,deltah):
    Omega_c = cosmo["Omega_c"]
    Omega_b = cosmo["Omega_b"]
    h = cosmo["h"]
    n_s = cosmo["n_s"]
    A_s = cosmo["A_s"]

    hp = h + deltah 
    Omega_c_p = np.power((h/hp),2) * Omega_c #\Omega_c h^2 is fixed
    Omega_b_p = np.power((h/hp),2) * Omega_b #\Omega_b h^2 is fixed

    hm = h - deltah 
    Omega_c_m = np.power((h/hm),2) * Omega_c #\Omega_c h^2 is fixed
    Omega_b_m = np.power((h/hm),2) * Omega_b #\Omega_b h^2 is fixed

    cosmo_hp = core.Cosmology(Omega_c=Omega_c_p,Omega_b=Omega_b_p,
                            h=hp, n_s=n_s, A_s=A_s)
                            
    cosmo_hm = core.Cosmology(Omega_c=Omega_c_m,Omega_b=Omega_b_m,
                            h=hm, n_s=n_s, A_s=A_s)

    return cosmo_hp, cosmo_hm
                            
def darkemu_Tk3D_SSC_test(cosmo, prof1, deltah=0.02,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     normprof1=False,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False):
   
    if lk_arr is None:
        status = 0
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status, cosmo=cosmo)
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status, cosmo=cosmo)

    k_use = np.exp(lk_arr)

    # Check inputs
    if not isinstance(prof1, halos.profiles.HaloProfile):
        raise TypeError("prof1 must be of type `HaloProfile`")
    
    h = cosmo["h"]
    k_emu = k_use / h   # [h/Mpc]

    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')
         
    # set cosmology for dark emulator
    emu = darkemu_set_cosmology(cosmo)

    # set h-modified cosmology to take finite differencing
    hp = h + deltah 
    hm = h - deltah 
    cosmo_hp, cosmo_hm = set_hmodified_cosmology(cosmo,deltah)

    emu_p = darkemu_set_cosmology(cosmo_hp)
    emu_m = darkemu_set_cosmology(cosmo_hm)

    # Growth factor                         
    Dp = cosmo_hp.growth_factor_unnorm(a_arr)
    Dm = cosmo_hm.growth_factor_unnorm(a_arr)

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    pk12 = np.zeros([na, nk])
    #dpk34 = np.zeros([na, nk])
    Mfor_hmf = np.linspace(8,16,200)
    Mh = np.linspace(log10Mh_min,log10Mh_max,2**5+1)#M_sol/h
    dMh = Mh[1] - Mh[0]
    dlogM = dMh
    b1_th_tink = np.zeros(len(Mh))
    #b2_th_tink = np.zeros(len(Mh))
    Pth = [0] * len(Mh)
    Pnth_hp = [0] * len(Mh)
    Pnth_hm = [0] * len(Mh)
    Pbin = [0] * len(Mh)
    nths = np.zeros(len(Mh))

    mass_def=halos.MassDef200m()
    hmf_DE = halos.MassFuncDarkEmulator(cosmo,mass_def=mass_def) 
    hbf = halos.hbias.HaloBiasTinker10(cosmo,mass_def=mass_def)
    
    if np.any(a_arr < 1/(1+1.48)):
        hmf = halos.MassFuncTinker10(cosmo,mass_def=mass_def)              
        nfw = halos.HaloProfileNFW(halos.ConcentrationDuffy08(mass_def),
                                   fourier_analytic=True)
        hmc = halos.HMCalculator(cosmo, hmf, hbf, mass_def)
    
        halomod_pk_arr = halos.halomod_power_spectrum(cosmo, hmc, k_use, a_arr,
                                    prof=nfw, prof_2pt=None,
                                    prof2=prof1, p_of_k_a=None,
                                    normprof1=True, normprof2=True,
                                    get_1h=True, get_2h=True,
                                    smooth_transition=None,
                                    supress_1h=None)

        halomod_tk3D, dpk12_halomod = halos.halomod_Tk3D_SSC(cosmo=cosmo, hmc=hmc,
                                          prof1=nfw,
                                          prof2=prof1,
                                          prof12_2pt=None,
                                          normprof1=True, normprof2=True,
                                          lk_arr=np.log(k_use), a_arr=a_arr,
                                          use_log=use_log)

    for ia, aa in enumerate(a_arr):
        z = 1. / aa - 1   # dark emulator is valid for 0 =< z <= 1.48
        if z > 1.48:
            dpk12[ia, :] = dpk12_halomod[ia, :]  
            pk12[ia, :] = halomod_pk_arr[ia, :]
            print("use halo model for z={:.2f}>1.48".format(z))
        else:
            # mass function 
            #dndlog10m_emu = ius(Mfor_hmf ,hmf_DE.get_mass_function(cosmo, 10**Mfor_hmf ,aa))  # Mpc^-3  #ius(np.log10(Mlist), dndm_emu * Mlist * np.log(10) * h ** 3)
            Mlist, dndm_emu = emu.get_dndm(z)  # Mlist [Msol/h]
            dndlog10m_emu = ius(np.log10(Mlist), dndm_emu * Mlist * np.log(10) * h ** 3)


            for  m in range(len(Mh)):
                Pth[m] = emu.get_phm_massthreshold(k_emu,10**Mh[m],z) * (1/h)**3
                nths[m] = emu.mass_to_dens(10**Mh[m],z) * h**3
                
                Pnth_hp[m] = emu_p.get_phm(k_emu*(h/hp),np.log10(nths[m]*(1/hp)**3),z)*(1/hp)**3
                Pnth_hm[m] = emu_m.get_phm(k_emu*(h/hm),np.log10(nths[m]*(1/hm)**3),z)*(1/hm)**3
                Pbin[m] = emu.get_phm_mass(k_emu, 10 ** Mh[m], z) * (1/h)**3
        
                Mh1 = np.linspace(Mh[m],15.9,2**5+1)
                dMh1 = Mh[1] - Mh[0]
                b1_th_tink[m] = integrate.romb(dndlog10m_emu(Mh1) * hbf.get_halo_bias(cosmo,(10 ** Mh1) / h, aa), dx = dMh1)\
                        /integrate.romb(dndlog10m_emu(Mh1), dx = dMh1)

                #b2_th_tink[m] = integrate.romb(dndlog10m_emu(Mh1) * b2H17(hbf.get_halo_bias(cosmo,(10 ** Mh1) / h, aa)), dx = dMh1)\
                #        /integrate.romb(dndlog10m_emu(Mh1), dx = dMh1)

            Nc = prof1._Nc(10 ** Mh / h, aa)
            Ns = prof1._Ns(10 ** Mh / h, aa)
            fc = prof1._fc(aa)
            Ng =  Nc * (fc + Ns)
            Mps = Mh + dlogM
            Mms = Mh - dlogM

            prof_Mp = prof1.fourier(cosmo, k_use, (10 ** Mps) / h, aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):
            prof_Mm = prof1.fourier(cosmo, k_use, (10 ** Mms) / h, aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):
            prof = prof1.fourier(cosmo, k_use,(10 ** Mh) / h, aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):

            dprof_dlogM = (prof_Mp - prof_Mm) / (2 * dlogM)#*np.log(10))
            nth_mat = np.tile(nths, (len(k_use), 1)).transpose()
            ng = integrate.romb(dndlog10m_emu(Mh) * Ng, dx = dMh, axis = 0)
            bgE = integrate.romb(dndlog10m_emu(Mh) * Ng * \
            (hbf.get_halo_bias(cosmo,(10 ** Mh) / h, aa)), dx = dMh, axis = 0) / ng

            bgE2 = integrate.romb(dndlog10m_emu(Mh) * Ng * \
            b2H17(hbf.get_halo_bias(cosmo,(10 ** Mh) / h, aa)), dx = dMh, axis = 0) / ng
            bgL = bgE - 1

            dndlog10m_func_mat = np.tile(dndlog10m_emu(Mh), (len(k_emu), 1)).transpose()  # M_sol,Mpc^-3
            b1L_th_mat = np.tile(b1_th_tink -1, (len(k_emu), 1)).transpose()
            Pgm = integrate.romb(dprof_dlogM * (nth_mat * np.array(Pth)), \
            dx = dMh, axis = 0) / ng   

            dPhm_db_nfix = (26. / 21.) * (np.array(Pnth_hp) - np.array(Pnth_hm)) / \
                        (2 * (np.log(Dp[ia]) - np.log(Dm[ia])))  # Mpc^3

            dnP_hm_db_emu = nth_mat * (dPhm_db_nfix + b1L_th_mat * np.array(Pbin))  # Dless

            # stitching
            k_switch = 0.08  # [h/Mpc]
            kmin = 1e-2  # [h/Mpc]
            dnP_gm_db = integrate.romb(dprof_dlogM * (dnP_hm_db_emu), dx = dMh, axis = 0) #Dless
                        
            Pgm_growth = dnP_gm_db / ng - bgL * Pgm  # Dless

            Pgm_d = -1. / 3. *  np.gradient(np.log(Pgm)) / np.gradient(np.log(k_emu)) * Pgm #Dless

            dpklin = pk2dlin.eval_dlogpk_dlogk(k_use, aa, cosmo) 

            Pgm_lin = bgE * pk2dlin.eval(k_use, aa, cosmo)
            dPgm_db_lin = (47/21 + bgE2/bgE - bgE -1/3 * dpklin) * \
                           bgE * pk2dlin.eval(k_use, aa, cosmo)
            dPgm_db = dPgm_db_lin * np.exp(-k_emu/k_switch) + \
                     (Pgm_growth + Pgm_d) * (1 - np.exp(-k_emu/k_switch))

            Pgm = Pgm_lin * np.exp(-k_emu/k_switch) + \
                  Pgm * (1 - np.exp(-k_emu/k_switch))

            # use linear theory below kmin
            dPgm_db[k_emu < kmin] = dPgm_db_lin[k_emu < kmin]
            dpk12[ia, :] = dPgm_db

            Pgm[k_emu < kmin] = Pgm_lin[k_emu < kmin]
            pk12[ia, :] = Pgm


    if use_log:
        if np.any(dpk12 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning)
            use_log = False
        else:
            dpk12 = np.log(dpk12)
            
    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                cosmo=cosmo, is_logp=False)
    
    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d, pk2d

def darkemu_Tk3D_SSC_old(cosmo, prof1, deltah=0.02, 
                     normprof1=False,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False):
    """ Returns a :class:`~pyccl.tk3d.Tk3D` object containing
    the super-sample covariance trispectrum, given by the tensor
    product of the power spectrum responses associated with the
    two pairs of quantities being correlated. Each response is
    calculated as:

    .. math::
        \\frac{\\partial P_{u,v}(k)}{\\partial\\delta_L} =
        \\left(\\frac{68}{21}-\\frac{d\\log k^3P_L(k)}{d\\log k}\\right)
        P_L(k)I^1_1(k,|u)I^1_1(k,|v)+I^1_2(k|u,v) - (b_{u} + b_{v})
        P_{u,v}(k)

    where the :math:`I^a_b` are defined in the documentation
    of :meth:`~HMCalculator.I_1_1` and  :meth:`~HMCalculator.I_1_2` and
    :math:`b_{u}` and :math:`b_{v}` are the linear halo biases for quantities
    :math:`u` and :math:`v`, respectively (zero if they are not clustering).

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof1 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
        prof2 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_2` above. If `None`,
            `prof1` will be used as `prof2`.
        prof12_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            a profile covariance object returning the the two-point
            moment of `prof1` and `prof2`. If `None`, the default
            second moment will be used, corresponding to the
            products of the means of both profiles.
        prof3 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_1` above. If `None`,
            `prof1` will be used as `prof3`.
        prof4 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`v_2` above. If `None`,
            `prof3` will be used as `prof4`.
        prof34_2pt (:class:`~pyccl.halos.profiles_2pt.Profile2pt`):
            same as `prof12_2pt` for `prof3` and `prof4`.
        normprof1 (bool): if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`
            (see :meth:`~HMCalculator.I_0_1`), where
            :math:`u` is the profile represented by `prof1`.
        normprof2 (bool): same as `normprof1` for `prof2`.
        normprof3 (bool): same as `normprof1` for `prof3`.
        normprof4 (bool): same as `normprof1` for `prof4`.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`): a `Pk2D` object to
            be used as the linear matter power spectrum. If `None`,
            the power spectrum stored within `cosmo` will be used.
        a_arr (array): an array holding values of the scale factor
            at which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the trispectrum should be calculated for
            interpolation. If `None`, the internal values used
            by `cosmo` will be used.
        extrap_order_lok (int): extrapolation order to be used on
            k-values below the minimum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        extrap_order_hik (int): extrapolation order to be used on
            k-values above the maximum of the splines. See
            :class:`~pyccl.tk3d.Tk3D`.
        use_log (bool): if `True`, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).

    Returns:
        :class:`~pyccl.tk3d.Tk3D`: SSC effective trispectrum.
    """
    if lk_arr is None:
        status = 0
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        lk_arr, status = lib.get_pk_spline_lk(cosmo.cosmo, nk, status)
        check(status, cosmo=cosmo)
    if a_arr is None:
        status = 0
        na = lib.get_pk_spline_na(cosmo.cosmo)
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status, cosmo=cosmo)

    k_use = np.exp(lk_arr)

    # Check inputs
    if not isinstance(prof1, halos.profiles.HaloProfile):
        raise TypeError("prof1 must be of type `HaloProfile`")
    
    h = cosmo["h"]
    k_emu = k_use / h   # [h/Mpc]

    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')
         
    # set cosmology for dark emulator
    emu = darkemu_set_cosmology(cosmo)

    # set h-modified cosmology to take finite differencing
    hp = h + deltah 
    hm = h - deltah 
    cosmo_hp, cosmo_hm = set_hmodified_cosmology(cosmo,deltah)

    emu_p = darkemu_set_cosmology(cosmo_hp)
    emu_m = darkemu_set_cosmology(cosmo_hm)

    # Growth factor                         
    Dp = cosmo_hp.growth_factor_unnorm(a_arr)
    Dm = cosmo_hm.growth_factor_unnorm(a_arr)

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    pk12 = np.zeros([na, nk])
    #dpk34 = np.zeros([na, nk])

    Mh = np.linspace(12.,15.9,2**5+1)#M_sol/h
    dMh = Mh[1] - Mh[0]
    dlogM = dMh
    b1_th_tink = np.zeros(len(Mh))
    #b2_th_tink = np.zeros(len(Mh))
    Pth = [0] * len(Mh)
    Pnth_hp = [0] * len(Mh)
    Pnth_hm = [0] * len(Mh)
    Pbin = [0] * len(Mh)
    nths = np.zeros(len(Mh))

    mass_def=halos.MassDef200m()
    hbf = halos.hbias.HaloBiasTinker10(cosmo,mass_def=mass_def)
    hmf = halos.MassFuncTinker10(cosmo,mass_def=mass_def)              
    nfw = halos.HaloProfileNFW(halos.ConcentrationDuffy08(mass_def),
                                   fourier_analytic=True)
    hmc = halos.HMCalculator(cosmo, hmf, hbf, mass_def)
    
    halomod_pk_arr = halos.halomod_power_spectrum(cosmo, hmc, k_use, a_arr,
                                    prof=nfw, prof_2pt=None,
                                    prof2=prof1, p_of_k_a=None,
                                    normprof1=True, normprof2=True,
                                    get_1h=True, get_2h=True,
                                    smooth_transition=None,
                                    supress_1h=None)

    halomod_tk3D, dpk12_halomod = halos.halomod_Tk3D_SSC(cosmo=cosmo, hmc=hmc,
                                          prof1=nfw,
                                          prof2=prof1,
                                          prof12_2pt=None,
                                          normprof1=True, normprof2=True,
                                          lk_arr=np.log(k_use), a_arr=a_arr,
                                          use_log=use_log)

    for ia, aa in enumerate(a_arr):
        z = 1. / aa - 1   # dark emulator is valid for 0 =< z <= 1.48
        if z > 1.48:
            dpk12[ia, :] = dpk12_halomod[ia, :]  
            pk12[ia, :] = halomod_pk_arr[ia, :]
            print("use halo model for z={:.2f}>1.48".format(z))
        else:
            # mass function 
            Mlist, dndm_emu = emu.get_dndm(z)  # Mlist [Msol/h]
            dndlog10m_emu = ius(np.log10(Mlist), dndm_emu * Mlist * np.log(10) * h ** 3)

            for  m in range(len(Mh)):
                Pth[m] = emu.get_phm_massthreshold(k_emu,10**Mh[m],z) * (1/h)**3
                nths[m] = emu.mass_to_dens(10**Mh[m],z) * h**3
                
                Pnth_hp[m] = emu_p.get_phm(k_emu*(h/hp),np.log10(nths[m]*(1/hp)**3),z)*(1/hp)**3
                Pnth_hm[m] = emu_m.get_phm(k_emu*(h/hm),np.log10(nths[m]*(1/hm)**3),z)*(1/hm)**3
                Pbin[m] = emu.get_phm_mass(k_emu, 10 ** Mh[m], z) * (1/h)**3
        
                Mh1 = np.linspace(Mh[m],15.9,2**5+1)
                dMh1 = Mh[1] - Mh[0]
                b1_th_tink[m] = integrate.romb(dndlog10m_emu(Mh1) * hbf.get_halo_bias(cosmo,(10 ** Mh1) / h, aa), dx = dMh1)\
                        /integrate.romb(dndlog10m_emu(Mh1), dx = dMh1)

                #b2_th_tink[m] = integrate.romb(dndlog10m_emu(Mh1) * b2H17(hbf.get_halo_bias(cosmo,(10 ** Mh1) / h, aa)), dx = dMh1)\
                #        /integrate.romb(dndlog10m_emu(Mh1), dx = dMh1)

            Nc = prof1._Nc(10 ** Mh / h, aa)
            Ns = prof1._Ns(10 ** Mh / h, aa)
            fc = prof1._fc(aa)
            Ng =  Nc * (fc + Ns)
            Mps = Mh + dlogM
            Mms = Mh - dlogM

            prof_Mp = prof1.fourier(cosmo, k_use, (10 ** Mps) / h, aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):
            prof_Mm = prof1.fourier(cosmo, k_use, (10 ** Mms) / h, aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):
            prof = prof1.fourier(cosmo, k_use,(10 ** Mh) / h, aa, mass_def) # def _fourier(self, cosmo, k, M, a, mass_def):

            dprof_dlogM = (prof_Mp - prof_Mm) / (2 * dlogM)#*np.log(10))
            nth_mat = np.tile(nths, (len(k_use), 1)).transpose()
            ng = integrate.romb(dndlog10m_emu(Mh) * Ng, dx = dMh, axis = 0)
            bgE = integrate.romb(dndlog10m_emu(Mh) * Ng * \
            (hbf.get_halo_bias(cosmo,(10 ** Mh) / h, aa)), dx = dMh, axis = 0) / ng

            bgE2 = integrate.romb(dndlog10m_emu(Mh) * Ng * \
            b2H17(hbf.get_halo_bias(cosmo,(10 ** Mh) / h, aa)), dx = dMh, axis = 0) / ng
            bgL = bgE - 1

            dndlog10m_func_mat = np.tile(dndlog10m_emu(Mh), (len(k_emu), 1)).transpose()  # M_sol,Mpc^-3
            b1L_th_mat = np.tile(b1_th_tink -1, (len(k_emu), 1)).transpose()
            Pgm = integrate.romb(dprof_dlogM * (nth_mat * np.array(Pth)), \
            dx = dMh, axis = 0) / ng   

            dPhm_db_nfix = (26. / 21.) * (np.array(Pnth_hp) - np.array(Pnth_hm)) / \
                        (2 * (np.log(Dp[ia]) - np.log(Dm[ia])))  # Mpc^3

            dnP_hm_db_emu = nth_mat * (dPhm_db_nfix + b1L_th_mat * np.array(Pbin))  # Dless

            # stitching
            k_switch = 0.08  # [h/Mpc]
            kmin = 1e-2  # [h/Mpc]
            dnP_gm_db = integrate.romb(dprof_dlogM * (dnP_hm_db_emu), dx = dMh, axis = 0) #Dless
                        
            Pgm_growth = dnP_gm_db / ng - bgL * Pgm  # Dless

            Pgm_d = -1. / 3. *  np.gradient(np.log(Pgm)) / np.gradient(np.log(k_emu)) * Pgm #Dless

            dpklin = pk2dlin.eval_dlogpk_dlogk(k_use, aa, cosmo) 

            Pgm_lin = bgE * pk2dlin.eval(k_use, aa, cosmo)
            dPgm_db_lin = (47/21 + bgE2/bgE - bgE -1/3 * dpklin) * \
                           bgE * pk2dlin.eval(k_use, aa, cosmo)
            dPgm_db = dPgm_db_lin * np.exp(-k_emu/k_switch) + \
                     (Pgm_growth + Pgm_d) * (1 - np.exp(-k_emu/k_switch))

            Pgm = Pgm_lin * np.exp(-k_emu/k_switch) + \
                  Pgm * (1 - np.exp(-k_emu/k_switch))

            # use linear theory below kmin
            dPgm_db[k_emu < kmin] = dPgm_db_lin[k_emu < kmin]
            dpk12[ia, :] = dPgm_db

            Pgm[k_emu < kmin] = Pgm_lin[k_emu < kmin]
            pk12[ia, :] = Pgm


    if use_log:
        if np.any(dpk12 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning)
            use_log = False
        else:
            dpk12 = np.log(dpk12)
            
    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                cosmo=cosmo, is_logp=False)
    
    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d, pk2d


