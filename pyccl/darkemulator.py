from . import ccllib as lib

from .pyutils import check, _get_spline2d_arrays, _get_spline3d_arrays
import numpy as np

from . import core
import warnings
from .errors import CCLWarning
from .pk2d import Pk2D
from .tk3d import Tk3D
  
from dark_emulator import darkemu
#from dark_emulator import model_hod
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.special import sici
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
            Mlist, dndm_emu = emu.get_dndm(z)
            dndlog10m_emu = ius(np.log10(Mlist/h), dndm_emu * Mlist * np.log(10) * h ** 3)

            # mass function 
            #dndlog10m_emu = ius(Mfor_hmf ,hmf_DE.get_mass_function(cosmo, 10**Mfor_hmf ,aa))  # Mpc^-3  #ius(np.log10(Mlist), dndm_emu * Mlist * np.log(10) * h ** 3)
            
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

            prof_Mp = prof1.fourier(cosmo, k_use, (10 ** Mps), aa, mass_def) 
            prof_Mm = prof1.fourier(cosmo, k_use, (10 ** Mms), aa, mass_def) 
            prof = prof1.fourier(cosmo, k_use,(10 ** M), aa, mass_def) 

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

def darkemu_pkarr_SSC(cosmo, prof_hod, deltah=0.02,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     log10Mh_pivot=12.5,
                     normprof_hod=False, k_max=2.0,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False, highk_HM=True, surface=False,
                     highz_HMresp=True):
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
        prof_hod (:class:`~pyccl.halos.profiles.HaloProfile`): halo
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
    if not isinstance(prof_hod, halos.profiles.HaloProfile):
        raise TypeError("prof_hod must be of type `HaloProfile`")
    
    h = cosmo["h"]
    k_emu = k_use / h   # [h/Mpc]
    #Omega_m = cosmo["Omega_b"] + cosmo["Omega_c"] + 0.00064/(h**2)
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')
         
    # set cosmology for dark emulator
    emu = darkemu_set_cosmology(cosmo)

    # set h-modified cosmology to take finite differencing
    hp = h + deltah 
    hm = h - deltah 
    cosmo_hp, cosmo_hm = set_hmodified_cosmology(cosmo, deltah)

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
    logMfor_hmf = np.linspace(8,17,200)
    logMh = np.linspace(log10Mh_min,log10Mh_max,2**5+1)  # M_sol/h
    logM = np.log10(10**logMh/h)
    Mh = 10**logMh
    M = 10**logM
    nM = len(M)
    Mh_pivot = 10**log10Mh_pivot  # M_sol/h
    M_pivot = 10**log10Mh_pivot/h  # M_sol
    dlogM = logM[1] - logM[0]
    b1_th_tink = np.zeros(nM)
    #b2_th_tink = np.zeros(nM)
    Pth = np.zeros((nM,nk))
    Pnth_hp = np.zeros((nM,nk))
    Pnth_hm = np.zeros((nM,nk))
    Pbin = np.zeros((nM,nk))
    surface_pgm = np.zeros((nM,nk))
    surface_resp = np.zeros((nM,nk))
    
    nths = np.zeros(nM)
    mass_hp = np.zeros(nM)
    mass_hm = np.zeros(nM)
    
    mass_def=halos.MassDef200m()
    
    hmf_DE = halos.MassFuncDarkEmulator(cosmo, mass_def=mass_def, darkemulator=emu) 
    hbf = halos.hbias.HaloBiasTinker10(cosmo, mass_def=mass_def)
    cM = halos.ConcentrationDiemer15_colossus200m(mass_def)
    #cM_vir = halos.ConcentrationDiemer15_colossus_vir(mass_def)
    
    if np.any(a_arr < 1/(1+1.48)):
        
        nfw = halos.HaloProfileNFW(cM, fourier_analytic=True)
        hmc = halos.HMCalculator(cosmo, hmf_DE, hbf, mass_def, log10M_min=logM[0], log10M_max=logM[-1])

        halomod_pk_arr = halos.halomod_power_spectrum(cosmo, hmc, k_use, a_arr,
                                        prof=nfw, prof_2pt=None,
                                        prof2=prof_hod, p_of_k_a=None,
                                        normprof1=True, normprof2=True,
                                        get_1h=True, get_2h=True,
                                        smooth_transition=None,
                                        supress_1h=None)

    if np.any(a_arr < 1/(1+1.48)) or (highk_HM and k_use[-1] > k_max):
        
        nfw = halos.HaloProfileNFW(cM, fourier_analytic=True)
        hmc = halos.HMCalculator(cosmo, hmf_DE, hbf, mass_def, log10M_min=logM[0], log10M_max=logM[-1])

        halomod_tk3D, dpk12_halomod = halos.halomod_Tk3D_SSC_orig(cosmo=cosmo, hmc=hmc,
                                              prof1=nfw, prof2=prof_hod,
                                              prof3=nfw, prof4=prof_hod,                                           
                                              prof12_2pt=None, prof34_2pt=None,
                                              normprof1=True, normprof2=True,
                                              normprof3=True, normprof4=True,
                                              lk_arr=np.log(k_use), a_arr=a_arr,
                                              use_log=False)

    for ia, aa in enumerate(a_arr):
        z = 1. / aa - 1   # dark emulator is valid for 0 =< z <= 1.48
        if z > 1.48:
            dpk12[ia, :] = dpk12_halomod[ia, :]  
            pk12[ia, :] = halomod_pk_arr[ia, :]
            print("use halo model for z={:.2f}>1.48".format(z))
        else:
            # mass function 
            #Mlist, dndm_emu = emu.get_dndm(z)
            #dndlog10m_emu = ius(np.log10(Mlist/h), dndm_emu * Mlist * np.log(10) * h ** 3)

            # mass function 
            dndlog10m_emu = ius(logMfor_hmf ,hmf_DE.get_mass_function(cosmo, 10**logMfor_hmf ,aa))  # Mpc^-3  
            if logMh[0] < log10Mh_pivot or highz_HMresp:  # Msol/h
                    nfw = halos.HaloProfileNFW(cM, fourier_analytic=True)
        
                    rho_m = cosmo.rho_x(1, "matter", is_comoving=True)  # same for h_plus/minus cosmology
                    hmf_hp = halos.MassFuncDarkEmulator(cosmo_hp, mass_def=mass_def, darkemulator=emu_p) 
                    dndlog10m_emu_hp = ius(logMfor_hmf, hmf_hp.get_mass_function(cosmo_hp, 10**logMfor_hmf ,aa))  # Mpc^-3              
                    hbf_hp = halos.hbias.HaloBiasTinker10(cosmo_hp, mass_def=mass_def)

                    hmf_hm = halos.MassFuncDarkEmulator(cosmo_hm, mass_def=mass_def, darkemulator=emu_m) 
                    dndlog10m_emu_hm = ius(logMfor_hmf, hmf_DE.get_mass_function(cosmo_hm, 10**logMfor_hmf ,aa))  # Mpc^-3 
                    hbf_hm = halos.hbias.HaloBiasTinker10(cosmo_hm, mass_def=mass_def)

            for m in range(nM):
                if logMh[m] < log10Mh_pivot:   # Msol/h
                    nths[m] = mass_to_dens(dndlog10m_emu, cosmo, M[m])               
                    mass_hp[m] = dens_to_mass(dndlog10m_emu_hp, cosmo_hp, nths[m])
                    mass_hm[m] = dens_to_mass(dndlog10m_emu_hm, cosmo_hm, nths[m])
                    
                else:  
                    Pth[m] = emu.get_phm_massthreshold(k_emu, Mh[m], z) * (1/h)**3
                    Pbin[m] = emu.get_phm_mass(k_emu, Mh[m], z) * (1/h)**3
                    
                    #nths[m] = mass_to_dens(dndlog10m_emu, cosmo, M[m])                   
                    nths[m] = emu.mass_to_dens(Mh[m] ,z) * h**3

                    if highz_HMresp and z > 0.5:
                        mass_hp[m] = dens_to_mass(dndlog10m_emu_hp, cosmo_hp, nths[m])
                        mass_hm[m] = dens_to_mass(dndlog10m_emu_hm, cosmo_hm, nths[m])
                    
                        Pnth_hp[m] = Pth_hm_HM_linb(k_use, mass_hp[m], cosmo_hp, dndlog10m_emu_hp, nfw, rho_m, hbf_hp, mass_def, aa)
                        Pnth_hm[m] = Pth_hm_HM_linb(k_use, mass_hm[m], cosmo_hm, dndlog10m_emu_hm, nfw, rho_m, hbf_hm, mass_def, aa)

                    else:
                        Pnth_hp[m] = emu_p.get_phm(k_emu*(h/hp), np.log10(nths[m]*(1/hp)**3), z) * (1/hp)**3
                        Pnth_hm[m] = emu_m.get_phm(k_emu*(h/hm), np.log10(nths[m]*(1/hm)**3), z) * (1/hm)**3
                             
                #logM1 = np.linspace(logM[m], np.log10(10**16./cosmo["h"]), 2**5+1)
                logM1 = np.linspace(logM[m], logM[-1], 2**5+1)
                dlogM1 = logM[1] - logM[0]
                #b1_th_tink[m] = integrate.romb(dndlog10m_emu(logM1) * hbf.get_halo_bias(cosmo, (10 ** logM1), aa), \
                #                dx = dlogM1) / nths[m] 
                b1_th_tink[m] = integrate.romb(dndlog10m_emu(logM1) * hbf.get_halo_bias(cosmo, (10 ** logM1), aa), \
                                dx = dlogM1) / integrate.romb(dndlog10m_emu(logM1), dx = dlogM1)

            if logMh[0] < log10Mh_pivot:  # Msol/h
                Pth[logMh < log10Mh_pivot] = Pth_hm_lowmass_BMO(k_use, M[logMh < log10Mh_pivot], M_pivot, emu, cosmo, dndlog10m_emu, hbf, cM, cM_vir, \
                                                    mass_def, rho_m, aa, b1_th_tink[logMh < log10Mh_pivot])
                Pnth_hp[logMh < log10Mh_pivot] = Pth_hm_lowmass_BMO(k_use, mass_hp[logMh < log10Mh_pivot], M_pivot, emu_p, cosmo_hp, dndlog10m_emu_hp, hbf_hp, cM, cM_vir, \
                                                    mass_def, rho_m, aa)
                Pnth_hm[logMh < log10Mh_pivot] = Pth_hm_lowmass_BMO(k_use, mass_hm[logMh < log10Mh_pivot], M_pivot, emu_m, cosmo_hm, dndlog10m_emu_hm, hbf_hm, cM, cM_vir, \
                                                    mass_def, rho_m, aa)
                Pbin[logMh < log10Mh_pivot] = Pbin_hm_lowmass_BMO(k_use, M[logMh < log10Mh_pivot], M_pivot, emu, cosmo, hbf, cM, cM_vir, mass_def, \
                                                    pk2dlin, rho_m ,aa)

            Nc = prof_hod._Nc(M, aa)
            Ns = prof_hod._Ns(M, aa)
            fc = prof_hod._fc(aa)
            Ng =  Nc * (fc + Ns)
            logMps = logM + dlogM
            logMms = logM - dlogM

            prof_Mp = prof_hod.fourier(cosmo, k_use, (10 ** logMps), aa, mass_def) 
            prof_Mm = prof_hod.fourier(cosmo, k_use, (10 ** logMms), aa, mass_def) 
            prof = prof_hod.fourier(cosmo, k_use, M, aa, mass_def) 
            #uk = prof_hod._usat_fourier(cosmo, k_use,(10 ** M), aa, mass_def) 
            #rho_cr = 2.775*h**2*1e11  # M_solMpc^-3 (w/o h in units)
            #factor_mat = np.tile(10**M/(Omega_m*rho_cr), (len(k_emu), 1)).transpose()
            
            dprof_dlogM = (prof_Mp - prof_Mm) / (2 * dlogM)#*np.log(10))
            nth_mat = np.tile(nths, (len(k_use), 1)).transpose()
            ng = integrate.romb(dndlog10m_emu(logM) * Ng, dx = dlogM, axis = 0)
            bgE = integrate.romb(dndlog10m_emu(logM) * Ng * \
            (hbf.get_halo_bias(cosmo, M, aa)), dx = dlogM, axis = 0) / ng

            bgE2 = integrate.romb(dndlog10m_emu(logM) * Ng * \
            b2H17(hbf.get_halo_bias(cosmo, M, aa)), dx = dlogM, axis = 0) / ng
            bgL = bgE - 1

            dndlog10m_func_mat = np.tile(dndlog10m_emu(logM), (len(k_emu), 1)).transpose()  # M_sol,Mpc^-3
            b1E_mat = np.tile((hbf.get_halo_bias(cosmo, M, aa)), (len(k_emu), 1)).transpose()
            
            b1L_th_mat = np.tile(b1_th_tink -1, (len(k_emu), 1)).transpose()
            
            #dPhm_db_nfix = (26. / 21.) * (np.array(Pnth_hp) - np.array(Pnth_hm)) / \
            #            (2 * (np.log(Dp[ia]) - np.log(Dm[ia])))  # Mpc^3

            dPhm_db_nfix = (26. / 21.) * np.log(np.array(Pnth_hp) / np.array(Pnth_hm)) * np.array(Pth) / \
                        (2 * (np.log(Dp[ia]) - np.log(Dm[ia])))  # Mpc^3

            dnP_hm_db_emu = nth_mat * (dPhm_db_nfix + b1L_th_mat * np.array(Pbin))  # Dless

            Pgm = integrate.romb(dprof_dlogM * (nth_mat * np.array(Pth)), \
            dx = dlogM, axis = 0) / ng   

            dnP_gm_db = integrate.romb(dprof_dlogM * (dnP_hm_db_emu), dx = dlogM, axis = 0) #Dless
            
            if surface:
                surface_pgm[ia, :] = ((prof[0] * nth_mat[0] * np.array(Pth)[0]) - (prof[-1] * nth_mat[-1] * np.array(Pth))[-1]) / ng
                Pgm += surface_pgm[ia, :]
                
                surface_resp[ia, :] = (prof[0] * dnP_hm_db_emu[0]) - (prof[-1] * dnP_hm_db_emu[-1])
                dnP_gm_db += surface_resp[ia, :]

            Pgm_growth = dnP_gm_db / ng - bgL * Pgm  # Dless

            Pgm_d = -1. / 3. *  np.gradient(np.log(Pgm)) / np.gradient(np.log(k_use)) * Pgm #Dless
            
            dPgm_db_emu = (Pgm_growth + Pgm_d)
            
            dpklin = pk2dlin.eval_dlogpk_dlogk(k_use, aa, cosmo) 

            Pgm_lin = bgE * pk2dlin.eval(k_use, aa, cosmo)
            dPgm_db_lin = (47/21 + bgE2/bgE - bgE -1/3 * dpklin) * \
                           bgE * pk2dlin.eval(k_use, aa, cosmo)
            
            # stitching
            k_switch = 0.08  # [h/Mpc]
            
            dPgm_db = dPgm_db_lin * np.exp(-k_emu/k_switch) + \
                    dPgm_db_emu * (1 - np.exp(-k_emu/k_switch))

            Pgm = Pgm_lin * np.exp(-k_emu/k_switch) + \
                  Pgm * (1 - np.exp(-k_emu/k_switch))

            # use linear theory below kmin
            kmin = 1e-2  # [h/Mpc]
            
            dPgm_db[k_emu < kmin] = dPgm_db_lin[k_emu < kmin]
            dpk12[ia, :] = dPgm_db

            Pgm[k_emu < kmin] = Pgm_lin[k_emu < kmin]
            pk12[ia, :] = Pgm
            
            # use Halo Model above k_max
            if highk_HM and k_use[-1] > k_max:
                k_HM = 1  # Mpc^-1
                dPgm_db = dPgm_db * np.exp(-k_use/k_HM) + \
                           dpk12_halomod[ia, :] * (1 - np.exp(-k_use/k_HM))
                dPgm_db[k_use > k_max] = dpk12_halomod[ia, k_use > k_max]

                #dpk_HM = dpk12_halomod[ia, :] - bgE * Pgm
                #dPgm_db = dPgm_db * np.exp(-k_use/k_HM) + \
                #          dpk_HM  * (1 - np.exp(-k_use/k_HM))
                #dPgm_db[k_use > k_max] = dpk_HM[k_use > k_max]

                dpk12[ia, :] = dPgm_db
                
    if use_log:
        if np.any(dpk12 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning)
            use_log = False
        else:
            dpk12 = np.log(dpk12)
        
    #if use_log:
    #    if np.any(dpk12 <= 0):
    ##        warnings.warn(
    #            "Some values were not positive. "
    #            "The negative values are substituted by 1e-5.",
    #            category=CCLWarning)
    #        np.where(dpk12 <= 0, 1e-5, dpk12)
    #        
    #    dpk12 = np.log(dpk12)
            
    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                cosmo=cosmo, is_logp=False)
    
    return dpk12, pk2d, surface_pgm, surface_resp

def darkemu_Pgg_SSC_zresp(cosmo, prof_hod, deltaz=0.1,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     log10Mh_pivot=12.5,
                     normprof_hod=False,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False, surface=False):
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
        prof_hod (:class:`~pyccl.halos.profiles.HaloProfile`): halo
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
    if not isinstance(prof_hod, halos.profiles.HaloProfile):
        raise TypeError("prof_hod must be of type `HaloProfile`")
    
    h = cosmo["h"]
    k_emu = k_use / h   # [h/Mpc]
    #Omega_m = cosmo["Omega_b"] + cosmo["Omega_c"] + 0.00064/(h**2)
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')
         
    # set cosmology for dark emulator
    emu = darkemu_set_cosmology(cosmo)

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    pk12 = np.zeros([na, nk])
    #Gresp2h_nfix = np.zeros([na, nk])
    #Gresp2h_thbin = np.zeros([na, nk])
    Gresp2h = np.zeros([na, nk])
    Gresp1h = np.zeros([na, nk])
    Pgg_2h = np.zeros([na, nk])
    Pgg_1h = np.zeros([na, nk])
    surface_pgg = np.zeros([na, nk])
    surface_resp1 = np.zeros([na, nk])
    surface_resp2 = np.zeros([na, nk])
    
    #dpk34 = np.zeros([na, nk])
    logMfor_hmf = np.linspace(8,17,200)
    logMh = np.linspace(log10Mh_min,log10Mh_max,2**5+1)  # M_sol/h
    logM = np.log10(10**logMh/h)
    Mh = 10**logMh
    M = 10**logM
    nM = len(M)
    Mh_pivot = 10**log10Mh_pivot  # M_sol/h
    M_pivot = 10**log10Mh_pivot/h  # M_sol
    dlogM = logM[1] - logM[0]
    b1_th_tink = np.zeros(nM)
    #b2_th_tink = np.zeros(nM)
    Pth = np.zeros((nM,nM,nk))
    Pth_zp = np.zeros((nM,nM,nk))
    Pth_zm = np.zeros((nM,nM,nk))
    
    Pth_bin = np.zeros((nM,nM,nk))
    nths = np.zeros(nM)
    
    mass_def=halos.MassDef200m()
    
    hmf_DE = halos.MassFuncDarkEmulator(cosmo, mass_def=mass_def, darkemulator=emu) 
    hbf = halos.hbias.HaloBiasTinker10(cosmo, mass_def=mass_def)
    
    for ia, aa in enumerate(a_arr):
        z = 1. / aa - 1   # dark emulator is valid for 0 =< z <= 1.48
        zp = z + deltaz
        zm = z - deltaz
        if zm < 0: zm = 0
        ap = 1/(1+zp)
        am = 1/(1+zm)
        #compute linear growth factor for its derivative of z
        D_ap = cosmo.growth_factor_unnorm(ap)
        D_am = cosmo.growth_factor_unnorm(am)

        if z > 1.5:
            print("dark emulator is valid for z={:.2f}<1.48")
        else:       
            # mass function 
            dndlog10m_emu = ius(logMfor_hmf ,hmf_DE.get_mass_function(cosmo, 10**logMfor_hmf ,aa))  # Mpc^-3  
            
            for m in range(nM):
                    
                nths[m] = mass_to_dens(dndlog10m_emu, cosmo, M[m])                   
                #nths[m] = emu.mass_to_dens(Mh[m] ,z) * h**3

                #logM1 = np.linspace(logM[m], np.log10(10**16./cosmo["h"]), 2**5+1)
                logM1 = np.linspace(logM[m], logM[-1], 2**5+1)
                dlogM1 = logM[1] - logM[0]
                #b1_th_tink[m] = integrate.romb(dndlog10m_emu(logM1) * hbf.get_halo_bias(cosmo, (10 ** logM1), aa), \
                #                dx = dlogM1) / nths[m] 
                b1_th_tink[m] = integrate.romb(dndlog10m_emu(logM1) * hbf.get_halo_bias(cosmo, (10 ** logM1), aa), \
                                dx = dlogM1) / integrate.romb(dndlog10m_emu(logM1), dx = dlogM1)

            for m in range(nM):
                for n in range(nM):
                    Pth[m,n] = emu.get_phh(k_emu, np.log10(nths[m]/(h**3)), np.log10(nths[n]/(h**3)), z) * (1/h)**3 
                    Pth_zp[m,n] = emu.get_phh(k_emu, np.log10(nths[m]/(h**3)), np.log10(nths[n]/(h**3)), zp) * (1/h)**3 
                    Pth_zm[m,n] = emu.get_phh(k_emu, np.log10(nths[m]/(h**3)), np.log10(nths[n]/(h**3)), zm) * (1/h)**3                   
                    Pth_bin[m,n] = emu.get_phh_massthreshold_mass(k_emu, Mh[m], Mh[n], z) * (1/h)**3
             
            Nc = prof_hod._Nc(M, aa)
            Ns = prof_hod._Ns(M, aa)
            fc = prof_hod._fc(aa)
            Ng =  Nc * (fc + Ns)
            logMps = logM + dlogM
            logMms = logM - dlogM

            prof_Mp = prof_hod.fourier(cosmo, k_use, (10 ** logMps), aa, mass_def) 
            prof_Mm = prof_hod.fourier(cosmo, k_use, (10 ** logMms), aa, mass_def) 
            prof = prof_hod.fourier(cosmo, k_use, M, aa, mass_def) 
            uk = prof_hod._usat_fourier(cosmo, k_use, M, aa, mass_def) 
            prof_1h = Nc[:, None] * ((2 * fc * Ns[:, None] * uk) + (Ns[:, None] ** 2 * uk ** 2))

            dprof_dlogM = (prof_Mp - prof_Mm) / (2 * dlogM)#*np.log(10))
            nth_mat = np.tile(nths, (len(k_use), 1)).transpose()
            ng = integrate.romb(dndlog10m_emu(logM) * Ng, dx = dlogM, axis = 0)
            b1 = hbf.get_halo_bias(cosmo, M, aa)
            bgE = integrate.romb(dndlog10m_emu(logM) * Ng * \
            b1, dx = dlogM, axis = 0) / ng

            bgE2 = integrate.romb(dndlog10m_emu(logM) * Ng * \
            b2H17(b1), dx = dlogM, axis = 0) / ng
            bgL = bgE - 1

            dndlog10m_func_mat = np.tile(dndlog10m_emu(logM), (len(k_emu), 1)).transpose()  # M_sol,Mpc^-3
            
            b1L_mat = np.tile(b1-1, (len(k_emu), 1)).transpose()
            b1L_th_mat = np.tile(b1_th_tink -1, (len(k_emu), 1)).transpose()
            
            ### P_gg(k)
            _Pgg_1h = integrate.romb(dndlog10m_func_mat * prof_1h, \
            dx = dlogM, axis = 0) / (ng ** 2)  

            Pgg_2h_int = list()
            for m in range(nM):
                Pgg_2h_int.append(integrate.romb(
                    Pth[m] * nth_mat * dprof_dlogM, axis=0, dx=dlogM))
            Pgg_2h_int = np.array(Pgg_2h_int)
            _Pgg_2h = integrate.romb(
            Pgg_2h_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM)/ (ng ** 2)

            if surface:
                surface1 =  (((prof[-1] * nth_mat[-1]) ** 2 * Pth[-1,-1]) \
                            - ((prof[0] * nth_mat[0]) ** 2 * Pth[0,0]) \
                            - 2 * ((prof[0] * nth_mat[0]) * (prof[-1] * nth_mat[-1]) * Pth[0,-1]) \
                            ) / (ng ** 2)
                
                surface2_int = (((prof[-1] * nth_mat[-1]) * Pth[-1]) \
                            - ((prof[0] * nth_mat[0]) * Pth[0]))
                surface2 = - 2 * integrate.romb(
                    surface2_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM) / (ng ** 2)
                
                surface_pgg[ia, :] = surface1 + surface2
                _Pgg_2h += surface1 + surface2

            Pgg = _Pgg_2h + _Pgg_1h

            ### 2-halo response
            dPhh_db_nfix = (26. / 21.) * (Pth_zp - Pth_zm)/ \
                        (2 * (np.log(D_ap) - np.log(D_am)))  # Mpc^3

            resp_2h_int = list()
            for m in range(nM):
                dP_hh_db_tot = dPhh_db_nfix[m] + 2 * b1L_th_mat * Pth_bin[m] 
                resp_2h_int.append(integrate.romb(
                    dP_hh_db_tot * nth_mat * dprof_dlogM, axis=0, dx=dlogM))
            resp_2h_int = np.array(resp_2h_int)
            resp_2h = integrate.romb(
            resp_2h_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM)/ (ng ** 2)

            #resp_2h_nfix_int = list()
            #resp_2h_thbin_int = list()
            #for m in range(nM):
            #    resp_2h_nfix_int.append(integrate.romb(
            #        dPhh_db_nfix[m] * nth_mat * dprof_dlogM, axis=0, dx=dlogM))
            #    resp_2h_thbin_int.append(integrate.romb(
            #        (2 * b1L_th_mat * Pth_bin[m]) * nth_mat * dprof_dlogM, axis=0, dx=dlogM))
            #resp_2h_nfix_int = np.array(resp_2h_nfix_int)
            #resp_2h_thbin_int = np.array(resp_2h_thbin_int)
            
            #resp_2h_nfix = integrate.romb(
            #resp_2h_nfix_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM)/ (ng ** 2)

            #resp_2h_thbin = integrate.romb(
            #resp_2h_thbin_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM)/ (ng ** 2)

            #resp_2h = resp_2h_nfix + resp_2h_thbin

            if surface:
                surface1_thbin =  (((prof[-1] * nth_mat[-1]) ** 2 * 2 * b1L_th_mat[-1] * Pth_bin[-1,-1]) \
                                - ((prof[0] * nth_mat[0]) ** 2 * 2 * b1L_th_mat[0] * Pth_bin[0,0]) \
                                - 2 * ((prof[0] * nth_mat[0]) * (prof[-1] * nth_mat[-1]) \
                                * ((b1L_th_mat[-1] * Pth_bin[0,-1]) + (b1L_th_mat[0] * Pth_bin[-1,0]))) \
                                ) / (ng ** 2)

                surface1_nfix = (((prof[-1] * nth_mat[-1]) ** 2 * dPhh_db_nfix[-1,-1]) \
                            - ((prof[0] * nth_mat[0]) ** 2 * dPhh_db_nfix[0,0]) \
                            - 2 * ((prof[0] * nth_mat[0]) * (prof[-1] * nth_mat[-1]) * dPhh_db_nfix[0,-1]) \
                            ) / (ng ** 2)
                surface1 = surface1_nfix + surface1_thbin
                
                surface2_int = (prof[-1] * nth_mat[-1]) * ((dPhh_db_nfix[-1]) \
                                + b1L_th_mat[-1] * Pth_bin[:,-1] + b1L_th_mat * Pth_bin[-1]) \
                                - ((prof[0] * nth_mat[0])) * ((dPhh_db_nfix[0]) \
                                + b1L_th_mat[0] * Pth_bin[:,0] + b1L_th_mat * Pth_bin[0])
                surface2 = - 2 * integrate.romb(
                            surface2_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM) / (ng ** 2)
                
                resp_2h += surface1 + surface2
                surface_resp1[ia, :] = surface1 
                surface_resp2[ia, :] = surface2
               
            ### 1-halo response
            resp_1h = integrate.romb(dndlog10m_func_mat * b1L_mat * prof_1h, \
            dx = dlogM, axis = 0) / (ng ** 2)  

            Pgg_growth = (resp_1h + resp_2h) - 2 * bgL * Pgg  

            Pgg_d = -1. / 3. *  np.gradient(np.log(Pgg)) / np.gradient(np.log(k_use)) * Pgg 
            
            dPgg_db_emu = Pgg_growth + Pgg_d - Pgg
            
            dpklin = pk2dlin.eval_dlogpk_dlogk(k_use, aa, cosmo) 

            Pgg_lin = bgE **2 * pk2dlin.eval(k_use, aa, cosmo)
            dPgg_db_lin = (47/21 + 2 * bgE2/bgE - 2 * bgE -1/3 * dpklin) * \
                           Pgg_lin
            # stitching
            k_switch = 0.08  # [h/Mpc]
            
            dPgg_db = dPgg_db_lin * np.exp(-k_emu/k_switch) + \
                    dPgg_db_emu * (1 - np.exp(-k_emu/k_switch))

            Pgg = Pgg_lin * np.exp(-k_emu/k_switch) + \
                  Pgg * (1 - np.exp(-k_emu/k_switch))

            # use linear theory below kmin
            kmin = 1e-2  # [h/Mpc]
            
            dPgg_db[k_emu < kmin] = dPgg_db_lin[k_emu < kmin]
            dpk12[ia, :] = dPgg_db

            Pgg[k_emu < kmin] = Pgg_lin[k_emu < kmin]
            pk12[ia, :] = Pgg            
            dpk12[ia, :] = dPgg_db

            #Gresp2h_nfix[ia, :] = resp_2h_nfix
            #Gresp2h_thbin[ia, :] = resp_2h_thbin
            Gresp2h[ia, :] = resp_2h
            Gresp1h[ia, :] = resp_1h
            Pgg_2h[ia, :] = _Pgg_2h
            Pgg_1h[ia, :] = _Pgg_1h
            
                
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
    
    #return dpk12, pk2d, Gresp2h_nfix, Gresp2h_thbin, Gresp1h, Pgg_2h, Pgg_1h 
    return dpk12, pk2d, Gresp2h, Gresp1h, Pgg_2h, Pgg_1h, surface_pgg, surface_resp1, surface_resp2


def darkemu_Pgg_SSC_Asresp(cosmo, prof_hod, deltalnAs=0.03,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     log10Mh_pivot=12.5,
                     normprof_hod=False,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False, surface=False):
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
        prof_hod (:class:`~pyccl.halos.profiles.HaloProfile`): halo
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
    if not isinstance(prof_hod, halos.profiles.HaloProfile):
        raise TypeError("prof_hod must be of type `HaloProfile`")
    
    h = cosmo["h"]
    k_emu = k_use / h   # [h/Mpc]
    #Omega_m = cosmo["Omega_b"] + cosmo["Omega_c"] + 0.00064/(h**2)
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')
         
    # set cosmology for dark emulator
    emu = darkemu_set_cosmology(cosmo)
    emu_Ap, emu_Am = darkemu_set_cosmology_forAsresp(cosmo, deltalnAs)

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    pk12 = np.zeros([na, nk])
    #Gresp2h_nfix = np.zeros([na, nk])
    #Gresp2h_thbin = np.zeros([na, nk])
    Gresp2h = np.zeros([na, nk])
    Gresp1h = np.zeros([na, nk])
    Pgg_2h = np.zeros([na, nk])
    Pgg_1h = np.zeros([na, nk])
    surface_pgg = np.zeros([na, nk])
    surface_resp = np.zeros([na, nk])
    
    #dpk34 = np.zeros([na, nk])
    logMfor_hmf = np.linspace(8,17,200)
    logMh = np.linspace(log10Mh_min,log10Mh_max,2**5+1)  # M_sol/h
    logM = np.log10(10**logMh/h)
    Mh = 10**logMh
    M = 10**logM
    nM = len(M)
    Mh_pivot = 10**log10Mh_pivot  # M_sol/h
    M_pivot = 10**log10Mh_pivot/h  # M_sol
    dlogM = logM[1] - logM[0]
    b1_th_tink = np.zeros(nM)
    Pth = np.zeros((nM,nM,nk))
    Pth_Ap = np.zeros((nM,nM,nk))
    Pth_Am = np.zeros((nM,nM,nk))
    Pth_bin = np.zeros((nM,nM,nk))
    nths = np.zeros(nM)
    
    mass_def=halos.MassDef200m()
    
    hmf_DE = halos.MassFuncDarkEmulator(cosmo, mass_def=mass_def, darkemulator=emu) 
    hbf = halos.hbias.HaloBiasTinker10(cosmo, mass_def=mass_def)
    
    for ia, aa in enumerate(a_arr):
        z = 1. / aa - 1   # dark emulator is valid for 0 =< z <= 1.48
        
        if z > 1.5:
            print("dark emulator is valid for z={:.2f}<1.48")
        else:       
            # mass function 
            dndlog10m_emu = ius(logMfor_hmf ,hmf_DE.get_mass_function(cosmo, 10**logMfor_hmf ,aa))  # Mpc^-3  
            
            for m in range(nM):
                    
                nths[m] = mass_to_dens(dndlog10m_emu, cosmo, M[m])                   
                #nths[m] = emu.mass_to_dens(Mh[m] ,z) * h**3

                #logM1 = np.linspace(logM[m], np.log10(10**16./cosmo["h"]), 2**5+1)
                logM1 = np.linspace(logM[m], logM[-1], 2**5+1)
                dlogM1 = logM[1] - logM[0]
                #b1_th_tink[m] = integrate.romb(dndlog10m_emu(logM1) * hbf.get_halo_bias(cosmo, (10 ** logM1), aa), \
                #                dx = dlogM1) / nths[m] 
                b1_th_tink[m] = integrate.romb(dndlog10m_emu(logM1) * hbf.get_halo_bias(cosmo, (10 ** logM1), aa), \
                                dx = dlogM1) / integrate.romb(dndlog10m_emu(logM1), dx = dlogM1)
            
            for m in range(nM):
                for n in range(nM):
                    Pth[m,n] = emu.get_phh(k_emu, np.log10(nths[m]/(h**3)), np.log10(nths[n]/(h**3)), z) * (1/h)**3 
                    Pth_Ap[m,n] = emu_Ap.get_phh(k_emu, np.log10(nths[m]/(h**3)), np.log10(nths[n]/(h**3)), z) * (1/h)**3 
                    Pth_Am[m,n] = emu_Am.get_phh(k_emu, np.log10(nths[m]/(h**3)), np.log10(nths[n]/(h**3)), z) * (1/h)**3                   
                    Pth_bin[m,n] = emu.get_phh_massthreshold_mass(k_emu, Mh[m], Mh[n], z) * (1/h)**3
             
            Nc = prof_hod._Nc(M, aa)
            Ns = prof_hod._Ns(M, aa)
            fc = prof_hod._fc(aa)
            Ng =  Nc * (fc + Ns)
            logMps = logM + dlogM
            logMms = logM - dlogM

            prof_Mp = prof_hod.fourier(cosmo, k_use, (10 ** logMps), aa, mass_def) 
            prof_Mm = prof_hod.fourier(cosmo, k_use, (10 ** logMms), aa, mass_def) 
            prof = prof_hod.fourier(cosmo, k_use, M, aa, mass_def) 
            uk = prof_hod._usat_fourier(cosmo, k_use, M, aa, mass_def) 
            prof_1h = Nc[:, None] * ((2 * fc * Ns[:, None] * uk) + (Ns[:, None] ** 2 * uk ** 2))

            dprof_dlogM = (prof_Mp - prof_Mm) / (2 * dlogM)#*np.log(10))
            nth_mat = np.tile(nths, (len(k_use), 1)).transpose()
            ng = integrate.romb(dndlog10m_emu(logM) * Ng, dx = dlogM, axis = 0)
            b1 = hbf.get_halo_bias(cosmo, M, aa)
            bgE = integrate.romb(dndlog10m_emu(logM) * Ng * \
            b1, dx = dlogM, axis = 0) / ng

            bgE2 = integrate.romb(dndlog10m_emu(logM) * Ng * \
            b2H17(b1), dx = dlogM, axis = 0) / ng
            bgL = bgE - 1

            dndlog10m_func_mat = np.tile(dndlog10m_emu(logM), (len(k_emu), 1)).transpose()  # M_sol,Mpc^-3
            
            b1L_mat = np.tile(b1-1, (len(k_emu), 1)).transpose()
            b1L_th_mat = np.tile(b1_th_tink -1, (len(k_emu), 1)).transpose()
            
            ### P_gg(k)
            _Pgg_1h = integrate.romb(dndlog10m_func_mat * prof_1h, \
            dx = dlogM, axis = 0) / (ng ** 2)  

            Pgg_2h_int = list()
            for m in range(nM):
                Pgg_2h_int.append(integrate.romb(
                    Pth[m] * nth_mat * dprof_dlogM, axis=0, dx=dlogM))
            Pgg_2h_int = np.array(Pgg_2h_int)
            _Pgg_2h = integrate.romb(
            Pgg_2h_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM) / (ng ** 2)

            if surface:
                surface1 =  (((prof[-1] * nth_mat[-1]) ** 2 * Pth[-1,-1]) \
                            - ((prof[0] * nth_mat[0]) ** 2 * Pth[0,0]) \
                            - 2 * ((prof[0] * nth_mat[0]) * (prof[-1] * nth_mat[-1]) * Pth[0,-1]) \
                            ) / (ng ** 2)
                
                surface2_int = (((prof[-1] * nth_mat[-1]) * Pth[-1]) \
                            - ((prof[0] * nth_mat[0]) * Pth[0]))
                surface2 = - 2 * integrate.romb(
                    surface2_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM) / (ng ** 2)
                
                surface_pgg[ia, :] = surface1 + surface2
                _Pgg_2h += surface1 + surface2
                

            Pgg = _Pgg_2h + _Pgg_1h

            ### 2-halo response
            dPhh_db_nfix = (26. / 21.) * (Pth_Ap - Pth_Am)/(2 * deltalnAs)
                    
            resp_2h_int = list()
            for m in range(nM):
                dP_hh_db_tot = dPhh_db_nfix[m] + 2 * b1L_th_mat * Pth_bin[m] 
                resp_2h_int.append(integrate.romb(
                    dP_hh_db_tot * nth_mat * dprof_dlogM, axis=0, dx=dlogM))
            resp_2h_int = np.array(resp_2h_int)
            resp_2h = integrate.romb(
            resp_2h_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM)/ (ng ** 2)

            #resp_2h_nfix_int = list()
            #resp_2h_thbin_int = list()
            #for m in range(nM):
            #    resp_2h_nfix_int.append(integrate.romb(
            #        dPhh_db_nfix[m] * nth_mat * dprof_dlogM, axis=0, dx=dlogM))
            #    resp_2h_thbin_int.append(integrate.romb(
            #        (2 * b1L_th_mat * Pth_bin[m]) * nth_mat * dprof_dlogM, axis=0, dx=dlogM))
            #resp_2h_nfix_int = np.array(resp_2h_nfix_int)
            #resp_2h_thbin_int = np.array(resp_2h_thbin_int)
            
            #resp_2h_nfix = integrate.romb(
            #resp_2h_nfix_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM)/ (ng ** 2)

            #resp_2h_thbin = integrate.romb(
            #resp_2h_thbin_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM)/ (ng ** 2)

            #resp_2h = resp_2h_nfix + resp_2h_thbin

            if surface:
                surface1_thbin =  (((prof[-1] * nth_mat[-1]) ** 2 * 2 * b1L_th_mat[-1] * Pth_bin[-1,-1]) \
                            - ((prof[0] * nth_mat[0]) ** 2 * 2 * b1L_th_mat[0] * Pth_bin[0,0]) \
                            - 2 * ((prof[0] * nth_mat[0]) * (prof[-1] * nth_mat[-1]) \
                            * ((b1L_th_mat[-1] * Pth_bin[0,-1]) + (b1L_th_mat[0] * Pth_bin[-1,0]))) \
                            ) / (ng ** 2)

                surface1_nfix = (((prof[-1] * nth_mat[-1]) ** 2 * dPhh_db_nfix[-1,-1]) \
                            - ((prof[0] * nth_mat[0]) ** 2 * dPhh_db_nfix[0,0]) \
                            - 2 * ((prof[0] * nth_mat[0]) * (prof[-1] * nth_mat[-1]) * dPhh_db_nfix[0,-1]) \
                            ) / (ng ** 2)
                surface1 = surface1_nfix + surface1_thbin
                
                surface2_int = (prof[-1] * nth_mat[-1]) * ((dPhh_db_nfix[-1]) \
                                + b1L_th_mat[-1] * Pth_bin[:,-1] + b1L_th_mat * Pth_bin[-1]) \
                                - ((prof[0] * nth_mat[0])) * ((dPhh_db_nfix[0]) \
                                + b1L_th_mat[0] * Pth_bin[:,0] + b1L_th_mat * Pth_bin[0])
                surface2 = - 2 * integrate.romb(
                    surface2_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM) / (ng ** 2)
                
                resp_2h += surface1 + surface2
                surface_resp[ia, :] = surface1 + surface2
                
            ### 1-halo response
            resp_1h = integrate.romb(dndlog10m_func_mat * b1L_mat * prof_1h, \
            dx = dlogM, axis = 0) / (ng ** 2)  


            Pgg_growth = (resp_1h + resp_2h) - 2 * bgL * Pgg  

            Pgg_d = -1. / 3. *  np.gradient(np.log(Pgg)) / np.gradient(np.log(k_use)) * Pgg 
            
            dPgg_db_emu = Pgg_growth + Pgg_d - Pgg
            
            dpklin = pk2dlin.eval_dlogpk_dlogk(k_use, aa, cosmo) 

            Pgg_lin = bgE **2 * pk2dlin.eval(k_use, aa, cosmo)
            dPgg_db_lin = (47/21 + 2 * bgE2/bgE - 2 * bgE -1/3 * dpklin) * \
                           Pgg_lin
            # stitching
            k_switch = 0.08  # [h/Mpc]
            
            dPgg_db = dPgg_db_lin * np.exp(-k_emu/k_switch) + \
                    dPgg_db_emu * (1 - np.exp(-k_emu/k_switch))

            Pgg = Pgg_lin * np.exp(-k_emu/k_switch) + \
                  Pgg * (1 - np.exp(-k_emu/k_switch))

            # use linear theory below kmin
            kmin = 1e-2  # [h/Mpc]
            
            dPgg_db[k_emu < kmin] = dPgg_db_lin[k_emu < kmin]
            dpk12[ia, :] = dPgg_db

            Pgg[k_emu < kmin] = Pgg_lin[k_emu < kmin]
            pk12[ia, :] = Pgg            
            dpk12[ia, :] = dPgg_db

            #Gresp2h_nfix[ia, :] = resp_2h_nfix
            #Gresp2h_thbin[ia, :] = resp_2h_thbin
            Gresp2h[ia, :] = resp_2h
            Gresp1h[ia, :] = resp_1h
            Pgg_2h[ia, :] = _Pgg_2h
            Pgg_1h[ia, :] = _Pgg_1h
            
                
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
    
    #return dpk12, pk2d, Gresp2h_nfix, Gresp2h_thbin, Gresp1h, Pgg_2h, Pgg_1h 
    return dpk12, pk2d, Gresp2h, Gresp1h, Pgg_2h, Pgg_1h, surface_pgg, surface_resp 


def darkemu_pgg(cosmo, prof_hod,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     log10Mh_pivot=12.5,
                     normprof_hod=False, k_max=2.0,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False):
    """ Returns a 2D array with shape `[na,nk]` describing the
    first function :math:`f_1(k,a)` that makes up a factorizable
    trispectrum :math:`T(k_1,k_2,a)=f_1(k_1,a)f_2(k_2,a)` The response is
    calculated as:

    .. math::
        \\frac{\\partial P_{u,v}(k)}{\\partial\\delta_L} =
        \\left(\\frac{68}{21}-\\frac{d\\log k^3P_L(k)}{d\\log k}\\right)
        P_L(k)I^1_1(k,|u)I^1_1(k,|v)+I^1_2(k|u,v) - (b_{u} + b_{v})
        P_{u,v}(k)

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof_hod (:class:`~pyccl.halos.profiles.HaloProfile`): halo
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
    if not isinstance(prof_hod, halos.profiles.HaloProfile):
        raise TypeError("prof_hod must be of type `HaloProfile`")
    
    h = cosmo["h"]
    k_emu = k_use / h   # [h/Mpc]
    #Omega_m = cosmo["Omega_b"] + cosmo["Omega_c"] + 0.00064/(h**2)
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')
         
    # set cosmology for dark emulator
    emu = darkemu_set_cosmology(cosmo)

    na = len(a_arr)
    nk = len(k_use)
    pk12 = np.zeros([na, nk])
    pk12_1h = np.zeros([na, nk])
    pk12_2h = np.zeros([na, nk])
    
    logMfor_hmf = np.linspace(8,17,200)
    logMh = np.linspace(log10Mh_min,log10Mh_max,2**5+1)  # M_sol/h
    logM = np.log10(10**logMh/h)
    Mh = 10**logMh
    M = 10**logM
    nM = len(M)
    Mh_pivot = 10**log10Mh_pivot  # M_sol/h
    M_pivot = 10**log10Mh_pivot/h  # M_sol
    dlogM = logM[1] - logM[0]
    Pth = np.zeros((nM,nM,nk))
    #dprof_dlogM_mat =  np.zeros((nM,nM,nk))
    nths = np.zeros(nM)
    mass_def=halos.MassDef200m()
    
    hmf_DE = halos.MassFuncDarkEmulator(cosmo, mass_def=mass_def, darkemulator=emu) 
    hbf = halos.hbias.HaloBiasTinker10(cosmo, mass_def=mass_def)
     
    for ia, aa in enumerate(a_arr):
        z = 1. / aa - 1   # dark emulator is valid for 0 =< z <= 1.48       
        # mass function 
        dndlog10m_emu = ius(logMfor_hmf ,hmf_DE.get_mass_function(cosmo, 10**logMfor_hmf ,aa))  # Mpc^-3  
                    
        for m in range(nM):
            #nths[m] = emu.mass_to_dens(Mh[m] ,z) * h**3
            nths[m] = mass_to_dens(dndlog10m_emu, cosmo, M[m])               
                    
        for m in range(nM):
            for n in range(nM):
                Pth[m,n] = emu.get_phh(k_emu, np.log10(nths[m]/(h**3)), np.log10(nths[n]/(h**3)), z) * (1/h)**3 
                                     
        Nc = prof_hod._Nc(M, aa)
        Ns = prof_hod._Ns(M, aa)
        fc = prof_hod._fc(aa)
        Ng =  Nc * (fc + Ns)
        logMps = logM + dlogM
        logMms = logM - dlogM

        prof_Mp = prof_hod.fourier(cosmo, k_use, (10 ** logMps), aa, mass_def) 
        prof_Mm = prof_hod.fourier(cosmo, k_use, (10 ** logMms), aa, mass_def) 
        uk = prof_hod._usat_fourier(cosmo, k_use, M, aa, mass_def) 
        prof_1h = Nc[:, None] * ((2 * fc * Ns[:, None] * uk) + (Ns[:, None] ** 2 * uk ** 2))

        dprof_dlogM = (prof_Mp - prof_Mm) / (2 * dlogM)#*np.log(10))
        #for m in range(nM):
        #    dprof_dlogM_mat[m] = dprof_dlogM
        nth_mat = np.tile(nths, (len(k_use), 1)).transpose()
        ng = integrate.romb(dndlog10m_emu(logM) * Ng, dx = dlogM, axis = 0)
        bgE = integrate.romb(dndlog10m_emu(logM) * Ng * \
            (hbf.get_halo_bias(cosmo, M, aa)), dx = dlogM, axis = 0) / ng

        dndlog10m_func_mat = np.tile(dndlog10m_emu(logM), (len(k_emu), 1)).transpose()  # M_sol,Mpc^-3
        
        Pgg_1h = integrate.romb(dndlog10m_func_mat * prof_1h, \
            dx = dlogM, axis = 0) / (ng ** 2)  

        Pgg_2h_int = list()
        for m in range(nM):
            Pgg_2h_int.append(integrate.romb(
                Pth[m] * nth_mat * dprof_dlogM, axis=0, dx=dlogM))
        Pgg_2h_int = np.array(Pgg_2h_int)
        Pgg_2h = integrate.romb(
        Pgg_2h_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM)/ (ng ** 2)

        Pgg = Pgg_2h + Pgg_1h
        pk12_1h[ia, :] = Pgg_1h
        pk12_2h[ia, :] = Pgg_2h
        
        Pgg_lin = bgE**2 * pk2dlin.eval(k_use, aa, cosmo)
            
        # stitching
        k_switch = 0.08  # [h/Mpc]
            
        Pgg = Pgg_lin * np.exp(-k_emu/k_switch) + \
                  Pgg * (1 - np.exp(-k_emu/k_switch))

        # use linear theory below kmin
        kmin = 1e-2  # [h/Mpc]
            
        Pgg[k_emu < kmin] = Pgg_lin[k_emu < kmin]
        pk12[ia, :] = Pgg
            
            
    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                cosmo=cosmo, is_logp=False)
    
    return pk2d, pk12, pk12_1h, pk12_2h, bgE

def darkemu_pgg_massbin(cosmo, prof_hod,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     log10Mh_pivot=12.5,
                     normprof_hod=False, k_max=2.0,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False):
    """ Returns a 2D array with shape `[na,nk]` describing the
    first function :math:`f_1(k,a)` that makes up a factorizable
    trispectrum :math:`T(k_1,k_2,a)=f_1(k_1,a)f_2(k_2,a)` The response is
    calculated as:

    .. math::
        \\frac{\\partial P_{u,v}(k)}{\\partial\\delta_L} =
        \\left(\\frac{68}{21}-\\frac{d\\log k^3P_L(k)}{d\\log k}\\right)
        P_L(k)I^1_1(k,|u)I^1_1(k,|v)+I^1_2(k|u,v) - (b_{u} + b_{v})
        P_{u,v}(k)

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        hmc (:class:`HMCalculator`): a halo model calculator.
        prof_hod (:class:`~pyccl.halos.profiles.HaloProfile`): halo
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
    if not isinstance(prof_hod, halos.profiles.HaloProfile):
        raise TypeError("prof_hod must be of type `HaloProfile`")
    
    h = cosmo["h"]
    k_emu = k_use / h   # [h/Mpc]
    #Omega_m = cosmo["Omega_b"] + cosmo["Omega_c"] + 0.00064/(h**2)
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')
         
    # set cosmology for dark emulator
    emu = darkemu_set_cosmology(cosmo)

    na = len(a_arr)
    nk = len(k_use)
    pk12 = np.zeros([na, nk])
    pk12_1h = np.zeros([na, nk])
    pk12_2h = np.zeros([na, nk])
    
    logMfor_hmf = np.linspace(8,17,200)
    logMh = np.linspace(log10Mh_min,log10Mh_max,2**5+1)  # M_sol/h
    logM = np.log10(10**logMh/h)
    Mh = 10**logMh
    M = 10**logM
    nM = len(M)
    Mh_pivot = 10**log10Mh_pivot  # M_sol/h
    M_pivot = 10**log10Mh_pivot/h  # M_sol
    dlogM = logM[1] - logM[0]
    Pbin = np.zeros((nM,nM,nk))
    mass_def=halos.MassDef200m()
    
    hmf_DE = halos.MassFuncDarkEmulator(cosmo, mass_def=mass_def, darkemulator=emu) 
    hbf = halos.hbias.HaloBiasTinker10(cosmo, mass_def=mass_def)
     
    for ia, aa in enumerate(a_arr):
        z = 1. / aa - 1   # dark emulator is valid for 0 =< z <= 1.48       
        # mass function 
        dndlog10m_emu = ius(logMfor_hmf ,hmf_DE.get_mass_function(cosmo, 10**logMfor_hmf ,aa))  # Mpc^-3  
                        
        for m in range(nM):
            for n in range(nM):
                Pbin[m,n] = emu.get_phh_mass(k_emu, Mh[m], Mh[n], z) * (1/h)**3 
                                     
        Nc = prof_hod._Nc(M, aa)
        Ns = prof_hod._Ns(M, aa)
        fc = prof_hod._fc(aa)
        Ng =  Nc * (fc + Ns)
        uk = prof_hod._usat_fourier(cosmo, k_use, M, aa, mass_def) 
        prof = prof_hod.fourier(cosmo, k_use, M, aa, mass_def) 

        prof_1h = Nc[:, None] * ((2 * fc * Ns[:, None] * uk) + (Ns[:, None] ** 2 * uk ** 2))

        ng = integrate.romb(dndlog10m_emu(logM) * Ng, dx = dlogM, axis = 0)
        bgE = integrate.romb(dndlog10m_emu(logM) * Ng * \
            (hbf.get_halo_bias(cosmo, M, aa)), dx = dlogM, axis = 0) / ng

        dndlog10m_func_mat = np.tile(dndlog10m_emu(logM), (len(k_emu), 1)).transpose()  # M_sol,Mpc^-3
        
        Pgg_1h = integrate.romb(dndlog10m_func_mat * prof_1h, \
            dx = dlogM, axis = 0) / (ng ** 2)  

        
        Pgg_2h_int = list()
        for m in range(nM):
            Pgg_2h_int.append(integrate.romb(
                Pbin[m] * dndlog10m_func_mat * prof, axis=0, dx=dlogM))
        Pgg_2h_int = np.array(Pgg_2h_int)
        Pgg_2h = integrate.romb(
        Pgg_2h_int * dndlog10m_func_mat * prof, axis=0, dx=dlogM)/ (ng ** 2)

        Pgg = Pgg_2h + Pgg_1h
        pk12_1h[ia, :] = Pgg_1h
        pk12_2h[ia, :] = Pgg_2h
        
        Pgg_lin = bgE**2 * pk2dlin.eval(k_use, aa, cosmo)
            
        # stitching
        k_switch = 0.08  # [h/Mpc]
            
        Pgg = Pgg_lin * np.exp(-k_emu/k_switch) + \
                  Pgg * (1 - np.exp(-k_emu/k_switch))

        # use linear theory below kmin
        kmin = 1e-2  # [h/Mpc]
            
        Pgg[k_emu < kmin] = Pgg_lin[k_emu < kmin]
        pk12[ia, :] = Pgg
            
            
    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                cosmo=cosmo, is_logp=False)
    
    return pk2d, pk12, pk12_1h, pk12_2h


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

def Pth_hm_HM_linb(k, M, cosmo, dndlog10m_emu, nfw, rho_m, hbf, mass_def, a):
    
    logM1 = np.linspace(np.log10(M), np.log10(10**15.9/cosmo["h"]), 2**5+1)
    dlogM1 = logM1[1] - logM1[0]
    dens = integrate.romb(dndlog10m_emu(logM1), dx = dlogM1)
    dndlog10m_func_mat = np.tile(dndlog10m_emu(logM1), (len(k), 1)).transpose()  # M_sol,Mpc^-3

    # 1 halo term 
    rho_h = nfw.fourier(cosmo, k, 10**logM1, a, mass_def)   
    P1h = integrate.romb(dndlog10m_func_mat * rho_h/rho_m, dx = dlogM1, axis=0)/dens
    
    # 2 halo term 
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')    
    pklin = pk2dlin.eval(k,a)
    b1_th = integrate.romb(dndlog10m_func_mat * hbf.get_halo_bias(cosmo,(10 ** logM1), a)[:, None] , dx = dlogM1, axis=0)\
                           /dens   
    P2h = b1_th * pklin
                                  
    P_HM = P1h + P2h
    
    return P_HM


def Pth_hm_lowmass_HM_linb(k, M, M_pivot, emu, cosmo, hmf, hbf, nfw, mass_def, a):
    # pivot mass (Dark Emulator)
    z = 1/a -1
    Pth_hm_pivot = emu.get_phm_massthreshold(k/cosmo["h"], M_pivot*cosmo["h"], z) / (cosmo["h"]**3)
    
    rho_m = ccl.rho_x(cosmo, a, "matter", is_comoving=True)
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')    
    pklin = pk2dlin.eval(k,a)
    
    Mfor_hmf = np.linspace(8,17,600)  # Msol
    dndlog10m_emu = ius(Mfor_hmf ,hmf.get_mass_function(cosmo, 10**Mfor_hmf ,a))  # Mpc^-3 
    
    M1 = np.linspace(np.log10(M), np.log10(10**15.9/cosmo["h"]), 2**5+1)
    dM1 = M1[1] - M1[0]
    dens = integrate.romb(dndlog10m_emu(M1), dx = dM1)
    dndlog10m_func_mat = np.tile(dndlog10m_emu(M1), (len(k), 1)).transpose()  # M_sol,Mpc^-3

    # 1 halo term 
    rho_h = nfw.fourier(cosmo, k, 10**M1, a, mass_def)
    
    P1h = integrate.romb(dndlog10m_func_mat * rho_h/rho_m, dx = dM1, axis=0)/dens
    
    # 2 halo term 
    b1_th = integrate.romb(dndlog10m_func_mat * hbf.get_halo_bias(cosmo,(10 ** M1), a)[:, None] , dx = dM1, axis=0)\
                           /dens
     
    P2h = b1_th * pklin
        
    M1 = np.linspace(np.log10(M_pivot), np.log10(10**15.9/cosmo["h"]), 2**5+1)
    dM1 = M1[1] - M1[0]
    dens = integrate.romb(dndlog10m_emu(M1), dx = dM1)
    
    dndlog10m_func_mat = np.tile(dndlog10m_emu(M1), (len(k), 1)).transpose()  # M_sol,Mpc^-3

    rho_h_pivot = nfw.fourier(cosmo, k, 10**M1, a, mass_def)
    
    P1h_pivot = integrate.romb(dndlog10m_func_mat * rho_h_pivot/rho_m, dx = dM1, axis=0)/dens   
                               
    # 2 halo term
    b1_th_pivot = integrate.romb(dndlog10m_func_mat * hbf.get_halo_bias(cosmo,(10 ** M1), a)[:, None] , dx = dM1, axis=0)\
                        /dens
    
    P2h_pivot = b1_th_pivot * pklin
 
    P_HM = P1h + P2h
    P_HM_pivot = P1h_pivot + P2h_pivot
    
    # rescaling
    Pth_hm = Pth_hm_pivot * (P_HM/P_HM_pivot)
    
    return Pth_hm

def Pth_hm_lowmass_BMO(k, M, M_pivot, emu, cosmo, dndlog10m_emu, hbf, cM, cM_vir, mass_def, rho_m, a, b1_th_tink=None):
    M_use = np.atleast_1d(M)
    k_use = np.atleast_1d(k)
    P1h = np.zeros((len(M_use),len(k_use)))
    if b1_th_tink is None:
        b1_th = np.zeros(len(M_use))
    else:
        b1_th = b1_th_tink
    for i in range(len(M_use)):
        logM1 = np.linspace(np.log10(M_use[i]), np.log10(10**15.9/cosmo["h"]), 2**5+1)
        dlogM1 = logM1[1] - logM1[0]
        M1 = 10**logM1
        dens = integrate.romb(dndlog10m_emu(logM1), dx = dlogM1)
        if b1_th_tink is None:
            b1_th[i] = integrate.romb(dndlog10m_emu(logM1) * hbf.get_halo_bias(cosmo, M1, a), dx = dlogM1)\
                                /dens
        dndlog10m_func_mat = np.tile(dndlog10m_emu(logM1), (len(k_use), 1)).transpose()  # M_sol,Mpc^-3

        # 1 halo term 
        P1h_bin = (M1[:, None]/rho_m) * u_M(k_use, M1, cosmo, cM, cM_vir, mass_def, a)
        
        P1h[i] = integrate.romb(dndlog10m_func_mat * P1h_bin, dx = dlogM1, axis=0)/dens

    # pivot mass
    Pth_pivot = emu.get_phm_massthreshold(k_use / cosmo["h"], M_pivot * cosmo["h"], 1./a -1) * (1/cosmo["h"])**3
                                       
    logM1 = np.linspace(np.log10(M_pivot), np.log10(10**15.9/cosmo["h"]), 2**5+1)
    dM1 = logM1[1] - logM1[0]
    M1 = 10**logM1
    dens = integrate.romb(dndlog10m_emu(logM1), dx = dlogM1)
    
    b1_th_pivot = integrate.romb(dndlog10m_emu(logM1) * hbf.get_halo_bias(cosmo, M1, a), dx = dlogM1)\
                        /dens
    dndlog10m_func_mat = np.tile(dndlog10m_emu(logM1), (len(k_use), 1)).transpose()  # M_sol,Mpc^-3

    P1h_bin_pivot = (M1[:, None]/rho_m) * u_M(k_use, M1, cosmo, cM, cM_vir, mass_def, a)
    
    P1h_pivot = integrate.romb(dndlog10m_func_mat * P1h_bin_pivot, dx = dlogM1, axis=0)/dens   
                             
    # 2 halo term
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')
    pklin = pk2dlin.eval(k_use,a)
    
    P2h = b1_th[:, None] * pklin[None, :]
    P2h_pivot = b1_th_pivot * pklin
    
    P_HM = P1h + P2h
    P_HM_pivot = P1h_pivot + P2h_pivot
    
    # rescaling
    Pth = Pth_pivot[None, :] * (P_HM/P_HM_pivot[None, :])
    
    if np.ndim(k) == 0:
            Pth = np.squeeze(Pth, axis=-1)
    if np.ndim(M) == 0:
            Pth = np.squeeze(Pth, axis=0)
    
    return Pth

def Pbin_hm_lowmass_BMO_Mvector(k, M, M_pivot, emu, cosmo, hbf, cM, mass_def, a, tau_v):
    M_use = np.atleast_1d(M)
    k_use = np.atleast_1d(k)

    # pivot mass (Dark Emulator)
    z = 1/a -1
    Pbin_hm_pivot = emu.get_phm_mass(k/cosmo["h"], M_pivot*cosmo["h"], z) / (cosmo["h"]**3)
    
    # 1 halo term 
    rho_m = ccl.rho_x(cosmo, a, "matter", is_comoving=True)
    P1h = (M_use[:, None]/rho_m) * u_M(k_use, M_use, cosmo, cM, mass_def, a, tau_v)
    P1h_pivot = (M_pivot/rho_m) * u_M(k_use, M_pivot, cosmo, cM, mass_def, a, tau_v)
    
    # 2 halo term
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power('delta_matter:delta_matter')    
    pklin = pk2dlin.eval(k,a)
    b1 = hbf.get_halo_bias(cosmo, M_use, a)
    b1_pivot = hbf.get_halo_bias(cosmo, M_pivot, a)
    
    P2h = b1[:, None] * pklin[None, :]
    P2h_pivot = b1_pivot * pklin
    
    P_HM = P1h + P2h
    P_HM_pivot = P1h_pivot + P2h_pivot
    
    # rescaling
    Pbin_hm = Pbin_hm_pivot[None, :] * (P_HM/P_HM_pivot[None, :])
    
    return Pbin_hm, P1h, P2h,  P_HM, P1h_pivot, P2h_pivot, P_HM_pivot

def Pbin_hm_lowmass_BMO(k, M, M_pivot, emu, cosmo, hbf, cM, cM_vir, mass_def, pk2dlin, rho_m ,a):
    M_use = np.atleast_1d(M)
    k_use = np.atleast_1d(k)

    # 1 halo term 
    P1h = (M_use[:, None]/rho_m) * u_M(k_use, M_use, cosmo, cM, cM_vir, mass_def, a)
    P1h_pivot = (M_pivot/rho_m) * u_M(k_use, M_pivot, cosmo, cM, cM_vir, mass_def, a)
    
    # 2 halo term
    pklin = pk2dlin.eval(k_use,a)
    b1 = hbf.get_halo_bias(cosmo, M_use, a)
    b1_pivot = hbf.get_halo_bias(cosmo, M_pivot, a)
    
    P2h = b1[:, None] * pklin[None, :]
    P2h_pivot = b1_pivot * pklin
    
    P_HM = P1h + P2h
    P_HM_pivot = P1h_pivot + P2h_pivot
    
    # rescaling
    Pbin_pivot = emu.get_phm_mass(k_use / cosmo["h"], M_pivot * cosmo["h"], 1./a -1) * (1/cosmo["h"])**3
                    
    Pbin = Pbin_pivot[None, :] * (P_HM/P_HM_pivot[None, :])
    
    if np.ndim(k) == 0:
            Pbin = np.squeeze(Pbin, axis=-1)
    if np.ndim(M) == 0:
            Pbin = np.squeeze(Pbin, axis=0)
    
    return Pbin


def mass_to_dens(dndlog10m_emu, cosmo, mass_thre):
    logM1 = np.linspace(np.log10(mass_thre), np.log10(10**16./cosmo["h"]), 2**6+1)   
    dlogM1 = logM1[1] - logM1[0]
    dens = integrate.romb(dndlog10m_emu(logM1), dx = dlogM1)
    
    return dens

def dens_to_mass(dndlog10m_emu, cosmo, dens, nint=60):#:, integration="quad"):
        mlist = np.linspace(8, np.log10(10**15.8/cosmo["h"]), nint)
        dlist = np.log(np.array([mass_to_dens(
           dndlog10m_emu, cosmo, 10**mlist[i]) for i in range(nint)]))
        d_to_m_interp = ius(-dlist, mlist)
        return 10**d_to_m_interp(-np.log(dens))



def u_M(k, M, cosmo, cM, cM_vir, mass_def, a):
    M_use = np.atleast_1d(M)
    k_use = np.atleast_1d(k)

    c = cM.get_concentration(cosmo, M_use, a)
    c_vir = cM_vir.get_concentration(cosmo, M_use, a)
    R = mass_def.get_radius(cosmo, M_use, a) / a  # comoving halo radius
        
    r_s = R/c  # scale radius from R_200m, c_200m
    x = k_use[None, :] *r_s[:, None]/a

    tau_v = 2.6
    
    tau1 = tau_v * c_vir
    tau = tau1[:, None]
    m_nfw = np.log(1 + c) - c/(1+c)
    prefactor = tau/(4 * m_nfw[:, None] * (1+tau**2)**3 * x)

    Si, Ci = sici(x) 

    F1 = 2 * (3 * tau**4 - 6 * tau**2 - 1) * P_fit(tau * x)

    F2 = -2 * tau * (tau**4 - 1) * x * Q_fit(tau * x)

    F3 = -2 * tau**2 * np.pi * np.exp(-tau*x) * ((tau**2 + 1) * x + 4 * tau)

    F4 = 2 * tau**3 * (np.pi - 2 * Si) * (4 * np.cos(x) + (tau**2 + 1) * x * np.sin(x))

    F5 = 4 * tau**3 * Ci * (4 * np.sin(x) - (tau**2 + 1) * x * np.cos(x))

    u_M = prefactor * (F1 + F2 + F3 + F4 + F5)
    
    if np.ndim(k) == 0:
            u_M = np.squeeze(u_M, axis=-1)
    if np.ndim(M) == 0:
            u_M = np.squeeze(u_M, axis=0)
            
    return u_M


def P_fit(x):
    a = 1.5652
    b = 3.38723
    c = 6.34891
    d = 0.817677
    e = -0.0895584
    f = 0.877375
    
    gamma = 0.57721566
    
    F1 = - (1/x + (b * x**e)/(c + (x - d)**2))
    F2 = (x**4/(x**4 + a**4))**f
    
    F3 = x * (gamma + np.log(x) - 1)
    F4 = (a**4/(x**4 + a**4))**f
    
    return F1*F2 + F3*F4

def Q_fit(x):
    a = 2.26901
    b = -2839.04
    c = 265.511
    d = -1.12459
    e = -2.90136
    f = 1.86475
    g = 1.52197
    
    gamma = 0.57721566
    
    F1 = 1/x**2 + (b * x**e)/(c + (x - d)**4)
    F2 = (x**4/(x**4 + a**4))**g
    
    F3 = (gamma + np.log(x)) * (1 + x**2 / 2) - 3/4 * x**2
    F4 = (a**4/(x**4 + a**4))**f
    
    return F1*F2 + F3*F4



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


def darkemu_set_cosmology_forAsresp(cosmo, deltalnAs):
    Omega_c = cosmo["Omega_c"]
    Omega_b = cosmo["Omega_b"]
    h = cosmo["h"]
    n_s = cosmo["n_s"]
    A_s = cosmo["A_s"]

    omega_c = Omega_c * h ** 2
    omega_b = Omega_b * h ** 2
    omega_nu = 0.00064
    Omega_L = 1 - ((omega_c + omega_b + omega_nu) / h **2)

    emu_Ap = darkemu.de_interface.base_class()  
    #Parameters cparam (numpy array) : Cosmological parameters (𝜔𝑏, 𝜔𝑐, Ω𝑑𝑒, ln(10^10 𝐴𝑠), 𝑛𝑠, 𝑤)  
    cparam = np.array([omega_b,omega_c,Omega_L,np.log(10 ** 10 * A_s) + deltalnAs, n_s, -1.])
    emu_Ap.set_cosmology(cparam)

    emu_Am = darkemu.de_interface.base_class() 
    #Parameters cparam (numpy array) : Cosmological parameters (𝜔𝑏, 𝜔𝑐, Ω𝑑𝑒, ln(10^10 𝐴𝑠), 𝑛𝑠, 𝑤)  
    cparam = np.array([omega_b,omega_c,Omega_L,np.log(10 ** 10 * A_s) - deltalnAs, n_s, -1.])
    emu_Am.set_cosmology(cparam)

    return emu_Ap, emu_Am


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

            prof_Mp = prof1.fourier(cosmo, k_use, (10 ** Mps) / h, aa, mass_def) 
            prof_Mm = prof1.fourier(cosmo, k_use, (10 ** Mms) / h, aa, mass_def) 
            prof = prof1.fourier(cosmo, k_use,(10 ** Mh) / h, aa, mass_def) 

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

            prof_Mp = prof1.fourier(cosmo, k_use, (10 ** Mps) / h, aa, mass_def) 
            prof_Mm = prof1.fourier(cosmo, k_use, (10 ** Mms) / h, aa, mass_def) 
            prof = prof1.fourier(cosmo, k_use,(10 ** Mh) / h, aa, mass_def) 

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

def halomod_Tk3D_SSC_orig(cosmo, hmc,
                     prof1, prof2=None, prof12_2pt=None,
                     prof3=None, prof4=None, prof34_2pt=None,
                     normprof1=False, normprof2=False,
                     normprof3=False, normprof4=False,
                     p_of_k_a=None, lk_arr=None, a_arr=None,
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
    if not isinstance(prof1, HaloProfile):
        raise TypeError("prof1 must be of type `HaloProfile`")
    if (prof2 is not None) and (not isinstance(prof2, HaloProfile)):
        raise TypeError("prof2 must be of type `HaloProfile` or `None`")
    if (prof3 is not None) and (not isinstance(prof3, HaloProfile)):
        raise TypeError("prof3 must be of type `HaloProfile` or `None`")
    if (prof4 is not None) and (not isinstance(prof4, HaloProfile)):
        raise TypeError("prof4 must be of type `HaloProfile` or `None`")
    if prof12_2pt is None:
        prof12_2pt = Profile2pt()
    elif not isinstance(prof12_2pt, Profile2pt):
        raise TypeError("prof12_2pt must be of type "
                        "`Profile2pt` or `None`")
    if (prof34_2pt is not None) and (not isinstance(prof34_2pt, Profile2pt)):
        raise TypeError("prof34_2pt must be of type `Profile2pt` or `None`")

    # number counts profiles must be normalized
    profs = {prof1: normprof1, prof2: normprof2,
             prof3: normprof3, prof4: normprof4}

    for i, (profile, normalization) in enumerate(profs.items()):
        if (profile is not None
                and profile.is_number_counts
                and not normalization):
            raise ValueError(
                f"normprof{i+1} must be True if prof{i+1} is number counts")

    if prof3 is None:
        prof3_bak = prof1
    else:
        prof3_bak = prof3
    if prof34_2pt is None:
        prof34_2pt_bak = prof12_2pt
    else:
        prof34_2pt_bak = prof34_2pt

    # Power spectrum
    if isinstance(p_of_k_a, Pk2D):
        pk2d = p_of_k_a
    elif (p_of_k_a is None) or (str(p_of_k_a) == 'linear'):
        pk2d = cosmo.get_linear_power('delta_matter:delta_matter')
    elif str(p_of_k_a) == 'nonlinear':
        pk2d = cosmo.get_nonlin_power('delta_matter:delta_matter')
    else:
        raise TypeError("p_of_k_a must be `None`, \'linear\', "
                        "\'nonlinear\' or a `Pk2D` object")

    def get_norm(normprof, prof, sf):
        if normprof:
            return hmc.profile_norm(cosmo, sf, prof)
        else:
            return 1

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    dpk34 = np.zeros([na, nk])
    for ia, aa in enumerate(a_arr):
        # Compute profile normalizations
        norm1 = get_norm(normprof1, prof1, aa)
        i11_1 = hmc.I_1_1(cosmo, k_use, aa, prof1)
        # Compute second profile normalization
        if prof2 is None:
            norm2 = norm1
            i11_2 = i11_1
        else:
            norm2 = get_norm(normprof2, prof2, aa)
            i11_2 = hmc.I_1_1(cosmo, k_use, aa, prof2)
        if prof3 is None:
            norm3 = norm1
            i11_3 = i11_1
        else:
            norm3 = get_norm(normprof3, prof3, aa)
            i11_3 = hmc.I_1_1(cosmo, k_use, aa, prof3)
        if prof4 is None:
            norm4 = norm3
            i11_4 = i11_3
        else:
            norm4 = get_norm(normprof4, prof4, aa)
            i11_4 = hmc.I_1_1(cosmo, k_use, aa, prof4)

        i12_12 = hmc.I_1_2(cosmo, k_use, aa, prof1,
                           prof12_2pt, prof2)
        if (prof3 is None) and (prof4 is None) and (prof34_2pt is None):
            i12_34 = i12_12
        else:
            i12_34 = hmc.I_1_2(cosmo, k_use, aa, prof3_bak,
                               prof34_2pt_bak, prof4)
        norm12 = norm1 * norm2
        norm34 = norm3 * norm4

        pk = pk2d.eval(k_use, aa, cosmo)
        dpk = pk2d.eval_dlogpk_dlogk(k_use, aa, cosmo)
        # (47/21 - 1/3 dlogPk/dlogk) * I11 * I11 * Pk+I12
        dpk12[ia, :] = norm12*((2.2380952381-dpk/3)*i11_1*i11_2*pk+i12_12)
        dpk34[ia, :] = norm34*((2.2380952381-dpk/3)*i11_3*i11_4*pk+i12_34)

        # Counter terms for clustering (i.e. - (bA + bB) * PAB
        if prof1.is_number_counts or (prof2 is None or prof2.is_number_counts):
            b1 = b2 = np.zeros_like(k_use)
            i02_12 = hmc.I_0_2(cosmo, k_use, aa, prof1, prof12_2pt, prof2)
            P_12 = norm12 * (pk * i11_1 * i11_2 + i02_12)

            if prof1.is_number_counts:
                b1 = i11_1 * norm1

            if prof2 is None:
                b2 = b1
            elif prof2.is_number_counts:
                b2 = i11_2 * norm2

            dpk12[ia, :] -= (b1 + b2) * P_12

        if prof3_bak.is_number_counts or \
                ((prof3_bak.is_number_counts and prof4 is None) or
                 (prof4 is not None) and prof4.is_number_counts):
            b3 = b4 = np.zeros_like(k_use)
            if (prof3 is None) and (prof4 is None) and (prof34_2pt is None):
                i02_34 = i02_12
            else:
                i02_34 = hmc.I_0_2(cosmo, k_use, aa, prof3_bak, prof34_2pt_bak,
                                   prof4)
            P_34 = norm34 * (pk * i11_3 * i11_4 + i02_34)

            if prof3 is None:
                b3 = b1
            elif prof3.is_number_counts:
                b3 = i11_3 * norm3

            if prof4 is None:
                b4 = b3
            elif prof4.is_number_counts:
                b4 = i11_4 * norm4

            dpk34[ia, :] -= (b3 + b4) * P_34

    if use_log:
        if np.any(dpk12 <= 0) or np.any(dpk34 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning)
            use_log = False
        else:
            dpk12 = np.log(dpk12)
            dpk34 = np.log(dpk34)

    tk3d = Tk3D(a_arr=a_arr, lk_arr=lk_arr,
                pk1_arr=dpk12, pk2_arr=dpk34,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik, is_logt=use_log)
    return tk3d, dpk12


