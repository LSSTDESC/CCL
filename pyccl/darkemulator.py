from . import ccllib as lib

from .pyutils import check, _get_spline2d_arrays, _get_spline3d_arrays
import numpy as np

from . import core
import warnings
from .errors import CCLWarning
from .pk2d import Pk2D
from .tk3d import Tk3D
  
from dark_emulator import darkemu
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from . import halos

def darkemu_Pgm_Tk3D_SSC(cosmo, prof1, deltah=0.02,
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

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        prof1 (:class:`~pyccl.halos.profiles.HaloProfile`): halo
            profile (corresponding to :math:`u_1` above.
    
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

            dnP_hm_db_emu = nth_mat * (dPhm_db_nfix + b1L_th_mat * np.array(Pbin))  

            # stitching
            k_switch = 0.08  # [h/Mpc]
            kmin = 1e-2  # [h/Mpc]
            dnP_gm_db = integrate.romb(dprof_dlogM * (dnP_hm_db_emu), dx = dM, axis = 0) 
                        
            Pgm_growth = dnP_gm_db / ng - bgL * Pgm  

            Pgm_d = -1. / 3. *  np.gradient(np.log(Pgm)) / np.gradient(np.log(k_use)) * Pgm 

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

def darkemu_Pgm_resp(cosmo, prof_hod, deltah=0.02,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     log10Mh_pivot=12.5,
                     normprof_hod=False, k_max=2.0,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False, highk_HM=True, surface=False,
                     highz_HMresp=True):
    
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
                    
                    nths[m] = emu.mass_to_dens(Mh[m] ,z) * h**3

                    if highz_HMresp and z > 0.5:
                        mass_hp[m] = dens_to_mass(dndlog10m_emu_hp, cosmo_hp, nths[m])
                        mass_hm[m] = dens_to_mass(dndlog10m_emu_hm, cosmo_hm, nths[m])
                    
                        Pnth_hp[m] = Pth_hm_HM_linb(k_use, mass_hp[m], cosmo_hp, dndlog10m_emu_hp, nfw, rho_m, hbf_hp, mass_def, aa)
                        Pnth_hm[m] = Pth_hm_HM_linb(k_use, mass_hm[m], cosmo_hm, dndlog10m_emu_hm, nfw, rho_m, hbf_hm, mass_def, aa)

                    else:
                        Pnth_hp[m] = emu_p.get_phm(k_emu*(h/hp), np.log10(nths[m]*(1/hp)**3), z) * (1/hp)**3
                        Pnth_hm[m] = emu_m.get_phm(k_emu*(h/hm), np.log10(nths[m]*(1/hm)**3), z) * (1/hm)**3
                             
                logM1 = np.linspace(logM[m], logM[-1], 2**5+1)
                dlogM1 = logM[1] - logM[0]
                 
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
            
            dPhm_db_nfix = (26. / 21.) * np.log(np.array(Pnth_hp) / np.array(Pnth_hm)) * np.array(Pth) / \
                        (2 * (np.log(Dp[ia]) - np.log(Dm[ia])))  # Mpc^3

            dnP_hm_db_emu = nth_mat * (dPhm_db_nfix + b1L_th_mat * np.array(Pbin))  

            Pgm = integrate.romb(dprof_dlogM * (nth_mat * np.array(Pth)), \
            dx = dlogM, axis = 0) / ng   

            dnP_gm_db = integrate.romb(dprof_dlogM * (dnP_hm_db_emu), dx = dlogM, axis = 0) 
            
            if surface:
                surface_pgm[ia, :] = ((prof[0] * nth_mat[0] * np.array(Pth)[0]) - (prof[-1] * nth_mat[-1] * np.array(Pth))[-1]) / ng
                Pgm += surface_pgm[ia, :]
                
                surface_resp[ia, :] = (prof[0] * dnP_hm_db_emu[0]) - (prof[-1] * dnP_hm_db_emu[-1])
                dnP_gm_db += surface_resp[ia, :]

            Pgm_growth = dnP_gm_db / ng - bgL * Pgm  

            Pgm_d = -1. / 3. *  np.gradient(np.log(Pgm)) / np.gradient(np.log(k_use)) * Pgm 
            
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
            
    pk2d = Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=pk12,
                extrap_order_lok=extrap_order_lok,
                extrap_order_hik=extrap_order_hik,
                cosmo=cosmo, is_logp=False)
    
    return dpk12, pk2d

def darkemu_Pgg_resp_zresp(cosmo, prof_hod, deltaz=0.1,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     log10Mh_pivot=12.5,
                     normprof_hod=False,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False, surface=False):
    
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
                logM1 = np.linspace(logM[m], logM[-1], 2**5+1)
                dlogM1 = logM[1] - logM[0]
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
    
    return dpk12, pk2d


def darkemu_Pgg_resp_Asresp(cosmo, prof_hod, deltalnAs=0.03,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     log10Mh_pivot=12.5,
                     normprof_hod=False,
                     lk_arr=None, a_arr=None,
                     extrap_order_lok=1, extrap_order_hik=1,
                     use_log=False, surface=False):
    
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
                logM1 = np.linspace(logM[m], logM[-1], 2**5+1)
                dlogM1 = logM[1] - logM[0]
                
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
    
    return dpk12, pk2d

def darkemu_Pgg(cosmo, prof_hod,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     log10Mh_pivot=12.5,
                     normprof_hod=False, k_max=2.0,
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
    
    return pk2d

def darkemu_Pgg_massbin(cosmo, prof_hod,
                     log10Mh_min=12.0,log10Mh_max=15.9,
                     log10Mh_pivot=12.5,
                     normprof_hod=False, k_max=2.0,
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
    
    return pk2d


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


def b2H17(b1):
    """ Implements fitting formula for secondary halo bias, b_2, described in         arXiv:1607.01024.
    """
    b2 = 0.77 - (2.43 * b1) + ( b1 * b1) 
    return b2


def b2L16(b1):
     """ Implements fitting formula for secondary halo bias, b_2, described in         arXiv:1511.01096.
    """
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

