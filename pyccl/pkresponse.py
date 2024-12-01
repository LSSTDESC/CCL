from . import ccllib as lib

from .pyutils import check
import numpy as np

from . import cosmology
import warnings
from .errors import CCLWarning

from dark_emulator import darkemu
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from . import halos


def Pmm_resp(
    cosmo,
    deltah=0.02,
    extra_parameters={
        "camb": {
            "halofit_version": "original",
        }
    },
    lk_arr=None,
    a_arr=None,
    extrap_order_lok=1,
    extrap_order_hik=1,
    use_log=False,
):
    """Implements the response of matter power spectrum to the long wavelength
    modes developed in Terasawa et al. 2023 (arXiv:2310.13330) as:

    .. math::
        \\frac{\\partial P_{mm}(k)}{\\partial\\delta_b} =
        \\left(1 + \\frac{26}{21}T_{h}^{mm}(k) - \\frac{1}{3}
        \\frac{d\\log P_{mm}(k)}{d\\log k}\\right)P_{mm}(k),

    where the :math:`T_{h}^{mm}(k)` is the normalized growth response to
    the Hubble parameter defined as
    :math:`T_{h}^{mm}(k)
    = \\frac{d\\log P_{mm}(k)}{dh}/(2\\frac{d\\log D}{dh})`.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): a Cosmology object.
        deltah (float): the variation of h to compute T_{h}(k) by
            the two-sided numerical derivative method.
        extra_parameters (:obj:`dict`): Dictionary holding extra
            parameters. Currently supports extra parameters for CAMB.
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
        Response of the matter power spectrum.
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

    # set h-modified cosmology to take finite differencing
    cosmo_hp, cosmo_hm = set_hmodified_cosmology(cosmo, deltah, extra_parameters)

    # Growth factor
    Dp = cosmo_hp.growth_factor_unnorm(a_arr)
    Dm = cosmo_hm.growth_factor_unnorm(a_arr)

    # Power spectrum
    cosmo.compute_linear_power()
    cosmo_hp.compute_linear_power()
    cosmo_hm.compute_linear_power()

    pk2dlin = cosmo.get_linear_power("delta_matter:delta_matter")

    pk2d = cosmo.get_nonlin_power("delta_matter:delta_matter")
    pk2d_hp = cosmo_hp.get_nonlin_power("delta_matter:delta_matter")
    pk2d_hm = cosmo_hm.get_nonlin_power("delta_matter:delta_matter")

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    dpk = np.zeros(nk)
    T_h = np.zeros(nk)

    kmin = 1e-2
    # use the perturbation theory below kmin
    T_h[k_use <= kmin] = 1

    for ia, aa in enumerate(a_arr):
        pk = pk2d(k_use, aa, cosmo)
        pk_hp = pk2d_hp(k_use, aa, cosmo_hp)
        pk_hm = pk2d_hm(k_use, aa, cosmo_hm)

        dpknl = pk2d(k_use, aa, cosmo, derivative=True)
        dpklin = pk2dlin(k_use, aa, cosmo, derivative=True)

        # Eq. 11 ((hp-hm) term is cancelled out)
        T_h[k_use > kmin] = (
            np.log(pk_hp[k_use > kmin]) - np.log(pk_hm[k_use > kmin])
        ) / (
            2 * (np.log(Dp[ia]) - np.log(Dm[ia]))
        ) 

        dpk[k_use <= kmin] = dpklin[k_use <= kmin]
        dpk[k_use > kmin] = dpknl[k_use > kmin]

        # Eq. 23
        dpk12[ia, :] = pk * (1.0 + (26.0 / 21.0) * T_h - dpk / 3.0)

    if use_log:
        if np.any(dpk12 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning,
            )
            use_log = False
        else:
            dpk12 = np.log(dpk12)

    return dpk12


def darkemu_Pgm_resp(
    cosmo,
    prof_hod,
    deltah=0.02,
    log10Mh_min=12.0,
    log10Mh_max=15.9,
    lk_arr=None,
    a_arr=None,
    extrap_order_lok=1,
    extrap_order_hik=1,
    use_log=False,
):
    """Implements the response of galaxy-matter power spectrum to
    the long wavelength modes, described in arXiv:2310.13330.
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
    k_emu = k_use / h  # [h/Mpc]
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power("delta_matter:delta_matter")

    # initialize dark emulator class
    emu = darkemu.de_interface.base_class()

    # set h-modified cosmology to take finite differencing
    hp = h + deltah
    hm = h - deltah
    cosmo_hp, cosmo_hm = set_hmodified_cosmology(cosmo, deltah)

    # Growth factor
    Dp = cosmo_hp.growth_factor_unnorm(a_arr)
    Dm = cosmo_hm.growth_factor_unnorm(a_arr)

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    logMfor_hmf = np.linspace(8, 17, 200)
    logMh = np.linspace(log10Mh_min, log10Mh_max, 2**5 + 1)  # M_sol/h
    logM = np.log10(10**logMh / h)
    Mh = 10**logMh
    M = 10**logM
    nM = len(M)
    dlogM = logM[1] - logM[0]
    b1_th_tink = np.zeros(nM)
    Pth = np.zeros((nM, nk))
    Pnth_hp = np.zeros((nM, nk))
    Pnth_hm = np.zeros((nM, nk))
    Pbin = np.zeros((nM, nk))

    nths = np.zeros(nM)

    mass_def = halos.MassDef200m
    hmf = halos.MassFuncNishimichi19(mass_def=mass_def, extrapolate=True)
    hbf = halos.HaloBiasTinker10(mass_def=mass_def)

    # dark emulator is valid for 0 =< z <= 1.48
    if np.any(1.0 / a_arr - 1) > 1.48:
        print("dark emulator is valid for z<=1.48")

    for ia, aa in enumerate(a_arr):
        z = 1.0 / aa - 1

        # mass function
        dndlog10m_emu = ius(
            logMfor_hmf, hmf(cosmo, 10**logMfor_hmf, aa)
        )  # Mpc^-3

        darkemu_set_cosmology(emu, cosmo)
        for m in range(nM):
            Pth[m] = emu.get_phm_massthreshold(k_emu, Mh[m], z) * (1 / h) ** 3
            Pbin[m] = emu.get_phm_mass(k_emu, Mh[m], z) * (1 / h) ** 3
            nths[m] = emu.mass_to_dens(Mh[m], z) * h**3

            logM1 = np.linspace(logM[m], logM[-1], 2**5 + 1)
            dlogM1 = logM[1] - logM[0]

            b1_th_tink[m] = integrate.romb(
                dndlog10m_emu(logM1) * hbf(cosmo, (10**logM1), aa), dx=dlogM1
            ) / integrate.romb(dndlog10m_emu(logM1), dx=dlogM1)

        darkemu_set_cosmology(emu, cosmo_hp)
        for m in range(nM):
            Pnth_hp[m] = (
                emu.get_phm(
                    k_emu * (h / hp), np.log10(nths[m] * (1 / hp) ** 3), z
                )
                * (1 / hp) ** 3
            )

        darkemu_set_cosmology(emu, cosmo_hm)
        for m in range(nM):
            Pnth_hm[m] = (
                emu.get_phm(
                    k_emu * (h / hm), np.log10(nths[m] * (1 / hm) ** 3), z
                )
                * (1 / hm) ** 3
            )

        Nc = prof_hod._Nc(M, aa)
        Ns = prof_hod._Ns(M, aa)
        fc = prof_hod._fc(aa)
        Ng = Nc * (fc + Ns)
        logMps = logM + dlogM
        logMms = logM - dlogM

        prof_Mp = prof_hod.fourier(cosmo, k_use, (10**logMps), aa)
        prof_Mm = prof_hod.fourier(cosmo, k_use, (10**logMms), aa)

        dprof_dlogM = (prof_Mp - prof_Mm) / (2 * dlogM)

        nth_mat = np.tile(nths, (len(k_use), 1)).transpose()
        
        # Eq. 18
        ng = integrate.romb(dndlog10m_emu(logM) * Ng, dx=dlogM, axis=0)
        
        # Eq. 17
        bgE = (
            integrate.romb(
                dndlog10m_emu(logM) * Ng * (hbf(cosmo, M, aa)),
                dx=dlogM,
                axis=0,
            )
            / ng
        )

        # Eq. 19
        bgE2 = (
            integrate.romb(
                dndlog10m_emu(logM) * Ng * b2H17(hbf(cosmo, M, aa)),
                dx=dlogM,
                axis=0,
            )
            / ng
        )
        bgL = bgE - 1

        b1L_th_mat = np.tile(b1_th_tink - 1, (len(k_emu), 1)).transpose()

        dPhm_db_nfix = (
            (26.0 / 21.0)
            * np.log(np.array(Pnth_hp) / np.array(Pnth_hm))
            * np.array(Pth)
            / (2 * (np.log(Dp[ia]) - np.log(Dm[ia])))
        )  # Mpc^3

        dnP_hm_db_emu = nth_mat * (dPhm_db_nfix + b1L_th_mat * np.array(Pbin))

        # Eq. A2
        nP = nth_mat * np.array(Pth)

        # Eq. A7
        Pgm = integrate.romb(dprof_dlogM * nP, dx=dlogM, axis=0) / ng

        # The first term of Eq. A8
        dnP_gm_db = integrate.romb(
            dprof_dlogM * (dnP_hm_db_emu), dx=dlogM, axis=0
        )

        # Here we assume the galaxies' profile around host halo
        # is fixed at physical corrdinate.
        dprof_dlogM_dlogk = np.zeros((nM, nk))
        for i in range(nM):
            dprof_dlogM_dlogk[i] = np.gradient((dprof_dlogM[i])) / np.gradient(
                np.log(k_use)
            )

        # The second term of Eq. A8
        G_prof = (
            +1.0
            / 3.0
            * integrate.romb(dprof_dlogM_dlogk * nP, dx=dlogM, axis=0)
        )

        # Eq. 25
        Pgm_growth = (dnP_gm_db + G_prof) / ng - bgL * Pgm

        Pgm_d = (
            -1.0
            / 3.0
            * np.gradient(np.log(Pgm))
            / np.gradient(np.log(k_use))
            * Pgm
        )

        # Eq. 22
        dPgm_db_emu = Pgm_growth + Pgm_d

        dpklin = pk2dlin(k_use, aa, cosmo, derivative=True)

        # Eq. 16
        dPgm_db_lin = (
            (47 / 21 + bgE2 / bgE - bgE - 1 / 3 * dpklin)
            * bgE
            * pk2dlin(k_use, aa, cosmo)
        )

        # stitching
        k_switch = 0.08  # [h/Mpc]

        # Eq. 27
        dPgm_db = dPgm_db_lin * np.exp(-k_emu / k_switch) + dPgm_db_emu * (
            1 - np.exp(-k_emu / k_switch)
        )

        # use the perturbation theory below kmin
        kmin = 1e-2  # [h/Mpc]

        dPgm_db[k_emu < kmin] = dPgm_db_lin[k_emu < kmin]
        dpk12[ia, :] = dPgm_db

    if use_log:
        if np.any(dpk12 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning,
            )
            use_log = False
        else:
            dpk12 = np.log(dpk12)

    return dpk12


def darkemu_Pgg_resp(
    cosmo,
    prof_hod,
    deltalnAs=0.03,
    log10Mh_min=12.0,
    log10Mh_max=15.9,
    lk_arr=None,
    a_arr=None,
    extrap_order_lok=1,
    extrap_order_hik=1,
    use_log=False,
):
    """Implements the response of galaxy-auto power spectrum to
    the long wavelength modes, described in arXiv:2310.13330.
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
    k_emu = k_use / h  # [h/Mpc]
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power("delta_matter:delta_matter")

    # initialize dark emulator class
    emu = darkemu.de_interface.base_class()

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    logMfor_hmf = np.linspace(8, 17, 200)
    logMh = np.linspace(log10Mh_min, log10Mh_max, 2**5 + 1)  # M_sol/h
    logM = np.log10(10**logMh / h)
    Mh = 10**logMh
    M = 10**logM
    nM = len(M)
    dlogM = logM[1] - logM[0]
    b1_th_tink = np.zeros(nM)
    Pth = np.zeros((nM, nM, nk))
    Pth_Ap = np.zeros((nM, nM, nk))
    Pth_Am = np.zeros((nM, nM, nk))
    Pth_bin = np.zeros((nM, nM, nk))
    nths = np.zeros(nM)

    mass_def = halos.MassDef200m
    hmf = halos.MassFuncNishimichi19(mass_def=mass_def, extrapolate=True)
    hbf = halos.HaloBiasTinker10(mass_def=mass_def)
    prof_2pt = halos.profiles_2pt.Profile2ptHOD()

    # dark emulator is valid for 0 =< z <= 1.48
    if np.any(1.0 / a_arr - 1) > 1.48:
        print("dark emulator is valid for z<=1.48")

    for ia, aa in enumerate(a_arr):
        z = 1.0 / aa - 1

        # mass function
        dndlog10m_emu = ius(
            logMfor_hmf, hmf(cosmo, 10**logMfor_hmf, aa)
        )  # Mpc^-3

        for m in range(nM):
            nths[m] = mass_to_dens(dndlog10m_emu, cosmo, M[m])
            logM1 = np.linspace(logM[m], logM[-1], 2**5 + 1)
            dlogM1 = logM[1] - logM[0]

            b1_th_tink[m] = integrate.romb(
                dndlog10m_emu(logM1) * hbf(cosmo, (10**logM1), aa), dx=dlogM1
            ) / integrate.romb(dndlog10m_emu(logM1), dx=dlogM1)

        # set cosmology for dark emulator
        darkemu_set_cosmology(emu, cosmo)
        for m in range(nM):
            for n in range(nM):
                Pth[m, n] = (
                    emu.get_phh(
                        k_emu,
                        np.log10(nths[m] / (h**3)),
                        np.log10(nths[n] / (h**3)),
                        z,
                    )
                    * (1 / h) ** 3
                )

                Pth_bin[m, n] = (
                    get_phh_massthreshold_mass(
                        emu, k_emu, nths[m] / (h**3), Mh[n], z
                    )
                    * (1 / h) ** 3
                )

        darkemu_set_cosmology_forAsresp(emu, cosmo, deltalnAs)
        for m in range(nM):
            for n in range(nM):
                Pth_Ap[m, n] = (
                    emu.get_phh(
                        k_emu,
                        np.log10(nths[m] / (h**3)),
                        np.log10(nths[n] / (h**3)),
                        z,
                    )
                    * (1 / h) ** 3
                )

        darkemu_set_cosmology_forAsresp(emu, cosmo, -deltalnAs)
        for m in range(nM):
            for n in range(nM):
                Pth_Am[m, n] = (
                    emu.get_phh(
                        k_emu,
                        np.log10(nths[m] / (h**3)),
                        np.log10(nths[n] / (h**3)),
                        z,
                    )
                    * (1 / h) ** 3
                )

        Nc = prof_hod._Nc(M, aa)
        Ns = prof_hod._Ns(M, aa)
        fc = prof_hod._fc(aa)
        Ng = Nc * (fc + Ns)
        logMps = logM + dlogM
        logMms = logM - dlogM

        prof_Mp = prof_hod.fourier(cosmo, k_use, (10**logMps), aa)
        prof_Mm = prof_hod.fourier(cosmo, k_use, (10**logMms), aa)
        prof_1h = prof_2pt.fourier_2pt(cosmo, k_use, M, aa, prof_hod)

        dprof_dlogM = (prof_Mp - prof_Mm) / (2 * dlogM)
        dprof_dlogM_dlogk = np.zeros((nM, nk))
        dprof_1h_dlogk = np.zeros((nM, nk))
        for i in range(nM):
            dprof_dlogM_dlogk[i] = np.gradient((dprof_dlogM[i])) / np.gradient(
                np.log(k_use)
            )
            dprof_1h_dlogk[i] = np.gradient((prof_1h[i])) / np.gradient(
                np.log(k_use)
            )

        nth_mat = np.tile(nths, (len(k_use), 1)).transpose()

        # Eq. 18
        ng = integrate.romb(dndlog10m_emu(logM) * Ng, dx=dlogM, axis=0)
        b1 = hbf(cosmo, M, aa)

        # Eq. 17
        bgE = (
            integrate.romb(dndlog10m_emu(logM) * Ng * b1, dx=dlogM, axis=0)
            / ng
        )

        #Eq. 19
        bgE2 = (
            integrate.romb(
                dndlog10m_emu(logM) * Ng * b2H17(b1), dx=dlogM, axis=0
            )
            / ng
        )
        bgL = bgE - 1

        dndlog10m_func_mat = np.tile(
            dndlog10m_emu(logM), (len(k_emu), 1)
        ).transpose()  # M_sol,Mpc^-3

        b1L_mat = np.tile(b1 - 1, (len(k_emu), 1)).transpose()
        b1L_th_mat = np.tile(b1_th_tink - 1, (len(k_emu), 1)).transpose()

        # P_gg(k)
        _Pgg_1h = integrate.romb(
            dndlog10m_func_mat * prof_1h, dx=dlogM, axis=0
        ) / (ng**2)

        Pgg_2h_int = list()
        for m in range(nM):
            Pgg_2h_int.append(
                integrate.romb(
                    Pth[m] * nth_mat * dprof_dlogM, axis=0, dx=dlogM
                )
            )
        Pgg_2h_int = np.array(Pgg_2h_int)

        # Eq. A12
        _Pgg_2h = integrate.romb(
            Pgg_2h_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM
        ) / (ng**2)

        Pgg = _Pgg_2h + _Pgg_1h

        # 2-halo response
        dPhh_db_nfix = (26.0 / 21.0) * (Pth_Ap - Pth_Am) / (2 * deltalnAs)

        resp_2h_int = list()
        for m in range(nM):
            dP_hh_db_tot = dPhh_db_nfix[m] + 2 * b1L_th_mat * Pth_bin[m]
            resp_2h_int.append(
                integrate.romb(
                    dP_hh_db_tot * nth_mat * dprof_dlogM, axis=0, dx=dlogM
                )
            )
        resp_2h_int = np.array(resp_2h_int)
        resp_2h = integrate.romb(
            resp_2h_int * nth_mat * dprof_dlogM, axis=0, dx=dlogM
        ) / (ng**2)

        #  Here we assume the galaxies' profile around host halo
        #  is fixed at physical corrdinate.
        G_prof = (
            +2.0
            / 3.0
            * integrate.romb(
                resp_2h_int * nth_mat * dprof_dlogM_dlogk, axis=0, dx=dlogM
            )
            / (ng**2)
        )

        # The first term of Eq. A14
        resp_2h = resp_2h + G_prof

        # 1-halo response
        resp_1h = integrate.romb(
            dndlog10m_func_mat * b1L_mat * prof_1h, dx=dlogM, axis=0
        ) / (ng**2)

        #  Here we assume the galaxies' profile around host halo
        #  is fixed at physical corrdinate.
        G_prof = (
            +1.0
            / 3.0
            * integrate.romb(
                dndlog10m_func_mat * dprof_1h_dlogk, dx=dlogM, axis=0
            )
            / (ng**2)
        )

        # The first term of Eq. A15
        resp_1h = resp_1h + G_prof

        Pgg_growth = (resp_1h + resp_2h) - 2 * bgL * Pgg

        Pgg_d = (
            -1.0
            / 3.0
            * np.gradient(np.log(Pgg))
            / np.gradient(np.log(k_use))
            * Pgg
        )

        # Eq. 22
        dPgg_db_emu = Pgg_growth + Pgg_d - Pgg

        dpklin = pk2dlin(k_use, aa, cosmo, derivative=True)

        Pgg_lin = bgE**2 * pk2dlin(k_use, aa, cosmo)

        # Eq. 16
        dPgg_db_lin = (
            47 / 21 + 2 * bgE2 / bgE - 2 * bgE - 1 / 3 * dpklin
        ) * Pgg_lin
        # stitching
        k_switch = 0.08  # [h/Mpc]

        # Eq. 27
        dPgg_db = dPgg_db_lin * np.exp(-k_emu / k_switch) + dPgg_db_emu * (
            1 - np.exp(-k_emu / k_switch)
        )

        Pgg = Pgg_lin * np.exp(-k_emu / k_switch) + Pgg * (
            1 - np.exp(-k_emu / k_switch)
        )

        # use the perturbation theory below kmin
        kmin = 1e-2  # [h/Mpc]

        dPgg_db[k_emu < kmin] = dPgg_db_lin[k_emu < kmin]
        dpk12[ia, :] = dPgg_db

    if use_log:
        if np.any(dpk12 <= 0):
            warnings.warn(
                "Some values were not positive. "
                "Will not interpolate in log-space.",
                category=CCLWarning,
            )
            use_log = False
        else:
            dpk12 = np.log(dpk12)

    return dpk12


# Utility functions ####################


def mass_to_dens(dndlog10m, cosmo, mass_thre):
    """Converts mass threshold to 
    """
    logM1 = np.linspace(
        np.log10(mass_thre), np.log10(10**16.0 / cosmo["h"]), 2**6 + 1
    )
    dlogM1 = logM1[1] - logM1[0]
    dens = integrate.romb(dndlog10m(logM1), dx=dlogM1)

    return dens


def dens_to_mass(dndlog10m_emu, cosmo, dens, nint=60):
    mlist = np.linspace(8, np.log10(10**15.8 / cosmo["h"]), nint)
    dlist = np.log(
        np.array(
            [
                mass_to_dens(dndlog10m_emu, cosmo, 10 ** mlist[i])
                for i in range(nint)
            ]
        )
    )
    d_to_m_interp = ius(-dlist, mlist)

    return 10 ** d_to_m_interp(-np.log(dens))


def get_phh_massthreshold_mass(emu, k_emu, dens1, Mbin, redshift):
    M2p = Mbin * 1.01
    M2m = Mbin * 0.99
    dens2p = emu.mass_to_dens(M2p, redshift)
    dens2m = emu.mass_to_dens(M2m, redshift)
    logdens1, logdens2p, logdens2m = (
        np.log10(dens1),
        np.log10(dens2p),
        np.log10(dens2m),
    )

    p_p = emu.get_phh(k_emu, logdens1, logdens2p, redshift)
    p_m = emu.get_phh(k_emu, logdens1, logdens2m, redshift)

    numer = dens2m * p_m - dens2p * p_p
    denom = dens2m - dens2p

    return numer / denom


def b2H17(b1):
    """Implements fitting formula for secondary halo bias, b_2, described in
    arXiv:1607.01024.
    """
    b2 = 0.77 - (2.43 * b1) + (b1 * b1)
    return b2


def b2L16(b1):
    """Implements fitting formula for secondary halo bias, b_2, described in
    arXiv:1511.01096.
    """
    b2 = 0.412 - (2.143 * b1) + (0.929 * b1 * b1) + (0.008 * b1 * b1 * b1)
    return b2


def darkemu_set_cosmology(emu, cosmo):
    """Input cosmology and initiallize the base class of DarkEmulator.
    """
    Omega_c = cosmo["Omega_c"]
    Omega_b = cosmo["Omega_b"]
    h = cosmo["h"]
    n_s = cosmo["n_s"]
    A_s = cosmo["A_s"]

    omega_c = Omega_c * h**2
    omega_b = Omega_b * h**2
    omega_nu = 0.00064
    Omega_L = 1 - ((omega_c + omega_b + omega_nu) / h**2)

    # Parameters cparam (numpy array) : Cosmological parameters
    # (𝜔𝑏, 𝜔𝑐, Ω𝑑𝑒, ln(10^10 𝐴𝑠), 𝑛𝑠, 𝑤)
    cparam = np.array(
        [omega_b, omega_c, Omega_L, np.log(10**10 * A_s), n_s, -1.0]
    )
    emu.set_cosmology(cparam)


def darkemu_set_cosmology_forAsresp(emu, cosmo, deltalnAs):
    """Input cosmology and initiallize the base class of DarkEmulator
    for cosmology with modified A_s.
    """
    Omega_c = cosmo["Omega_c"]
    Omega_b = cosmo["Omega_b"]
    h = cosmo["h"]
    n_s = cosmo["n_s"]
    A_s = cosmo["A_s"]

    omega_c = Omega_c * h**2
    omega_b = Omega_b * h**2
    omega_nu = 0.00064
    Omega_L = 1 - ((omega_c + omega_b + omega_nu) / h**2)

    # Parameters cparam (numpy array) : Cosmological parameters
    # (𝜔𝑏, 𝜔𝑐, Ω𝑑𝑒, ln(10^10 𝐴𝑠), 𝑛𝑠, 𝑤)
    cparam = np.array(
        [
            omega_b,
            omega_c,
            Omega_L,
            np.log(10**10 * A_s) + deltalnAs,
            n_s,
            -1.0,
        ]
    )
    emu.set_cosmology(cparam)

    return emu


def set_hmodified_cosmology(cosmo, deltah, extra_parameters=None):
    """Input cosmology and initiallize the base class of DarkEmulator
    for cosmology with modified Hubble parameter h.
    """
    Omega_c = cosmo["Omega_c"]
    Omega_b = cosmo["Omega_b"]
    h = cosmo["h"]
    n_s = cosmo["n_s"]
    A_s = cosmo["A_s"]

    # \Omega_c h^2, \Omega_b h^2 is fixed
    hp = h + deltah
    Omega_c_p = np.power((h / hp), 2) * Omega_c
    Omega_b_p = np.power((h / hp), 2) * Omega_b

    hm = h - deltah
    Omega_c_m = np.power((h / hm), 2) * Omega_c
    Omega_b_m = np.power((h / hm), 2) * Omega_b

    cosmo_hp = cosmology.Cosmology(
        Omega_c=Omega_c_p, Omega_b=Omega_b_p, h=hp, n_s=n_s, A_s=A_s,
        extra_parameters=extra_parameters
    )

    cosmo_hm = cosmology.Cosmology(
        Omega_c=Omega_c_m, Omega_b=Omega_b_m, h=hm, n_s=n_s, A_s=A_s,
        extra_parameters=extra_parameters
    )

    return cosmo_hp, cosmo_hm
