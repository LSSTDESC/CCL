import numpy as np

from . import cosmology
from . import CCLWarning, CCLError, warnings

from dark_emulator import darkemu
from scipy import integrate
from . import halos

# use the perturbation theory below khmin
khmin = 1e-2  # [h/Mpc]


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
    use_log=False,
    khmin=khmin,
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
            interpolation.
        lk_arr (array): an array holding values of the natural
            logarithm of the wavenumber (in units of Mpc^-1) at
            which the trispectrum should be calculated for
            interpolation.
        use_log (bool): if `True`, the trispectrum will be
            interpolated in log-space (unless negative or
            zero values are found).

    Returns:
        Response of the matter power spectrum.
    """

    # Make sure input makes sense
    if (a_arr is None) or (lk_arr is None):
        raise ValueError("you must provide arrays")

    k_use = np.exp(lk_arr)

    # set h-modified cosmology to take finite differencing
    cosmo_hp, cosmo_hm = _set_hmodified_cosmology(
        cosmo, deltah, extra_parameters
    )

    # Growth factor
    Dp = cosmo_hp.growth_factor_unnorm(a_arr)
    Dm = cosmo_hm.growth_factor_unnorm(a_arr)

    # Linear power spectrum
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

    # use the perturbation theory below kmin
    kmin = khmin * cosmo["h"]
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
        ) / (2 * (np.log(Dp[ia]) - np.log(Dm[ia])))

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
    use_log=False,
    khmin=khmin,
):
    """Implements the response of galaxy-matter power spectrum to
    the long wavelength modes, described in arXiv:2310.13330.
    """
    # Make sure input makes sense
    if (a_arr is None) or (lk_arr is None):
        raise ValueError("you must provide arrays")

    k_use = np.exp(lk_arr)

    # Check inputs
    if not isinstance(prof_hod, halos.profiles.HaloProfile):
        raise TypeError("prof_hod must be of type `HaloProfile`")

    # dark emulator is valid for 0 =< z <= 1.48
    if np.any((1.0 / a_arr - 1) > 1.48):
        warnings.warn(
            "dark emulator is valid for z<=1.48", category=CCLWarning
        )

    # dark emulator support range is 10^12 <= M200m <= 10^16 Msun/h
    if log10Mh_min < 12.0 or log10Mh_max > 16.0:
        warnings.warn(
            "Input mass range is not supported."
            "The supported range is from 10^12 to 10^16 Msun/h.",
            category=CCLWarning,
        )

    h = cosmo["h"]
    k_emu = k_use / h  # [h/Mpc]
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power("delta_matter:delta_matter")

    # initialize dark emulator class
    emu = darkemu.de_interface.base_class()

    # set h-modified cosmology to take finite differencing
    hp = h + deltah
    hm = h - deltah
    cosmo_hp, cosmo_hm = _set_hmodified_cosmology(cosmo, deltah)

    # Growth factor
    Dp = cosmo_hp.growth_factor_unnorm(a_arr)
    Dm = cosmo_hm.growth_factor_unnorm(a_arr)

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    nM = 2**5 + 1
    log10M_min = np.log10(10**log10Mh_min / h)
    log10M_max = np.log10(10**log10Mh_max / h)

    b1_th_tink = np.zeros(nM)
    Pth = np.zeros((nM, nk))
    Pnth_hp = np.zeros((nM, nk))
    Pnth_hm = np.zeros((nM, nk))
    Pbin = np.zeros((nM, nk))
    nths = np.zeros(nM)

    mass_def = halos.MassDef200m
    hmf = halos.MassFuncNishimichi19(mass_def=mass_def, extrapolate=True)
    hbf = halos.HaloBiasTinker10(mass_def=mass_def)
    hmc = halos.HMCalculator(
        mass_function=hmf,
        halo_bias=hbf,
        mass_def=mass_def,
        log10M_min=log10M_min,
        log10M_max=log10M_max,
        nM=nM,
    )

    logM = hmc._lmass
    M = hmc._mass
    Mh = M * h
    dlogM = logM[1] - logM[0]

    for ia, aa in enumerate(a_arr):
        z = 1.0 / aa - 1
        hmc._get_ingredients(cosmo, aa, get_bf=True)

        _darkemu_set_cosmology(emu, cosmo)
        for m in range(nM):
            hmc_m = halos.HMCalculator(
                mass_function=hmf,
                halo_bias=hbf,
                mass_def=mass_def,
                log10M_min=logM[m],
                log10M_max=17.0,
            )
            hmc_m._get_ingredients(cosmo, aa, get_bf=True)

            Pth[m] = emu.get_phm_massthreshold(k_emu, Mh[m], z) * (1 / h) ** 3
            Pbin[m] = emu.get_phm_mass(k_emu, Mh[m], z) * (1 / h) ** 3
            nths[m] = emu.mass_to_dens(Mh[m], z) * h**3

            array_2 = np.ones(len(hmc_m._mass))
            array_2[..., 0] = 0
            b1_th_tink[m] = hmc_m._integrate_over_mbf(
                array_2
            ) / hmc_m._integrate_over_mf(array_2)

        _darkemu_set_cosmology(emu, cosmo_hp)
        for m in range(nM):
            Pnth_hp[m] = (
                emu.get_phm(
                    k_emu * (h / hp), np.log10(nths[m] * (1 / hp) ** 3), z
                )
                * (1 / hp) ** 3
            )

        _darkemu_set_cosmology(emu, cosmo_hm)
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
        ng = hmc._integrate_over_mf(Ng)

        # Eq. 17
        bgE = hmc._integrate_over_mbf(Ng) / ng

        # Eq. 19
        bgE2 = hmc._integrate_over_mf(Ng * _b2H17(hmc._bf)) / ng

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

        # use the perturbation theory below khmin
        dPgm_db[k_emu < khmin] = dPgm_db_lin[k_emu < khmin]
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
    use_log=False,
    khmin=khmin,
):
    """Implements the response of galaxy-auto power spectrum to
    the long wavelength modes, described in arXiv:2310.13330.
    """
    # Make sure input makes sense
    if (a_arr is None) or (lk_arr is None):
        raise ValueError("you must provide arrays")

    k_use = np.exp(lk_arr)

    # Check inputs
    if not isinstance(prof_hod, halos.profiles.HaloProfile):
        raise TypeError("prof_hod must be of type `HaloProfile`")

    # dark emulator is valid for 0 =< z <= 1.48
    if np.any((1.0 / a_arr - 1) > 1.48):
        warnings.warn(
            "dark emulator is valid for z<=1.48",
            category=CCLWarning,
        )

    # dark emulator support range is 10^12 <= M200m <= 10^16 Msun/h
    if log10Mh_min < 12.0 or log10Mh_max > 16.0:
        warnings.warn(
            "Input mass range is not supported."
            "The supported range is from 10^12 to 10^16 Msun/h.",
            category=CCLWarning,
        )

    h = cosmo["h"]
    k_emu = k_use / h  # [h/Mpc]
    cosmo.compute_linear_power()
    pk2dlin = cosmo.get_linear_power("delta_matter:delta_matter")

    # initialize dark emulator class
    emu = darkemu.de_interface.base_class()

    na = len(a_arr)
    nk = len(k_use)
    dpk12 = np.zeros([na, nk])
    nM = 2**5 + 1
    log10M_min = np.log10(10**log10Mh_min / h)
    log10M_max = np.log10(10**log10Mh_max / h)

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

    hmc = halos.HMCalculator(
        mass_function=hmf,
        halo_bias=hbf,
        mass_def=mass_def,
        log10M_min=log10M_min,
        log10M_max=log10M_max,
        nM=nM,
    )

    logM = hmc._lmass
    M = hmc._mass
    Mh = M * h
    dlogM = logM[1] - logM[0]

    for ia, aa in enumerate(a_arr):
        z = 1.0 / aa - 1
        hmc._get_ingredients(cosmo, aa, get_bf=True)

        for m in range(nM):
            hmc_m = halos.HMCalculator(
                mass_function=hmf,
                halo_bias=hbf,
                mass_def=mass_def,
                log10M_min=logM[m],
                log10M_max=17.0,
            )
            hmc_m._get_ingredients(cosmo, aa, get_bf=True)

            nths[m] = emu.mass_to_dens(Mh[m], z) * h**3

            array_2 = np.ones(len(hmc_m._mass))
            array_2[..., 0] = 0
            b1_th_tink[m] = hmc_m._integrate_over_mbf(
                array_2
            ) / hmc_m._integrate_over_mf(array_2)

        # set cosmology for dark emulator
        _darkemu_set_cosmology(emu, cosmo)
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
                    _get_phh_massthreshold_mass(
                        emu, k_emu, nths[m] / (h**3), Mh[n], z
                    )
                    * (1 / h) ** 3
                )

        _darkemu_set_cosmology_forAsresp(emu, cosmo, deltalnAs)
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

        _darkemu_set_cosmology_forAsresp(emu, cosmo, -deltalnAs)
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
        ng = hmc._integrate_over_mf(Ng)
        b1 = hbf(cosmo, M, aa)

        # Eq. 17
        bgE = hmc._integrate_over_mbf(Ng) / ng

        # Eq. 19
        bgE2 = hmc._integrate_over_mf(Ng * _b2H17(b1)) / ng

        bgL = bgE - 1
        b1L_mat = np.tile(b1 - 1, (len(k_emu), 1)).transpose()
        b1L_th_mat = np.tile(b1_th_tink - 1, (len(k_emu), 1)).transpose()

        # P_gg(k)
        _Pgg_1h = hmc._integrate_over_mf(prof_1h.T) / (ng**2)

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
        resp_1h = hmc._integrate_over_mf(b1L_mat.T * prof_1h.T) / (ng**2)

        #  Here we assume the galaxies' profile around host halo
        #  is fixed at physical corrdinate.
        G_prof = hmc._integrate_over_mf(dprof_1h_dlogk.T) / (ng**2) / 3.0

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

        # use the perturbation theory below khmin
        dPgg_db[k_emu < khmin] = dPgg_db_lin[k_emu < khmin]
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
def _get_phh_massthreshold_mass(emu, k_emu, dens1, Mbin, redshift):
    """Compute the halo-halo power spectrum between
    mass bin halo sample and mass threshold halo sample
    specified by the corresponding cumulative number density.
    """
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


def _b2H17(b1):
    """Implements fitting formula for secondary halo bias, b_2, described in
    arXiv:1607.01024.
    """
    b2 = 0.77 - (2.43 * b1) + (b1 * b1)
    return b2


def _darkemu_set_cosmology(emu, cosmo):
    """Input cosmology and initiallize the base class of DarkEmulator."""
    h = cosmo["h"]
    n_s = cosmo["n_s"]
    A_s = cosmo["A_s"]
    if np.isnan(A_s):
        raise ValueError("A_s must be provided to use the Dark Emulator")

    omega_c = cosmo["Omega_c"] * h**2
    omega_b = cosmo["Omega_b"] * h**2
    omega_nu = 0.00064  # we fix this value (Nishimichi et al. 2019)
    Omega_L = 1 - ((omega_c + omega_b + omega_nu) / h**2)

    # Parameters cparam (numpy array) : Cosmological parameters
    # (omega_b,omega_c,Omega_de,ln(10^10As),ns,w)
    cparam = np.array(
        [omega_b, omega_c, Omega_L, np.log(10**10 * A_s), n_s, -1.0]
    )
    if darkemu.cosmo_util.test_cosm_range(cparam):
        raise ValueError(
            ("cosmological parameter out of supported range of DarkEmulator")
        )

    emu.set_cosmology(cparam)


def _darkemu_set_cosmology_forAsresp(emu, cosmo, deltalnAs):
    """Input cosmology and initiallize the base class of DarkEmulator
    for cosmology with modified A_s.
    """
    h = cosmo["h"]
    n_s = cosmo["n_s"]
    A_s = cosmo["A_s"]
    if np.isnan(A_s):
        raise ValueError("A_s must be provided to use the Dark Emulator")

    omega_c = cosmo["Omega_c"] * h**2
    omega_b = cosmo["Omega_b"] * h**2
    omega_nu = 0.00064  # we fix this value (Nishimichi et al. 2019)
    Omega_L = 1 - ((omega_c + omega_b + omega_nu) / h**2)

    # Parameters cparam (numpy array) : Cosmological parameters
    # (omega_b,omega_c,Omega_de,ln(10^10As),ns,w)
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
    if darkemu.cosmo_util.test_cosm_range(cparam):
        raise ValueError(
            ("cosmological parameter out of supported range of DarkEmulator")
        )

    emu.set_cosmology(cparam)

    return emu


def _set_hmodified_cosmology(cosmo, deltah, extra_parameters=None):
    """Create the Cosmology objects with modified Hubble parameter h."""
    Omega_c = cosmo["Omega_c"]
    Omega_b = cosmo["Omega_b"]
    h = cosmo["h"]

    cosmo_hmodified = []
    for i in [+1, -1]:
        hp = h + i * deltah

        # \Omega_c h^2, \Omega_b h^2 is fixed
        Omega_c_p = np.power((h / hp), 2) * Omega_c
        Omega_b_p = np.power((h / hp), 2) * Omega_b

        cosmo_hp_dict = cosmo.to_dict()
        cosmo_hp_dict["h"] = hp
        cosmo_hp_dict["Omega_c"] = Omega_c_p
        cosmo_hp_dict["Omega_b"] = Omega_b_p
        cosmo_hp_dict["extra_parameters"] = extra_parameters
        cosmo_hp = cosmology.Cosmology(**cosmo_hp_dict)
        cosmo_hmodified.append(cosmo_hp)

    return cosmo_hmodified[0], cosmo_hmodified[1]
