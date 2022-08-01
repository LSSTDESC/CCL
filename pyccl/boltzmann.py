import numpy as np

try:
    import isitgr  # noqa: F401
except ModuleNotFoundError:
    pass  # prevent nans from isitgr

from . import ccllib as lib
from .pyutils import check
from .pk2d import Pk2D
from .errors import CCLError
from .parameters import physical_constants


def get_camb_pk_lin(cosmo, nonlin=False):
    """Run CAMB and return the linear power spectrum.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological
            parameters. The cosmological parameters with
            which to run CAMB.
        nonlin (:obj:`bool`, optional): Whether to compute and return the
            non-linear power spectrum as well.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: Power spectrum object. The linear power \
            spectrum. If ``nonlin=True``, returns a tuple \
            ``(pk_lin, pk_nonlin)``.
    """
    import camb
    import camb.model

    # Get extra CAMB parameters that were specified
    extra_camb_params = {}
    try:
        extra_camb_params = cosmo["extra_parameters"]["camb"]
    except (KeyError, TypeError):
        pass

    # z sampling from CCL parameters
    na = lib.get_pk_spline_na(cosmo.cosmo)
    status = 0
    a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
    check(status, cosmo=cosmo)
    a_arr = np.sort(a_arr)
    zs = 1.0 / a_arr - 1
    zs = np.clip(zs, 0, np.inf)

    # deal with normalization
    if np.isfinite(cosmo["A_s"]):
        A_s_fid = cosmo["A_s"]
    elif np.isfinite(cosmo["sigma8"]):
        # in this case, CCL will internally normalize for us when we init
        # the linear power spectrum - so we just get close
        A_s_fid = 2.43e-9 * (cosmo["sigma8"] / 0.87659)**2
    else:
        raise CCLError(
            "Could not normalize the linear power spectrum! "
            "A_s = %f, sigma8 = %f" % (
                cosmo['A_s'], cosmo['sigma8']))

    # init camb params
    cp = camb.model.CAMBparams()

    # turn some stuff off
    cp.WantCls = False
    cp.DoLensing = False
    cp.Want_CMB = False
    cp.Want_CMB_lensing = False
    cp.Want_cl_2D_array = False
    cp.WantTransfer = True

    # basic background stuff
    h2 = cosmo['h']**2
    cp.H0 = cosmo['h'] * 100
    cp.ombh2 = cosmo['Omega_b'] * h2
    cp.omch2 = cosmo['Omega_c'] * h2
    cp.omk = cosmo['Omega_k']

    # "constants"
    cp.TCMB = cosmo['T_CMB']

    # neutrinos
    # We maually setup the CAMB neutrinos to match the adjustments CLASS
    # makes to their temperatures.
    cp.share_delta_neff = False
    cp.omnuh2 = cosmo['Omega_nu_mass'] * h2
    cp.num_nu_massless = cosmo['N_nu_rel']
    cp.num_nu_massive = int(cosmo['N_nu_mass'])
    cp.nu_mass_eigenstates = int(cosmo['N_nu_mass'])

    delta_neff = cosmo['Neff'] - 3.046  # used for BBN YHe comps

    # CAMB defines a neutrino degeneracy factor as T_i = g^(1/4)*T_nu
    # where T_nu is the standard neutrino temperature from first order
    # computations
    # CLASS defines the temperature of each neutrino species to be
    # T_i_eff = TNCDM * T_cmb where TNCDM is a fudge factor to get the
    # total mass in terms of eV to match second-order computations of the
    # relationship between m_nu and Omega_nu.
    # We are trying to get both codes to use the same neutrino temperature.
    # thus we set T_i_eff = T_i = g^(1/4) * T_nu and solve for the right
    # value of g for CAMB. We get g = (TNCDM / (11/4)^(-1/3))^4
    g = np.power(
        physical_constants.TNCDM / np.power(11.0/4.0, -1.0/3.0),
        4.0)

    if cosmo['N_nu_mass'] > 0:
        nu_mass_fracs = cosmo['m_nu'][:cosmo['N_nu_mass']]
        nu_mass_fracs = nu_mass_fracs / np.sum(nu_mass_fracs)

        cp.nu_mass_numbers = np.ones(cosmo['N_nu_mass'], dtype=np.int)
        cp.nu_mass_fractions = nu_mass_fracs
        cp.nu_mass_degeneracies = np.ones(int(cosmo['N_nu_mass'])) * g
    else:
        cp.nu_mass_numbers = []
        cp.nu_mass_fractions = []
        cp.nu_mass_degeneracies = []

    # get YHe from BBN
    cp.bbn_predictor = camb.bbn.get_predictor()
    cp.YHe = cp.bbn_predictor.Y_He(
        cp.ombh2 * (camb.constants.COBE_CMBTemp / cp.TCMB) ** 3,
        delta_neff)

    camb_de_models = ['DarkEnergyPPF', 'ppf', 'DarkEnergyFluid', 'fluid']
    camb_de_model = extra_camb_params.get('dark_energy_model', 'fluid')
    if camb_de_model not in camb_de_models:
        raise ValueError("The only dark energy models CCL supports with"
                         " camb are fluid and ppf.")
    cp.set_classes(
        dark_energy_model=camb_de_model
    )

    if camb_de_model not in camb_de_models[:2] and cosmo['wa'] and \
            (cosmo['w0'] < -1 - 1e-6 or
                1 + cosmo['w0'] + cosmo['wa'] < - 1e-6):
        raise ValueError("If you want to use w crossing -1,"
                         " then please set the dark_energy_model to ppf.")
    cp.DarkEnergy.set_params(
        w=cosmo['w0'],
        wa=cosmo['wa']
    )

    if nonlin:
        cp.NonLinearModel = camb.nonlinear.Halofit()
        halofit_version = extra_camb_params.get("halofit_version", "mead")
        options = {k: extra_camb_params[k] for k in
                   ["HMCode_A_baryon",
                    "HMCode_eta_baryon",
                    "HMCode_logT_AGN"] if k in extra_camb_params}
        cp.NonLinearModel.set_params(halofit_version=halofit_version,
                                     **options)

    cp.set_matter_power(
        redshifts=[_z for _z in zs],
        kmax=extra_camb_params.get("kmax", 10.0),
        nonlinear=nonlin)
    if not nonlin:
        assert cp.NonLinear == camb.model.NonLinear_none

    cp.set_for_lmax(extra_camb_params.get("lmax", 5000))
    cp.InitPower.set_params(
        As=A_s_fid,
        ns=cosmo['n_s'])

    # run CAMB and get results
    camb_res = camb.get_results(cp)
    k, z, pk = camb_res.get_linear_matter_power_spectrum(
        hubble_units=True, nonlinear=False)

    # convert to non-h inverse units
    k *= cosmo['h']
    pk /= (h2 * cosmo['h'])

    # now build interpolant
    nk = k.shape[0]
    lk_arr = np.log(k)
    a_arr = 1.0 / (1.0 + z)
    na = a_arr.shape[0]
    sinds = np.argsort(a_arr)
    a_arr = a_arr[sinds]
    ln_p_k_and_z = np.zeros((na, nk), dtype=np.float64)
    for i, sind in enumerate(sinds):
        ln_p_k_and_z[i, :] = np.log(pk[sind, :])

    pk_lin = Pk2D(
        pkfunc=None,
        a_arr=a_arr,
        lk_arr=lk_arr,
        pk_arr=ln_p_k_and_z,
        is_logp=True,
        extrap_order_lok=1,
        extrap_order_hik=2,
        cosmo=cosmo)

    if not nonlin:
        return pk_lin
    else:
        k, z, pk = camb_res.get_linear_matter_power_spectrum(
            hubble_units=True, nonlinear=True)

        # convert to non-h inverse units
        k *= cosmo['h']
        pk /= (h2 * cosmo['h'])

        # now build interpolant
        nk = k.shape[0]
        lk_arr = np.log(k)
        a_arr = 1.0 / (1.0 + z)
        na = a_arr.shape[0]
        sinds = np.argsort(a_arr)
        a_arr = a_arr[sinds]
        ln_p_k_and_z = np.zeros((na, nk), dtype=np.float64)
        for i, sind in enumerate(sinds):
            ln_p_k_and_z[i, :] = np.log(pk[sind, :])

        pk_nonlin = Pk2D(
            pkfunc=None,
            a_arr=a_arr,
            lk_arr=lk_arr,
            pk_arr=ln_p_k_and_z,
            is_logp=True,
            extrap_order_lok=1,
            extrap_order_hik=2,
            cosmo=cosmo)

        return pk_lin, pk_nonlin


def get_isitgr_pk_lin(cosmo):
    """Run ISiTGR-CAMB and return the linear power spectrum.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological
            parameters. The cosmological parameters with
            which to run ISiTGR-CAMB.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: Power spectrum \
            object. The linear power spectrum.
    """
    import isitgr  # noqa: F811
    import isitgr.model

    # Get extra CAMB parameters that were specified
    extra_camb_params = {}
    try:
        extra_camb_params = cosmo["extra_parameters"]["camb"]
    except (KeyError, TypeError):
        pass

    # z sampling from CCL parameters
    na = lib.get_pk_spline_na(cosmo.cosmo)
    status = 0
    a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
    check(status, cosmo=cosmo)
    a_arr = np.sort(a_arr)
    zs = 1.0 / a_arr - 1
    zs = np.clip(zs, 0, np.inf)

    # deal with normalization
    if np.isfinite(cosmo["A_s"]):
        A_s_fid = cosmo["A_s"]
    elif np.isfinite(cosmo["sigma8"]):
        # in this case, CCL will internally normalize for us when we init
        # the linear power spectrum - so we just get close
        A_s_fid = 2.43e-9 * (cosmo["sigma8"] / 0.87659)**2
    else:
        raise CCLError(
            "Could not normalize the linear power spectrum! "
            "A_s = %f, sigma8 = %f" % (
                cosmo['A_s'], cosmo['sigma8']))

    # init isitgr params
    cp = isitgr.model.CAMBparams()

    # turn some stuff off
    cp.WantCls = False
    cp.DoLensing = False
    cp.Want_CMB = False
    cp.Want_CMB_lensing = False
    cp.Want_cl_2D_array = False
    cp.WantTransfer = True

    # basic background stuff
    h2 = cosmo['h']**2
    cp.H0 = cosmo['h'] * 100
    cp.ombh2 = cosmo['Omega_b'] * h2
    cp.omch2 = cosmo['Omega_c'] * h2
    cp.omk = cosmo['Omega_k']
#   cp.GR = 1 means GR modified!
    cp.GR = 1
    cp.ISiTGR_muSigma = True
    cp.mu0 = cosmo['mu_0']
    cp.Sigma0 = cosmo['sigma_0']
    cp.c1 = cosmo['c1_mg']
    cp.c2 = cosmo['c2_mg']
    cp.Lambda = cosmo['lambda_mg']

    # "constants"
    cp.TCMB = cosmo['T_CMB']

    # neutrinos
    # We maually setup the CAMB neutrinos to match the adjustments CLASS
    # makes to their temperatures.
    cp.share_delta_neff = False
    cp.omnuh2 = cosmo['Omega_nu_mass'] * h2
    cp.num_nu_massless = cosmo['N_nu_rel']
    cp.num_nu_massive = int(cosmo['N_nu_mass'])
    cp.nu_mass_eigenstates = int(cosmo['N_nu_mass'])

    delta_neff = cosmo['Neff'] - 3.046  # used for BBN YHe comps

    # ISiTGR built on CAMB which defines a neutrino degeneracy
    # factor as T_i = g^(1/4)*T_nu
    # where T_nu is the standard neutrino temperature from first order
    # computations
    # CLASS defines the temperature of each neutrino species to be
    # T_i_eff = TNCDM * T_cmb where TNCDM is a fudge factor to get the
    # total mass in terms of eV to match second-order computations of the
    # relationship between m_nu and Omega_nu.
    # We are trying to get both codes to use the same neutrino temperature.
    # thus we set T_i_eff = T_i = g^(1/4) * T_nu and solve for the right
    # value of g for CAMB. We get g = (TNCDM / (11/4)^(-1/3))^4
    g = np.power(
        physical_constants.TNCDM / np.power(11.0/4.0, -1.0/3.0),
        4.0)

    if cosmo['N_nu_mass'] > 0:
        nu_mass_fracs = cosmo['m_nu'][:cosmo['N_nu_mass']]
        nu_mass_fracs = nu_mass_fracs / np.sum(nu_mass_fracs)

        cp.nu_mass_numbers = np.ones(cosmo['N_nu_mass'], dtype=np.int)
        cp.nu_mass_fractions = nu_mass_fracs
        cp.nu_mass_degeneracies = np.ones(int(cosmo['N_nu_mass'])) * g
    else:
        cp.nu_mass_numbers = []
        cp.nu_mass_fractions = []
        cp.nu_mass_degeneracies = []

    # get YHe from BBN
    cp.bbn_predictor = isitgr.bbn.get_predictor()
    cp.YHe = cp.bbn_predictor.Y_He(
        cp.ombh2 * (isitgr.constants.COBE_CMBTemp / cp.TCMB) ** 3,
        delta_neff)

    camb_de_models = ['DarkEnergyPPF', 'ppf', 'DarkEnergyFluid', 'fluid']
    camb_de_model = extra_camb_params.get('dark_energy_model', 'fluid')
    if camb_de_model not in camb_de_models:
        raise ValueError("The only dark energy models CCL supports with"
                         " camb are fluid and ppf.")
    cp.set_classes(
        dark_energy_model=camb_de_model
    )
    if camb_de_model not in camb_de_models[:2] and cosmo['wa'] and \
            (cosmo['w0'] < -1 - 1e-6 or
                1 + cosmo['w0'] + cosmo['wa'] < - 1e-6):
        raise ValueError("If you want to use w crossing -1,"
                         " then please set the dark_energy_model to ppf.")
    cp.DarkEnergy.set_params(
        w=cosmo['w0'],
        wa=cosmo['wa']
    )
    # cp.set_cosmology()
    cp.set_matter_power(
        redshifts=[_z for _z in zs],
        kmax=10,
        nonlinear=False)
    assert cp.NonLinear == isitgr.model.NonLinear_none

    cp.set_for_lmax(5000)
    cp.InitPower.set_params(
        As=A_s_fid,
        ns=cosmo['n_s'])

    # run ISITGR and get results
    isitgr_res = isitgr.get_results(cp)
    k, z, pk = isitgr_res.get_linear_matter_power_spectrum(
        hubble_units=True, nonlinear=False)

    # convert to non-h inverse units
    k *= cosmo['h']
    pk /= (h2 * cosmo['h'])

    # now build interpolant
    nk = k.shape[0]
    lk_arr = np.log(k)
    a_arr = 1.0 / (1.0 + z)
    na = a_arr.shape[0]
    sinds = np.argsort(a_arr)
    a_arr = a_arr[sinds]
    ln_p_k_and_z = np.zeros((na, nk), dtype=np.float64)
    for i, sind in enumerate(sinds):
        ln_p_k_and_z[i, :] = np.log(pk[sind, :])

    pk_lin = Pk2D(
        pkfunc=None,
        a_arr=a_arr,
        lk_arr=lk_arr,
        pk_arr=ln_p_k_and_z,
        is_logp=True,
        extrap_order_lok=1,
        extrap_order_hik=2,
        cosmo=cosmo)
    return pk_lin


def get_class_pk_lin(cosmo):
    """Run CLASS and return the linear power spectrum.

    Args:
        cosmo (:class:`~pyccl.core.Cosmology`): Cosmological
            parameters. The cosmological parameters with
            which to run CLASS.

    Returns:
        :class:`~pyccl.pk2d.Pk2D`: Power spectrum object.\
            The linear power spectrum.
    """
    import classy

    params = {
        "output": "mPk",
        "non linear": "none",
        "P_k_max_1/Mpc": cosmo.cosmo.spline_params.K_MAX_SPLINE,
        "z_max_pk": 1.0/cosmo.cosmo.spline_params.A_SPLINE_MINLOG_PK-1.0,
        "modes": "s",
        "lensing": "no",
        "h": cosmo["h"],
        "Omega_cdm": cosmo["Omega_c"],
        "Omega_b": cosmo["Omega_b"],
        "Omega_k": cosmo["Omega_k"],
        "n_s": cosmo["n_s"]}

    # cosmological constant?
    # set Omega_Lambda = 0.0 if w !=-1 or wa != 0
    if cosmo['w0'] != -1 or cosmo['wa'] != 0:
        params["Omega_Lambda"] = 0
        params['w0_fld'] = cosmo['w0']
        params['wa_fld'] = cosmo['wa']

    # neutrino parameters
    # massless neutrinos
    if cosmo["N_nu_rel"] > 1e-4:
        params["N_ur"] = cosmo["N_nu_rel"]
    else:
        params["N_ur"] = 0.0

    # massive neutrinos
    if cosmo["N_nu_mass"] > 0:
        params["N_ncdm"] = cosmo["N_nu_mass"]
        masses = lib.parameters_get_nu_masses(cosmo._params, 3)
        params["m_ncdm"] = ", ".join(
            ["%g" % m for m in masses[:cosmo["N_nu_mass"]]])

    params["T_cmb"] = cosmo["T_CMB"]

    # if we have sigma8, we need to find A_s
    if np.isfinite(cosmo["A_s"]):
        params["A_s"] = cosmo["A_s"]
    elif np.isfinite(cosmo["sigma8"]):
        # in this case, CCL will internally normalize for us when we init
        # the linear power spectrum - so we just get close
        A_s_fid = 2.43e-9 * (cosmo["sigma8"] / 0.87659)**2
        params["A_s"] = A_s_fid
    else:
        raise CCLError(
            "Could not normalize the linear power spectrum! "
            "A_s = %f, sigma8 = %f" % (
                cosmo['A_s'], cosmo['sigma8']))

    model = None
    try:
        model = classy.Class()
        model.set(params)
        model.compute()

        # Set k and a sampling from CCL parameters
        nk = lib.get_pk_spline_nk(cosmo.cosmo)
        na = lib.get_pk_spline_na(cosmo.cosmo)
        status = 0
        a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
        check(status, cosmo=cosmo)

        # FIXME - getting the lowest CLASS k value from the python interface
        # appears to be broken - setting to 1e-5 which is close to the
        # old value
        lk_arr = np.log(np.logspace(
            -5,
            np.log10(cosmo.cosmo.spline_params.K_MAX_SPLINE), nk))

        # we need to cut this to the max value used for calling CLASS
        msk = lk_arr < np.log(cosmo.cosmo.spline_params.K_MAX_SPLINE)
        nk = int(np.sum(msk))
        lk_arr = lk_arr[msk]

        # now do interp by hand
        ln_p_k_and_z = np.zeros((na, nk), dtype=np.float64)
        for aind in range(na):
            z = max(1.0 / a_arr[aind] - 1, 1e-10)
            for kind in range(nk):
                ln_p_k_and_z[aind, kind] = np.log(
                    model.pk_lin(np.exp(lk_arr[kind]), z))
    finally:
        if model is not None:
            model.struct_cleanup()
            model.empty()

    params["P_k_max_1/Mpc"] = cosmo.cosmo.spline_params.K_MAX_SPLINE

    # make the Pk2D object
    pk_lin = Pk2D(
        pkfunc=None,
        a_arr=a_arr,
        lk_arr=lk_arr,
        pk_arr=ln_p_k_and_z,
        is_logp=True,
        extrap_order_lok=1,
        extrap_order_hik=2,
        cosmo=cosmo)

    return pk_lin
