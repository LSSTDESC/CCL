import numpy as np

try:
    import classy
    HAVE_CLASS = True
except ImportError:
    HAVE_CLASS = False

try:
    import camb
    import camb.model
    HAVE_CAMB = True
except ImportError:
    HAVE_CAMB = False

from . import ccllib as lib
from .pyutils import check
from .pk2d import Pk2D
from .errors import CCLError


def get_camb_pk_lin(cosmo):
    """Run CAMB and return the linear power spectrum.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
            The cosmological parameters with which to run CAMB.

    Returns:
        pk_lin (:obj:`Pk2D`): power spectrum object
            The linear power spectrum.
    """

    assert HAVE_CAMB, (
        "You must have the `camb` python package "
        "installed to run CCL with CAMB!")

    # z sampling from CCL parameters
    na = lib.get_pk_spline_na(cosmo.cosmo)
    status = 0
    a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
    check(status)
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
    h2 = cosmo['h']**2
    kwargs = dict(
        # basic background stuff
        H0=cosmo['h'] * 100,
        ombh2=cosmo['Omega_b'] * h2,
        omch2=cosmo['Omega_c'] * h2,
        omk=cosmo['Omega_k'],

        # neutrinos
        omnuh2=cosmo['Omega_n_mass'] * h2,
        nnu=cosmo['Neff'],
        standard_neutrino_neff=3.046,
        # CAMB and CLASS do slightly different things here
        # CLASS has extra factors in the relationship between the CMB and
        # neutrino temperature, which causes cosmo['N_nu_rel'] to not be
        # exactly cosmo['Neff'] - cosmo['N_nu_mass'].
        # CAMB internally does not do this, so we are left with having
        # to fudge things a bit here. :/
        num_nu_massless=cosmo['Neff'] - cosmo['N_nu_mass'],
        num_nu_massive=int(cosmo['N_nu_mass']),
        share_delta_neff=True,

        # dark energy
        dark_energy_model='fluid',
        w=cosmo['w0'],
        wa=cosmo['wa'],

        # matter P(k) outputs
        redshifts=[_z for _z in zs],
        kmax=10,
        nonlinear=False,  # no halofit
        WantTransfer=True,  # compute the matter transfer function

        # ICs
        As=A_s_fid,
        ns=cosmo['n_s'],

        # "constants"
        TCMB=cosmo['T_CMB'],
    )

    if cosmo['N_nu_mass'] > 0:
        nu_mass_fracs = cosmo['mnu'][:cosmo['N_nu_mass']]
        nu_mass_fracs = nu_mass_fracs / np.sum(nu_mass_fracs)
        kwargs['nu_mass_eigenstates'] = int(cosmo['N_nu_mass'])
        kwargs['nu_mass_numbers'] = np.ones(cosmo['N_nu_mass'], dtype=np.int)
        kwargs['nu_mass_fractions'] = nu_mass_fracs
        kwargs['nu_mass_degeneracies'] = (
            np.ones(int(cosmo['N_nu_mass'])) *
            cosmo['Neff'] / int(cosmo['Neff']))

    cp = camb.set_params(**kwargs)

    # internally CAMB will try to set these using some default
    # assumptions that we do not want
    # thus we reset them here
    if cosmo['N_nu_mass'] > 0:
        cp.nu_mass_degeneracies = (
            np.ones(int(cosmo['N_nu_mass'])) *
            cosmo['Neff'] / int(cosmo['Neff']))
        nu_mass_fracs = cosmo['mnu'][:cosmo['N_nu_mass']]
        nu_mass_fracs = nu_mass_fracs / np.sum(nu_mass_fracs)
        cp.nu_mass_eigenstates = int(cosmo['N_nu_mass'])
        cp.nu_mass_numbers = np.ones(cosmo['N_nu_mass'], dtype=np.int)
        cp.nu_mass_fractions = nu_mass_fracs
    else:
        cp.nu_mass_eigenstates = 0
        cp.nu_mass_numbers = []
        cp.nu_mass_fractions = []
        cp.nu_mass_degeneracies = []

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

    return pk_lin


def get_class_pk_lin(cosmo):
    """Run CLASS and return the linear power spectrum.

    Args:
        cosmo (:obj:`Cosmology`): Cosmological parameters.
            The cosmological parameters with which to run CLASS.

    Returns:
        pk_lin (:obj:`Pk2D`): power spectrum object
            The linear power spectrum.
    """

    assert HAVE_CLASS, (
        "You must have the python wrapper for CLASS "
        "installed to run CCL with CLASS!")

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
        check(status)

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
