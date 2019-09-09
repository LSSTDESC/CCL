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
        "You must have the python packahe for CAMB "
        "installed to run CCL with CAMB!")

    # z sampling from CCL parameters
    na = lib.get_pk_spline_na(cosmo.cosmo)
    status = 0
    a_arr, status = lib.get_pk_spline_a(cosmo.cosmo, na, status)
    check(status)
    zs = 1.0 / a_arr - 1
    zs = np.clip(zs, 0, np.inf)
    assert np.allclose(zs[-1], 0)

    # init camb params
    h2 = cosmo['h']**2
    kwargs = dict(
        WantTransfer=True,  # compute the matter transfer function
        NonLinear=camb.model.NonLinear_none,  # only linear P(k)
        ns=cosmo['n_s'],
        H0=cosmo['h'] * 100,
        ombh2=cosmo['Omega_b'] * h2,
        omch2=cosmo['Omega_c'] * h2,
        omk=cosmo['Omega_k'],
        w=cosmo['w0'],
        wa=cosmo['wa'],
        dark_energy_model='DarkEnergyFluid',
        kmax=10,
        redshifts=[_z for _z in zs],
        # FIXME - neutrinos!
    )

    if np.isfinite(cosmo["A_s"]):
        kwargs["As"] = cosmo["A_s"]
    else:
        assert np.isfinite(cosmo["sigma8"])
        A_s_fid = 2.43e-9 * (cosmo["sigma8"] / 0.87659)**2
        kwargs["As"] = A_s_fid

    camb_params = camb.set_params(**kwargs)
    camb_res = camb.get_results(camb_params)

    k, z, pk = camb_res.get_linear_matter_power_spectrum(
        hubble_units=True, nonlinear=False)
    # convert to non-h inverse units
    k *= cosmo['h']
    pk /= (h2 * cosmo['h'])

    # make sure to deal with sigma8 if needed
    if np.isfinite(cosmo["sigma8"]):
        fac = (cosmo['sigma8'] / camb_res.get_sigma8()[-1])**2
    else:
        fac = 1
    pk *= fac

    # now build interpolant
    nk = k.shape[0]
    lk_arr = np.log(k)

    a_arr = 1.0 / (1.0 + z)
    assert a_arr.shape[0] == na
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

    # if w have sigma8, we need to find A_s
    if np.isfinite(cosmo["A_s"]):
        params["A_s"] = cosmo["A_s"]
    else:
        assert np.isfinite(cosmo["sigma8"])
        A_s_fid = 2.43e-9 * (cosmo["sigma8"] / 0.87659)**2
        params["A_s"] = A_s_fid

    model = None
    try:
        model = classy.Class()
        model.set(params)
        model.compute()

        if np.isfinite(cosmo["sigma8"]):
            fac = (cosmo['sigma8'] / model.sigma8())**2
        else:
            fac = 1.0

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
                    model.pk_lin(np.exp(lk_arr[kind]), z) * fac)
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
