import numpy as np

try:
    import classy
    HAVE_CLASS = True
except ImportError:
    HAVE_CLASS = False

from . import ccllib as lib
from .pyutils import check
from .pk2d import Pk2D


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

        # if we are using sigma8, we rerun the code with a new A_s
        # FIXME - I don't think rerunning class is needed...
        if np.isfinite(cosmo["sigma8"]):
            fac = (cosmo['sigma8'] / model.sigma8())**2
            params['A_s'] = A_s_fid * fac

            model.struct_cleanup()
            model.empty()
            model = None
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
