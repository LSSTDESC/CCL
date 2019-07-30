import numpy as np

try:
    import classy
    HAVE_CLASS = True
except ImportError:
    HAVE_CLASS = False

try:
    import camb
    HAVE_CAMB = True
except ImportError:
    HAVE_CAMB = False

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
        masses = []
        for i in range(cosmo["N_nu_mass"]):
            masses.append(cosmo["mnu"][i])
        params["m_ncdm"] = ", ".join(["%g" % m for m in masses])

    params["T_cmb"] = cosmo["T_CMB"]

    # if w have sigma8, we need to find A_s
    if np.isfinite(cosmo["A_s"]):
        params["A_s"] = cosmo["A_s"]
    else:
        assert np.isfinite(cosmo["sigma8"])
        A_s_fid = 2.43e-9 * (cosmo["sigma8"] / 0.87659)**2
        params["A_s"] = A_s_fid
        params["P_k_max_1/Mpc"] = 10.0

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

        p_k_and_z, k_arr, z_arr = model.get_pk_and_k_and_z()
    finally:
        if model is not None:
            model.struct_cleanup()
            model.empty()

    params["P_k_max_1/Mpc"] = cosmo.cosmo.spline_params.K_MAX_SPLINE

    # current shape is [nk, nz] and we want [na, nk]
    # so we transpose
    p_k_and_z = p_k_and_z.T

    # convert to log
    ln_p_k_and_z = np.log(p_k_and_z)

    # convert z to a
    a_arr = 1.0 / (1.0 + z_arr)

    # make the Pk2D object
    pk_lin = Pk2D(
        pkfunc=None,
        a_arr=a_arr,
        lk_arr=np.log(k_arr),
        pk_arr=ln_p_k_and_z,
        is_logp=True,
        extrap_order_lok=1,
        extrap_order_hik=2,
        cosmo=cosmo)
    return pk_lin
