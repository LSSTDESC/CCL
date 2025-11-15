__all__ = ("angular_cl",)

import numpy as np

from pyccl import DEFAULT_POWER_SPECTRUM, CCLWarning, check, lib, warnings
from .pyutils import integ_types
from .nonlimber_fkem.core import nonlimber_fkem


def angular_cl(
    cosmo,
    tracer1,
    tracer2,
    ell,
    *,
    p_of_k_a=DEFAULT_POWER_SPECTRUM,
    ell_limber=-1,
    limber_max_error=0.01,
    limber_integration_method="qag_quad",
    non_limber_integration_method="FKEM",
    fkem_chi_min=None,
    fkem_nchi=None,
    p_of_k_a_lin=DEFAULT_POWER_SPECTRUM,
    return_meta=False,
):
    """Calculate the angular (cross-)power spectrum for a pair of tracers.

    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): A Cosmology object.
        tracer1 (:class:`~pyccl.tracers.Tracer`): a Tracer object,
            of any kind.
        tracer2 (:class:`~pyccl.tracers.Tracer`): a second Tracer object.
        ell (:obj:`float` or `array`): Angular multipole(s) at which to evaluate
            the angular power spectrum.
        p_of_k_a (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`): 3D Power
            spectrum to project. If a string, it must correspond to one of
            the non-linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).
        ell_limber (int, float or 'auto') : Angular wavenumber beyond which
            Limber's approximation will be used. Defaults to -1. If 'auto',
            then the non-limber integrator will be used to compute the right
            transition point given the value of limber_max_error.
        limber_max_error (float) : Maximum fractional error for Limber integration.
        limber_integration_method (string) : integration method to be used
            for the Limber integrals. Possibilities: 'qag_quad' (GSL's `qag`
            method backed up by `quad` when it fails) and 'spline' (the
            integrand is splined and then integrated numerically).
        non_limber_integration_method (string) : integration method to be used
            for the non-Limber integrals. Currently the only method implemented
            is ``'FKEM'`` (see the `N5K paper <https://arxiv.org/abs/2212.04291>`_
            for details).
        fkem_chi_min: Minimum comoving distance used by `FKEM` to sample the
            tracer radial kernels. If ``None``, the minimum distance over which
            the kernels are defined will be used (capped to 1E-6 Mpc if this
            value is zero). Users are encouraged to experiment with this parameter
            and ``fkem_nchi`` to ensure the robustness of the output
            :math:`C_\\ell` s.
        fkem_nchi: Number of values of the comoving distance over which `FKEM`
            will interpolate the radial kernels. If ``None`` the smallest number
            over which the kernels are currently sampled will be used. Note that
            `FKEM` will use a logarithmic sampling for distances between
            ``fkem_chi_min`` and the maximum distance over which the tracers
            are defined.  Users are encouraged to experiment with this parameter
            and ``fkem_chi_min`` to ensure the robustness of the output
            :math:`C_\\ell` s.
        p_of_k_a_lin (:class:`~pyccl.pk2d.Pk2D`, :obj:`str` or :obj:`None`):
            3D linear Power spectrum to project, for special use in
            PT calculations using the FKEM non-limber integration technique.
            If a string, it must correspond to one of
            the linear power spectra stored in `cosmo` (e.g.
            `'delta_matter:delta_matter'`).
        return_meta (bool): if `True`, also return a dictionary with various
            metadata about the calculation, such as ell_limber as calculated by the
            non-limber integrator.

    Returns:
        :obj:`float` or `array`: Angular (cross-)power spectrum values, \
            :math:`C_\\ell`, for the pair of tracers, as a function of \
            :math:`\\ell`.
    """  # noqa
    if cosmo["Omega_k"] != 0:
        warnings.warn(
            "CCL does not properly use the hyperspherical Bessel functions "
            "when computing angular power spectra in non-flat cosmologies!",
            category=CCLWarning,
            importance="low",
        )

    if limber_integration_method not in integ_types:
        raise ValueError(
            "Limber integration method %s not supported"
            % limber_integration_method
        )
    if non_limber_integration_method not in ["FKEM"]:
        raise ValueError(
            "Non-Limber integration method %s not supported"
            % limber_integration_method
        )
    if type(ell_limber) is str:
        if ell_limber != "auto":
            raise ValueError("l_limber must be an integer or'auto'")
        auto_limber = True
    else:
        auto_limber = False

    # we need the distances for the integrals
    cosmo.compute_distances()

    if p_of_k_a is None:
        p_of_k_a = DEFAULT_POWER_SPECTRUM
    psp = cosmo.parse_pk2d(p_of_k_a, is_linear=False)

    # Create tracer colections
    status = 0
    clt1, status = lib.cl_tracer_collection_t_new(status)
    clt2, status = lib.cl_tracer_collection_t_new(status)
    for t in tracer1._trc:
        status = lib.add_cl_tracer_to_collection(clt1, t, status)
    for t in tracer2._trc:
        status = lib.add_cl_tracer_to_collection(clt2, t, status)

    ell_use = np.atleast_1d(ell)
    if not (np.diff(ell_use) > 0).all():
        raise ValueError("ell values must be monotonically increasing")

    # sanity check for non-Limber configuration:
    # If the user requested FKEM with a positive ell_limber, it must not be
    # smaller than the smallest ell. We allow:
    #   - ell_limber <= 0  : treated as "no FKEM, pure Limber"
    #   - ell_limber == min(ell) : FKEM active starting at the first ell
    if (
        non_limber_integration_method == "FKEM"
        and not auto_limber
        and isinstance(ell_limber, (int, float))
        and ell_limber > 0
        and ell_limber < ell_use[0]
    ):
        raise ValueError(
            "For FKEM non-Limber integration, a positive `ell_limber` must be "
            "at least as large as the smallest requested ell."
        )

    cl_non_limber = np.array([])

    if auto_limber or (
        not isinstance(ell_limber, str) and ell_use[0] < ell_limber
    ):
        if non_limber_integration_method == "FKEM":
            ell_limber_eff, cl_non_limber, status = nonlimber_fkem(
                cosmo=cosmo,
                tracer1=tracer1,
                tracer2=tracer2,
                p_of_k_a=p_of_k_a,
                ell_values=ell_use,
                ell_limber=ell_limber,
                pk_linear=p_of_k_a_lin,
                limber_max_error=limber_max_error,
                n_chi=fkem_nchi,
                chi_min=fkem_chi_min,
                n_consec_ell=3,
            )
        check(status, cosmo=cosmo)

    n_nl = cl_non_limber.size
    if n_nl > 0:
        ell_use_limber = ell_use[n_nl:]
    else:
        ell_use_limber = ell_use

    if len(ell_use_limber) > 0:
        cl_limber, status = lib.angular_cl_vec_limber(
            cosmo.cosmo,
            clt1,
            clt2,
            psp,
            ell_use_limber,
            integ_types[limber_integration_method],
            ell_use_limber.size,
            status,
        )
    else:
        cl_limber = np.array([])

    cl = np.concatenate((cl_non_limber, cl_limber))
    if np.ndim(ell) == 0:
        cl = cl[0]

    # Free up tracer collections
    lib.cl_tracer_collection_t_free(clt1)
    lib.cl_tracer_collection_t_free(clt2)

    if return_meta:
        meta = {"ell_limber": ell_limber}

    check(status, cosmo=cosmo)
    return (cl, meta) if return_meta else cl
