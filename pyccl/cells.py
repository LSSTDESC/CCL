import warnings as _warnings

import numpy as np

from pyccl import DEFAULT_POWER_SPECTRUM, CCLWarning, check, lib, warnings
from .pyutils import integ_types
from .nonlimber_fkem.core import nonlimber_fkem

__all__ = ("angular_cl",)


def angular_cl(
    cosmo,
    tracer1,
    tracer2,
    ell,
    *,
    p_of_k_a=DEFAULT_POWER_SPECTRUM,
    l_limber=-1,              # OLD name (deprecated)
    limber_max_error=0.01,
    limber_integration_method="qag_quad",
    non_limber_integration_method="FKEM",
    fkem_chi_min=None,  # OLD name (deprecated)
    fkem_Nchi=None,  # OLD name (deprecated)
    p_of_k_a_lin=DEFAULT_POWER_SPECTRUM,
    return_meta=False,
    ell_limber=None,  # NEW name for v4
    chi_min_fkem=None,  # NEW name for v4
    n_chi_fkem=None,  # NEW for v4
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
        l_limber (int, float or 'auto'):
            Angular wavenumber beyond which Limber's approximation will be used.
            Deprecated alias for ``ell_limber``. Will be removed in CCL v4.
        limber_max_error (float) : Maximum fractional error for Limber integration.
        limber_integration_method (string) : integration method to be used
            for the Limber integrals. Possibilities: 'qag_quad' (GSL's `qag`
            method backed up by `quad` when it fails) and 'spline' (the
            integrand is splined and then integrated numerically).
        non_limber_integration_method (string) : integration method to be used
            for the non-Limber integrals. Currently the only method implemented
            is ``'FKEM'`` (see the `N5K paper <https://arxiv.org/abs/2212.04291>`_
            for details).
        chi_min_fkem: Minimum comoving distance used by `FKEM` to sample the
            tracer radial kernels. If ``None``, the minimum distance over which
            the kernels are defined will be used (capped to 1E-6 Mpc if this
            value is zero). Users are encouraged to experiment with this parameter
            and ``n_chi_fkem`` to ensure the robustness of the output
            :math:`C_\\ell` s.
        fkem_chi_min (float or None):
            Deprecated alias for ``chi_min_fkem``. Will be removed in CCL v4.
        n_chi_fkem: Number of values of the comoving distance over which `FKEM`
            will interpolate the radial kernels. If ``None`` the smallest number
            over which the kernels are currently sampled will be used. Note that
            `FKEM` will use a logarithmic sampling for distances between
            ``chi_min_fkem`` and the maximum distance over which the tracers
            are defined.  Users are encouraged to experiment with this parameter
            and ``chi_min_fkem`` to ensure the robustness of the output
            :math:`C_\\ell` s.
        fkem_Nchi (:obj:`int` or :obj:`None`):
            Deprecated alias for ``n_chi_fkem``. Will be removed in CCL v4.
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
        msg = (
            "Limber integration method "
            f"{limber_integration_method} not supported"
        )
        raise ValueError(msg)

    if non_limber_integration_method not in ["FKEM"]:
        msg = (
            "Non-Limber integration method "
            f"{non_limber_integration_method} not supported"
        )
        raise ValueError(msg)

    # Backwards-compat: support both `ell_limber` (new) and `l_limber` (old).
    # - If only `l_limber` is set, we treat it as `ell_limber` and emit a
    #   deprecation warning.
    # - If both are set, we raise, so users can't mix old and new names.
    if ell_limber is not None and l_limber != -1:
        raise ValueError(
            "Pass only one of `l_limber` (deprecated) or `ell_limber`."
        )

    if ell_limber is None:
        ell_limber_eff = l_limber
        if l_limber != -1:
            _warnings.warn(
                "`l_limber` is deprecated and will be removed in CCL v4. "
                "Use `ell_limber` instead.",
                FutureWarning,
                stacklevel=2,
            )
    else:
        ell_limber_eff = ell_limber
        if l_limber != -1:
            raise ValueError(
                "Pass only one of `l_limber` (deprecated) or `ell_limber`."
            )

    # Same pattern for chi_min:
    #   - prefer the new `chi_min_fkem`,
    #   - allow `fkem_chi_min` with a deprecation warning,
    #   - forbid passing both.
    if chi_min_fkem is not None and fkem_chi_min is not None:
        raise ValueError(
            "Pass only one of `fkem_chi_min` (deprecated) or "
            "`chi_min_fkem`."
        )

    if chi_min_fkem is None:
        chi_min_fkem_eff = fkem_chi_min
        if fkem_chi_min is not None:
            _warnings.warn(
                "`fkem_chi_min` is deprecated and will be removed in CCL v4. "
                "Use `chi_min_fkem` instead.",
                FutureWarning,
                stacklevel=2,
            )
    else:
        chi_min_fkem_eff = chi_min_fkem

    # And for the FKEM chi sampling:
    #   - `n_chi_fkem` is the new name,
    #   - `fkem_Nchi` is the deprecated alias.
    if n_chi_fkem is not None and fkem_Nchi is not None:
        raise ValueError(
            "Pass only one of `fkem_Nchi` (deprecated) or `n_chi_fkem`."
        )

    if n_chi_fkem is None:
        n_chi_fkem_eff = fkem_Nchi
        if fkem_Nchi is not None:
            _warnings.warn(
                "`fkem_Nchi` is deprecated and will be removed in CCL v4. "
                "Use `n_chi_fkem` instead.",
                FutureWarning,
                stacklevel=2,
            )
    else:
        n_chi_fkem_eff = n_chi_fkem

    if isinstance(ell_limber_eff, str):
        if ell_limber_eff != "auto":
            raise ValueError("ell_limber must be an integer or 'auto'")
        auto_limber = True
    else:
        auto_limber = False

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
    ell0 = ell_use[0]
    is_valid_eff = (
        isinstance(ell_limber_eff, (int, float))
        and 0 < ell_limber_eff < ell0
    )
    if (
            non_limber_integration_method == "FKEM"
            and not auto_limber
            and is_valid_eff
    ):
        raise ValueError(
            "For FKEM non-Limber integration, a positive `ell_limber` must be "
            "at least as large as the smallest requested ell."
        )

    cl_non_limber = np.array([])

    # We always call FKEM with the *effective* parameters:
    #   - ell_limber_eff: resolved from old/new names,
    #   - chi_min_fkem_eff, n_chi_fkem_eff: same idea for chi grid.
    if auto_limber or (
            not isinstance(ell_limber_eff, str) and ell_use[0] < ell_limber_eff
    ):
        if non_limber_integration_method == "FKEM":
            ell_limber_eff, cl_non_limber, status = nonlimber_fkem(
                cosmo=cosmo,
                tracer1=tracer1,
                tracer2=tracer2,
                p_of_k_a=p_of_k_a,
                ell=ell_use,
                ell_limber=ell_limber_eff,
                pk_linear=p_of_k_a_lin,
                limber_max_error=limber_max_error,
                n_chi_fkem=n_chi_fkem_eff,
                chi_min_fkem=chi_min_fkem_eff,
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
        # Here we report the *effective* Limber scale;
        # keep both keys for backwards compat.
        meta = {"ell_limber": ell_limber_eff, "l_limber": ell_limber_eff}

    check(status, cosmo=cosmo)
    return (cl, meta) if return_meta else cl
