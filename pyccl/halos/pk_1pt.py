__all__ = ("halomod_mean_profile_1pt", "halomod_bias_1pt",)

import numpy as np

from .. import warn_api


def _Ix1(func, cosmo, hmc, k, a, prof, normprof):
    # I_X_1 dispatcher for internal use
    """
    Args:
        cosmo (:class:`~pyccl.cosmology.Cosmology`): a Cosmology object.
        hmc (:class:`~pyccl.halos.halo_model.HMCalculator`): a halo model
            calculator.
        k (float or array_like): comoving wavenumber in
            :math:`{\\rm Mpc}^{-1}`.
        a (float or array_like): scale factor.
        prof (:class:`~pyccl.halos.profiles.profile_base.HaloProfile`): halo
            profile.
        normprof (bool): (Deprecated - do not use)
            if `True`, this integral will be
            normalized by :math:`I^0_1(k\\rightarrow 0,a|u)`,
            where :math:`u` is the profile represented by ``prof``.

    Returns:
        float or array_like: integral values evaluated at each
        combination of ``k`` and ``a``. The shape of the output will
        be ``(N_a, N_k)`` where ``N_k`` and ``N_a`` are the sizes of
        ``k`` and ``a`` respectively. If ``k`` or ``a`` are scalars, the
        corresponding dimension will be squeezed out on output.
    """
    func = getattr(hmc, func)

    a_use = np.atleast_1d(a).astype(float)
    k_use = np.atleast_1d(k).astype(float)

    na = len(a_use)
    nk = len(k_use)
    out = np.zeros([na, nk])
    for ia, aa in enumerate(a_use):
        i11 = func(cosmo, k_use, aa, prof)
        norm = prof.get_normalization(cosmo, aa, hmc=hmc) if normprof else 1
        # TODO: CCLv3 remove if
        out[ia] = i11 / norm

    if np.ndim(a) == 0:
        out = np.squeeze(out, axis=0)
    if np.ndim(k) == 0:
        out = np.squeeze(out, axis=-1)
    return out


@warn_api
def halomod_mean_profile_1pt(cosmo, hmc, k, a, prof, *, normprof=None):
    """ Returns the mass-weighted mean halo profile.

    .. math::
        I^0_1(k,a|u) = \\int dM\\,n(M,a)\\,\\langle u(k,a|M)\\rangle,

    where :math:`n(M,a)` is the halo mass function, and
    :math:`\\langle u(k,a|M)\\rangle` is the halo profile as a
    function of scale, scale factor and halo mass.
    """
    return _Ix1("I_0_1", cosmo, hmc, k, a, prof, normprof)


@warn_api
def halomod_bias_1pt(cosmo, hmc, k, a, prof, *, normprof=None):
    """ Returns the mass-and-bias-weighted mean halo profile.

    .. math::
        I^1_1(k,a|u) = \\int dM\\,n(M,a)\\,b(M,a)\\,
        \\langle u(k,a|M)\\rangle,

    where :math:`n(M,a)` is the halo mass function,
    :math:`b(M,a)` is the halo bias, and
    :math:`\\langle u(k,a|M)\\rangle` is the halo profile as a
    function of scale, scale factor and halo mass.
    """
    return _Ix1("I_1_1", cosmo, hmc, k, a, prof, normprof)


halomod_mean_profile_1pt.__doc__ += _Ix1.__doc__
halomod_bias_1pt.__doc__ += _Ix1.__doc__
