__all__ = ("HaloProfileEinasto",)

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gamma, gammainc

from .. import MassDef, mass_translator, get_delta_c
from . import HaloProfileMatter


class HaloProfileEinasto(HaloProfileMatter):
    """ `Einasto 1965
    <https://ui.adsabs.harvard.edu/abs/1965TrAlm...5...87E/abstract>`_
    profile.

    .. math::
       \\rho(r) = \\rho_0\\,\\exp(-2 ((r/r_s)^\\alpha-1) / \\alpha)

    where :math:`r_s` is related to the comoving spherical overdensity
    halo radius :math:`r_\\Delta(M)` through the concentration
    parameter :math:`c(M)` as

    .. math::
       r_\\Delta(M) = c(M)\\,r_s

    and the normalization :math:`\\rho_0` is

    .. math::
       \\rho_0 = \\frac{M}{4\\pi\\,r_s^3}
       \\frac{2^{(3/\\alpha)}\\,\\alpha^{(1-3/\\alpha)}
       \\,{\\rm exp}(-2/\\alpha)}
       {\\gamma(\\frac{3}{\\alpha}, \\frac{2}{\\alpha}c^{\\alpha})},

    where :math:`\\gamma` is the lower incomplete gamma function.

    The index :math:`\\alpha` can be a free parameter or dependent on halo mass
    and redshift. In the latter case, we use the parameterization of
    `Diemer & Kravtsov <https://arxiv.org/abs/1401.1216>`_.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        concentration (:class:`~pyccl.halos.halo_model_base.Concentration`):
            concentration-mass relation to use with this profile.
        truncated (:obj:`bool`): set to ``True`` if the profile should be
            truncated at :math:`r = r_\\Delta`.
        projected_quad (:obj:`bool`): set to ``True`` to calculate the
            projected profile with numerical integration.
        alpha (:obj:`float` or :obj:`str`): :math:`\\alpha` parameter, or
            set to ``'cosmo'`` to calculate the value from cosmology.
    """
    __repr_attrs__ = __eq_attrs__ = (
        "truncated", "alpha", "projected_quad",
        "mass_def", "concentration", "precision_fftlog",)

    def __init__(self, *, mass_def, concentration,
                 truncated=False,
                 projected_quad=False,
                 alpha='cosmo'):
        self.truncated = truncated
        self.projected_quad = projected_quad
        self.alpha = alpha
        if projected_quad:
            if truncated:
                raise ValueError("projected_quad profile not supported "
                                 "for truncated Einasto. Set `truncated` or "
                                 "`projected_quad` to `False`.")
            self._projected = self._projected_quad
        super().__init__(mass_def=mass_def, concentration=concentration)
        self._to_virial_mass = mass_translator(
            mass_in=self.mass_def, mass_out=MassDef("vir", "matter"),
            concentration=self.concentration)
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=1000,
                                     plaw_fourier=-2.)

    def update_parameters(self, alpha=None):
        """Update any of the parameters associated with this profile.
        Any parameter set to ``None`` won't be updated.

        Args:
            alpha (:obj:`float` or :obj:`str`): :math:`\\alpha` parameter, or
                set to ``'cosmo'`` to calculate the value from cosmology.
        """
        if alpha is not None and alpha != self.alpha:
            self.alpha = alpha

    def _get_alpha(self, cosmo, M, a):
        if self.alpha == 'cosmo':
            Mvir = self._to_virial_mass(cosmo, M, a)
            sM = cosmo.sigmaM(Mvir, a)
            nu = get_delta_c(cosmo, a, kind='EdS_approx') / sM
            return 0.155 + 0.0095 * nu * nu
        return np.full_like(M, self.alpha)

    def _norm(self, M, Rs, c, alpha):
        # Einasto normalization from mass, radius, concentration and alpha
        return M / (np.pi * Rs**3 * 2**(2-3/alpha) * alpha**(-1+3/alpha)
                    * np.exp(2/alpha)
                    * gamma(3/alpha) * gammainc(3/alpha, 2/alpha*c**alpha))

    def _real(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M

        alpha = self._get_alpha(cosmo, M_use, a)

        norm = self._norm(M_use, R_s, c_M, alpha)

        x = r_use[None, :] / R_s[:, None]
        prof = norm[:, None] * np.exp(-2. * (x**alpha[:, None] - 1) /
                                      alpha[:, None])
        if self.truncated:
            prof[r_use[None, :] > R_M[:, None]] = 0

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _projected_quad(self, cosmo, r, M, a):
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Comoving virial radius
        R_M = self.mass_def.get_radius(cosmo, M_use, a) / a
        c_M = self.concentration(cosmo, M_use, a)
        R_s = R_M / c_M

        alpha = self._get_alpha(cosmo, M_use, a)

        prof, _ = quad_vec(
            self._projected_quad_integrand, 0., np.inf,
            args=(r_use[None, :], R_s[:, None], alpha[:, None]))

        prof *= 2 * self._norm(M_use, R_s, c_M, alpha)[:, None]

        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)
        return prof

    def _projected_quad_integrand(self, z, R, R_s, alpha):
        x = np.sqrt(z**2. + R**2.) / R_s
        return np.exp(-2. * (x**alpha - 1.) / alpha)
