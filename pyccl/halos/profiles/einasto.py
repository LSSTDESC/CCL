from __future__ import annotations

__all__ = ("HaloProfileEinasto",)

from numbers import Real
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from scipy.special import gamma, gammainc

from ... import update, warn_api
from .. import MassDef, mass_translator
from . import HaloProfileMatter

if TYPE_CHECKING:
    from .. import Concentration


class HaloProfileEinasto(HaloProfileMatter):
    r"""Halo profile by :footcite:t:`Einasto65`.

    .. math::

        \rho(r) = \rho_0 \, \exp \left[ -\frac{2}{\alpha}
        \left( \left( \frac{r}{r_s} \right)^\alpha -1 \right) \right]


    where :math:`r_s` is related to the spherical overdensityhalo radius
    :math:`R_\Delta(M)` through the concentration parameter :math:`c(M)` as

    .. math::

        R_\Delta(M) = c(M) \, r_s

    and the normalization :math:`\rho_0` is the mean density within the
    :math:`R_\Delta(M)` of the halo. The index :math:`\alpha` depends on halo
    mass and redshift, and we use the parameterization of
    :footcite:t:`Diemer14`.

    By default, this profile is truncated at :math:`r = R_\Delta(M)`.

    Parameters
    ----------
    concentration
        Concentration-mass relation. If a string, `mass_def` must be specified.
    truncated
        If True, the profile is truncated at :math:`r = R_\Delta` .
    alpha
        Slope of the profile. If `'cosmo'`, it is calculated using the
        cosmological parameters.
    mass_def
        Halo mass definition. If `concentration` is instantiated, this
        parameter is optional.

        .. versionadded:: 2.8.0

    References
    ----------
    .. footbibliography::
    """
    __repr_attrs__ = __eq_attrs__ = (
        "truncated", "alpha", "mass_def", "concentration", "precision_fftlog",)

    @warn_api(pairs=[("c_M_relation", "concentration")])
    def __init__(
            self,
            *,
            concentration: Union[str, Concentration],
            truncated: bool = True,
            alpha: Union[Literal["cosmo"], Real] = "cosmo",
            mass_def: Optional[Union[str, MassDef]] = None
    ):
        self.truncated = truncated
        self.alpha = alpha
        super().__init__(mass_def=mass_def, concentration=concentration)
        self._init_mass_translator()
        self.update_precision_fftlog(padding_hi_fftlog=1E2,
                                     padding_lo_fftlog=1E-2,
                                     n_per_decade=1000,
                                     plaw_fourier=-2.)

    def _init_mass_translator(self):
        # Set the mass translator to Mvir as an attribute.
        # TODO: Move to `__init__` in CCLv3 (without the context).
        with self.unlock():
            self._to_virial_mass = mass_translator(
                mass_in=self.mass_def, mass_out=MassDef("vir", "matter"),
                concentration=self.concentration)

    @warn_api
    @update(names=["alpha"])
    def update_parameters(self) -> None:
        """Update the profile parameters. All numerical parameters in
        :meth:`__init__` are updatable.
        """

    def _get_alpha(self, cosmo, M, a):
        if self.alpha == 'cosmo':
            self._init_mass_translator()  # TODO: Remove for CCLv3.
            Mvir = self._to_virial_mass(cosmo, M, a)
            sM = cosmo.sigmaM(Mvir, a)
            nu = 1.686 / sM
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
