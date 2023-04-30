from __future__ import annotations

__all__ = ("DEFAULT_POWER_SPECTRUM", "CosmologyParams", "NeutrinoMassSplits",)

import warnings
from enum import Enum
from numbers import Real
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from ... import CCLDeprecationWarning, lib
from . import Parameters


DEFAULT_POWER_SPECTRUM = "delta_matter:delta_matter"
"""Name of the default power spectrum."""


class NeutrinoMassSplits(Enum):
    """Neutrino mass splits."""
    SUM = 'sum'
    """Sum of the neutrino masses."""

    SINGLE = 'single'
    """One massive neutrino."""

    EQUAL = 'equal'
    """Equally split into 3 massive neutrinos."""

    NORMAL = 'normal'
    """Normal mass hierarchy."""

    INVERTED = 'inverted'
    """Inverted mass hierarchy."""

    LIST = 'list'  # TODO: Remove for CCLv3.
    """Sequence of split masses.

    .. deprecated:: 2.8.0

        This option only exists for backwards-compatibility and will be removed
        in the next major release.
    """


class CosmologyParams(Parameters, factory=lib.parameters):
    """Instances of this class hold cosmological parameters.

    Stored in instances of :class:`~pyccl.cosmology.Cosmology`.
    """
    Omega_k: Real = 0
    r"""Density parameter of curvature, :math:`\Omega_k`.

    :meta hide-value:
    """

    k_sign: int = 0
    r"""Sign of curvature, :math:`{\rm sgn}(k)`.

    :meta hide-value:
    """

    sqrtk: Real = 0
    r"""Square root of the magnitude of curvature,
    :math:`\sqrt{ | \Omega_k | } \, h`.

    :meta hide-value:
    """

    n_s: Real = 0
    r"""Spectral index of primordial scalar pertubations, :math:`n_s`.

    :meta hide-value:
    """

    sigma8: Real = np.nan
    r"""Variance of matter density perturbations at a scale of
    :math:`8 {\rm Mpc}/h`, :math:`\sigma_8`.

    :meta hide-value:
    """

    A_s: Real = np.nan
    """Normalization of the primordial power spectrum, :math:`A_s`.

    :meta hide-value:
    """

    N_nu_mass: int = 0
    r"""Number of non-relativistic neutrino species today.

    :meta hide-value:
    """

    N_nu_rel: int = 0
    r"""Number of relativistic neutrino species today.

    :meta hide-value:
    """

    Neff: Real = 3.046  # TODO: Change default for CCLv3.
    r"""Effective number of relativistic neutrino species in the early
    universe :footcite:p:`Froustey20` :footcite:p:`Bennett21`.

    .. deprecated:: 2.8.0

        The default will change to reflect newest measurements in the next
        major relase.

    .. footbibliography::
    """

    Omega_nu_mass: Real = 0
    r"""Density parameter of massive neutrinos today,
    :math:`\Omega_{\rm \nu_m}`.

    :meta hide-value:
    """

    Omega_nu_rel: Real = 0
    r"""Density parameter of massless neutrinos today,
    :math:`\Omega_{\rm \nu_r}`.

    :meta hide-value:
    """

    sum_nu_masses: Real = 0
    r"""Sum of the neutrino masses, :math:`{\rm \Sigma} m_{\rm \nu}`.

    :meta hide-value:
    """

    m_nu: Sequence[Real] = (0,)
    r"""Neutrino masses, :math:`m_{\rm \nu}`.

    :meta hide-value:
    """

    mass_split: str = "normal"
    r"""Neutrino mass split. Available options in :class:`~NeutrinoMassSplits`.

    :meta hide-value:
    """

    T_nu: Real = 0
    r"""Temperature of neutrino background today, :math:`T_{\rm \nu}`.

    :meta hide-value:
    """

    Omega_g: Real = np.nan
    r"""Density parameter of photons today, :math:`\Omega_{\rm \gamma}`.

    :meta hide-value:
    """

    T_CMB: Real = 2.725  # TODO: Change default for CCLv3.
    r"""Temperature of the microwave background today, :math:`T_{\rm CMB}`.

    .. deprecated:: 2.8.0

        The default will change to :math:`2.7255` in the next major release.
    """
    warnings.warn(
        "The default CMB temperaHaloProfileCIBture (T_CMB) will change in "
        "CCLv3.0.0, from 2.725 to 2.7255 (Kelvin).", CCLDeprecationWarning)

    T_ncdm: Real = 0.71611
    r"""Non-CDM temperature in units of photon temperature,
    :math:`T_{\rm nCDM}`
    """

    Omega_c: Real = 0
    r"""Density parameter of cold dark matter today,
    :math:`\Omega_{{\rm CDM}, 0}`.

    :meta hide-value:
    """

    Omega_b: Real = 0
    r"""Density parameter of baryons today, :math:`\Omega_{{\rm b}, 0}`.

    :meta hide-value:
    """

    Omega_m: Real = 0
    r"""Density parameter of matter,
    :math:`\Omega_{\rm m} := \Omega_{\rm b} + \Omega_{\rm CDM}
    + \Omega_{\rm \nu_m}`.

    :meta hide-value:
    """

    h: Real = 0
    r"""Hubble parameter today, :math:`\frac{H_0}{100 \, \rm km/s/Mpc}`.

    :meta hide-value:
    """

    H0: Real = 0
    r"""Hubble constant, :math:`H_0`.

    :meta hide-value:
    """

    Omega_l: Real = 0
    r"""Density parameter of dark energy today, :math:`\Omega_{\rm \Lambda}`.

    :meta hide-value:
    """

    w0: Real = -1
    r""":math:`w_0` of the equation of state of dark energy."""

    wa: Real = 0
    r""":math:`w_a` of the equation of state of dark energy."""

    # Modified gravity
    mu_0: Real = 0
    r"""Parameter of the :math:`\mu, \Sigma` phenomenological parametrization
    of modified gravity :footcite:p:`Zhao10`.

    .. footbibliography
    """

    sigma_0: Real = 0
    r"""Parameter of the :math:`\mu, \Sigma` phenomenological parametrization
    of modified gravity :footcite:p:`Zhao10`.

    .. footbibliography
    """

    c1_mg: Real = 1
    r"""Parameter of the :math:`\mu, \Sigma` phenomenological parametrization
    of modified gravity :footcite:p:`Zhao10`.

    .. footbibliography
    """

    c2_mg: Real = 1
    r"""Parameter of the :math:`\mu, \Sigma` phenomenological parametrization
    of modified gravity :footcite:p:`Zhao10`.

    .. footbibliography
    """

    lambda_mg: Real = 0
    r"""Parameter of the :math:`\mu, \Sigma` phenomenological parametrization
    of modified gravity :footcite:p:`Zhao10`.

    .. footbibliography
    """

    # TODO: Remove all deprecated for CCLv3.
    bcm_log10Mc: Real = np.log10(1.2e14)
    r"""Parameter of the baryonic feedback model of :footcite:t:`Schneider15`.

    .. deprecated:: 2.8.0

        Use the :mod:`~pyccl.baryons` subpackage.

    :meta hide-value:

    .. footbibliography::
    """

    bcm_etab: Real = 0.5
    r"""Parameter of the baryonic feedback model of :footcite:t:`Schneider15`.

    .. deprecated:: 2.8.0

        Use the :mod:`~pyccl.baryons` subpackage.

    :meta hide-value:

    .. footbibliography::
    """

    bcm_ks: Real = 55
    r"""Parameter of the baryonic feedback model of :footcite:t:`Schneider15`.

    .. deprecated:: 2.8.0

        Use the :mod:`~pyccl.baryons` subpackage.

    :meta hide-value:

    .. footbibliography::
    """

    z_mgrowth: Optional[NDArray[Real]] = None
    r"""Modified growth :math:`z` array.

    .. deprecated:: 2.8.0

        This parameter will be removed in the next major release.

    :meta hide-value:
    """

    df_mgrowth: Optional[NDArray[Real]] = None
    r"""Modified growth array.

    .. deprecated:: 2.8.0

        This parameter will be removed in the next major release.

    :meta hide-value:
    """

    nz_mgrowth: int = 0
    r"""Number of samples of the modified growth arrays.

    .. deprecated:: 2.8.0

        This parameter will be removed in the next major release.

    :meta hide-value:
    """

    has_mgrowth: bool = False
    r"""Flag indicating whether the cosmology has modified growth.

    .. deprecated:: 2.8.0

        This parameter will be removed in the next major release.

    :meta hide-value:
    """

    @property
    def z_mg(self) -> Optional[NDArray[Real]]:
        """Alias for `z_mgrowth`."""
        return self.z_mgrowth

    @property
    def df_mg(self) -> Optional[NDArray[Real]]:
        """Alias for `df_mgrowth`."""
        return self.df_mgrowth

    def __setattr__(self, name, value):
        if name == "m_nu":
            object.__setattr__(self, "m_nu", list(value))
            return lib.parameters_m_nu_set_custom(self._instance, value)
        if name == "mgrowth":
            object.__setattr__(self, "z_mgrowth", value[0])
            object.__setattr__(self, "df_mgrowth", value[1])
            object.__setattr__(self, "nz_mgrowth", len(value[0]))
            object.__setattr__(self, "has_mgrowth", True)
            return lib.parameters_mgrowth_set_custom(self._instance, *value)
        super().__setattr__(name, value)

    def __del__(self):
        lib.parameters_free(self._instance)
