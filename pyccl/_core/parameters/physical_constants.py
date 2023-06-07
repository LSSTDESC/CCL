__all__ = ("PhysicalConstants", "physical_constants",)

from ... import ccllib as lib
from .parameters_base import Parameters


class PhysicalConstants(Parameters, instance=lib.cvar.constants, frozen=True):
    """Instances of this class hold the physical constants."""
    PI: float = 3.14_15_92_65_35_89_79_32
    r""":math:`\pi`"""

    CLIGHT: float = 299_792_458.
    r"""Speed of light, :math:`c`, in :math:`\rm [m \, s^{-1}]`."""

    CLIGHT_HMPC: float = CLIGHT / 100_000
    r"""Speed of light in :math:`{\rm Mpc}/h`, :math:`c/h`."""

    GNEWT: float = 6.674_08e-11
    r"""Universal gravitational constant, :math:`G`, in
    :math:`\rm m^3 \, kg^{-1} \, s^{-2}`.
    """

    SOLAR_MASS: float = 1.988_475_415_338_143_8E+30
    r"""Solar mass, :math:`\rm M_\odot`, in :math:`\rm kg`."""

    MPC_TO_METER: float = 3.085_677_581_491_367_399_198_952_281E+22
    r"""Megaparsec, :math:`\rm \frac{Mpc}{m}`."""

    PC_TO_METER: float = MPC_TO_METER / 1e6
    r"""Parsec, :math:`\rm \frac{pc}{m}`."""

    RHO_CRITICAL: float = 3*1e4/(8*PI*GNEWT) * 1e6 * MPC_TO_METER / SOLAR_MASS
    r"""Critical density, :math:`\rho_{\rm c}`, in
    :math:`\frac{\rm{M_\odot} / h}{({\rm Mpc} / h)^3}`.
    """

    KBOLTZ: float = 1.380_648_52E-23
    r"""Boltzmann's constant, :math:`k_B`, in :math:`\rm J \, kg^{-1}`."""

    STBOLTZ: float = 5.670_367E-8
    r"""Stefan-Boltzmann's constant, :math:`\sigma`, in
    :math:`\rm kg \, s^{-3} \, K^{-4}`.
    """

    HPLANCK: float = 6.626_070_040E-34
    r"""Planck's constant, :math:`h`, in :math:`\rm kg \, m^2 \, s^{-1}`."""

    EV_IN_J: float = 1.602_176_620_8e-19
    r"""Electron-volt, :math:`\rm \frac{eV}{J}`."""

    DELTAM12_sq: float = 7.62e-5
    r"""Difference of squared neutrino masses :footcite:p:`Lesgourgues12`,
    :math:`{\rm \Delta}m^2_{\rm 21} := m^2_{\rm 2}-m^2_{\rm 1}`, in
    :math:`10^{-5} \rm eV^2`.

    .. footbibliography::
    """

    DELTAM13_sq_pos: float = 2.55e-3
    r"""Difference of squared neutrino masses :footcite:p:`Lesgourgues12`,
    :math:`{\rm \Delta}m^2_{\rm 31} := m^2_{\rm 3}-m^2_{\rm 1}`, in
    :math:`10^{-3} \rm eV^2`.  This difference is in the normal mass hierarchy.

    .. footbibliography::
    """

    DELTAM13_sq_neg: float = -2.43e-3
    r"""Difference of squared neutrino masses :footcite:p:`Lesgourgues12`,
    :math:`{\rm \Delta}m^2_{\rm 31} := m^2_{\rm 3}-m^2_{\rm 1}`, in
    :math:`10^{-3} \rm eV^2`. This difference is in the inverted mass
    hierarchy.

    .. footbibliography::
    """

    T_CMB: float = 2.725  # TODO: Remove for CCLv3.
    r"""Temperature of the CMB, :math:`T_{\rm CMB}`, in :math:`\rm K`.

    .. deprecated:: 2.8.0

        This parameter has been moved to `Cosmology` and will be removed in the
        next major release.
    """

    T_NCDM: float = 0.71611  # TODO> Remove for CCLv3.
    r"""Non-CDM temperature in units of photon temperature,
    :math:`T_{\rm nCDM}`

    .. deprecated:: 2.8.0

        This parameter has been moved to `Cosmology` and will be removed in the
        next major release.
    """


physical_constants = PhysicalConstants()
# physical_constants.freeze()
