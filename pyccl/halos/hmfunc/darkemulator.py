__all__ = ("MassFuncDarkEmulator",)

import numpy as np

from dark_emulator import darkemu
from . import MassFunc


class MassFuncDarkEmulator(MassFunc):
    """Implements mass function described in 2019ApJ...884..29P.
    This parametrization is only valid for '200m' masses.

    Args:
        mass_def (:class:`~pyccl.halos.massdef.MassDef` or :obj:`str`):
            a mass definition object, or a name string.
        mass_def_strict (:obj:`bool`): if ``False``, consistency of the mass
            definition will be ignored.
    """

    name = "DarkEmulator"

    def __init__(self, *, mass_def="200m", mass_def_strict=True):
        super().__init__(mass_def=mass_def, mass_def_strict=mass_def_strict)
        self.emu = darkemu.de_interface.base_class()

    def _check_mass_def_strict(self, mass_def):
        if isinstance(mass_def.Delta, str):
            return True
        elif int(mass_def.Delta) == 200:
            if mass_def.rho_type != "matter":
                return True
        return False

    def _setup(self):
        pass

    def _get_fsigma(self, cosmo, sigM, a, lnM):
        z = 1.0 / a - 1

        Omega_c = cosmo["Omega_c"]
        Omega_b = cosmo["Omega_b"]
        h = cosmo["h"]
        n_s = cosmo["n_s"]
        A_s = cosmo["A_s"]

        omega_c = Omega_c * h**2
        omega_b = Omega_b * h**2
        omega_nu = 0.00064
        Omega_L = 1 - ((omega_c + omega_b + omega_nu) / h**2)

        # Parameters cparam (numpy array) : Cosmological parameters
        # (𝜔𝑏, 𝜔𝑐, Ω𝑑𝑒, ln(10^10 𝐴𝑠), 𝑛𝑠, 𝑤)
        cparam = np.array(
            [omega_b, omega_c, Omega_L, np.log(10**10 * A_s), n_s, -1.0]
        )
        self.emu.set_cosmology(cparam)

        alpha = 10 ** (-((0.75 / (np.log10(200 / 75.0))) ** 1.2))

        pA = self.emu.massfunc.coeff_Anorm_spl(-z)
        pa = self.emu.massfunc.coeff_a_spl(-z)
        pb = 2.57 * pa**alpha
        pc = 1.19

        return pA * ((pb / sigM) ** pa + 1.0) * np.exp(-pc / sigM**2)
