from .gas_halo_profile import GasHaloProfile
from .mass_fraction import MassFraction
from .stellar_halo_profile import StellarHaloProfile
from .power_spectra_calculator import PowerSpectraCalculator
from .presets import Presets
from .profile_interpolation import ProfileInterpolation


class BaryonHaloModel:
    """
    Class to encapsulate the baryon halo model calculations.
    Methods are used to calculate the gas halo profile, the mass fraction, the stellar halo profile,
    the power spectra, and the profile interpolation.
    Parameters
    ----------
    cosmology : object
        A pyCCL Cosmology object.
    k_array : array_like
        Array of wavenumbers.
    scale_factor : float
        Scale factor.
    halo_mass_definition : ccl.halos.MassDef, optional.
    """

    def __init__(self, cosmology, k_array, scale_factor, halo_mass_definition=None):
        # Initialize the Presets with cosmology settings and optional halo mass definition
        self.presets = Presets(cosmology, k_array, scale_factor, halo_mass_definition)

        # Initialize other classes that depend on the presets
        self.gas_halo_profile = GasHaloProfile(self.presets)
        self.mass_fraction = MassFraction(self.presets)
        self.stellar_halo_profile = StellarHaloProfile(self.presets)
        self.power_spectra_calculator = PowerSpectraCalculator(self.presets)
        self.profile_interpolation = ProfileInterpolation(self.presets)

    def __getattr__(self, name):
        for component in (self.gas_halo_profile, self.mass_fraction, self.stellar_halo_profile,
                          self.power_spectra_calculator, self.profile_interpolation, self.presets):
            if hasattr(component, name):
                return getattr(component, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
