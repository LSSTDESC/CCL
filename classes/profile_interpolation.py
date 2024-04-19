from .gas_halo_profile import GasHaloProfile
import numpy as np
from .presets import Presets
from scipy import interpolate
from .stellar_halo_profile import StellarHaloProfile


class ProfileInterpolation:
    """
    Interpolates the Fourier transform of the gas, stellar, and dark matter profiles
    over a grid of masses and wavenumbers.
    Methods
    -------
    update_precision(profile_instance)
        Updates the precision of the profile instance using the FFTLog algorithm.
    interpolate_profile(component, profile_instance)
        Interpolates the Fourier transform of the profile instance over the grid.
    interpolated_profiles(components=None)
        Interpolates specified profiles and returns a dictionary of interpolators.
        If no components are specified, all profiles are interpolated.
    Attributes
    ----------
    presets : Presets
        Instance of the Presets class containing cosmological and halo model settings.
    cosmology : dict
        Dictionary containing cosmological parameters.
    scale_factor : float
        Scale factor at which the profiles are calculated.
    interpolation_grid : dict
        Dictionary containing the grid of masses and wavenumbers for interpolation.
    gas_profile : GasHaloProfile
        Instance of the GasHaloProfile class.
    stellar_profile : StellarHaloProfile
        Instance of the StellarHaloProfile class.
    dark_mater_profile : NFWProfile
        Instance of the NFWProfile class.
    interpolators : dict
        Dictionary containing the interpolators for each profile.
    """

    def __init__(self, presets):
        """
        Initialize the ProfileInterpolation class with the given presets.
        Parameters
        ----------
        presets
            Instance of the Presets class containing cosmological and halo model settings.
        Raises
            TypeError
                If presets is not an instance of the Presets class.
        """
        if not isinstance(presets, Presets):
            raise TypeError("Expected a Presets instance.")

        self.presets = presets
        self.cosmology = presets.cosmology
        self.scale_factor = presets.scale_factor
        self.interpolation_grid = presets.interpolation_grid
        self.gas_profile = GasHaloProfile(presets)
        self.stellar_profile = StellarHaloProfile(presets)
        self.dark_mater_profile = presets.halo_model_quantities['nfw_profile']
        self.interpolators = {}

    def update_precision(self, profile_instance):
        """
        Updates the precision of the profile instance using the FFTLog algorithm.
        A wrapper function for the update_precision_fftlog method in the profile instance.
        Parameters
        ----------
        profile_instance : GasHaloProfile, StellarHaloProfile, or NFWProfile

        Returns
        -------
        profile_instance : GasHaloProfile, StellarHaloProfile, or NFWProfile
            Updated instance of the profile class with the new precision settings.
        """
        profile_instance.update_precision_fftlog(
            padding_hi_fftlog=1E3,
            padding_lo_fftlog=1E-3,
            n_per_decade=1000,
            plaw_fourier=-2.0
        )
        return profile_instance

    def interpolate_profile(self, component, profile_instance):
        """"
        Interpolates the Fourier transform of the profile instance over the grid.
        Parameters
        ----------
        component : str
            Component of the profile to interpolate.
        profile_instance : GasHaloProfile, StellarHaloProfile, or NFWProfile
            Instance of the profile class to interpolate.
        Returns
        -------
        interpolator : RegularGridInterpolator
            Interpolator for the Fourier transform of the profile instance.
        """
        # Get the grid of masses and wavenumbers for interpolation
        grid = self.interpolation_grid[component]
        k_vector = grid['k']  # Wavenumber vector
        mass_vector = grid['mass']  # Mass vector

        # Initialize the array to store Fourier transform results
        if component != 'stars':
            # For stars, the Fourier transform is calculated for each mass and k
            # and then divided by the mass.
            fourier_results = np.zeros((len(mass_vector), len(k_vector)))
            # Iterate over each mass and calculate the Fourier transform for each k
            for j in range(len(mass_vector)):  # Iterate over mass values
                fourier_results[j, :] = profile_instance.fourier(self.cosmology,
                                                                 k_vector,
                                                                 mass_vector[j],
                                                                 self.scale_factor) / mass_vector[j]
        else:
            # Other components have a 2D array of masses and k values.
            M2D = np.tile(mass_vector[:, np.newaxis],
                          (1, len(k_vector)))  # Ensure M2D replicates mass_vector across k_vector
            fourier_results = profile_instance.fourier(self.cosmology,
                                                       k_vector,
                                                       mass_vector,
                                                       self.scale_factor) / M2D

        # Interpolate using a RegularGridInterpolator
        interpolator = interpolate.RegularGridInterpolator((mass_vector, k_vector), fourier_results, method='cubic')
        self.interpolators[component] = interpolator

        return interpolator

    def interpolated_profiles(self, components=None):
        """
        Interpolates specified profiles and returns a dictionary of interpolators.
        If no components are specified, all profiles are interpolated.
        Parameters
        ----------
        components : list of str, optional
            List of components to interpolate. Default is None. If None, all profiles are interpolated.
            Components to choose from are: 'gas', 'stars', 'dark_matter'.
        Returns
        -------
        profiles : dict
            Dictionary containing the interpolators for each component.
        """
        # Define profiles and their corresponding instances
        profiles = {
            'dark_matter': self.dark_mater_profile,
            'gas': self.gas_profile,
            'stars': self.stellar_profile,
        }

        # If components are specified, filter the profiles to interpolate
        if components:
            profiles = {comp: prof for comp, prof in profiles.items() if comp in components}

        # Update precision and interpolate each profile
        for component, profile in profiles.items():
            updated_profile = self.update_precision(profile)
            profiles[component] = self.interpolate_profile(component, updated_profile)

        # Display completion message
        if components:
            component_list = ', '.join(profiles.keys())
            print(f"{component_list.capitalize()} profiles have been interpolated and stored.")
        else:
            print(f"All profiles ({', '.join(profiles.keys())}) have been interpolated and stored.")

        return profiles

    def profiles_dict(self):
        """
        Return the dictionary of interpolators for each profile.
        Returns
        -------
        interpolators : dict
            Dictionary containing the interpolators for each profile.
        """
        return self.interpolated_profiles()
