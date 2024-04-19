import numpy as np
import pyccl as ccl
from scipy import integrate
from .presets import Presets
from .profile_interpolation import ProfileInterpolation


class PowerSpectraCalculator:
    """
    Class to calculate the power spectra for the baryon halo model. This class calculates the auto and cross power
    spectra for the dark matter, stars, and gas components, as well as the total power spectrum.
    We follow Fedeli (2014), arXiv:1401.2997, and use the halo model to calculate the power spectra.

    Methods
    -------
    _integrand_two_point_term(mass, k, profile_1, profile_2)
        Integrand for the two-point term in the halo model calculation.
    integrated_two_point_term(min_mass, max_mass, k, profile
        Calculate the integral of the two-point term in the halo model calculation.
    _integrand_bias_term(mass, k, profile
        Integrand for the bias term in the halo model calculation.
    integrated_bias_term(min_mass, max_mass, k, profile)
        Calculate the integral of the bias term in the halo model calculation.
    calculate_one_halo_term(k, component)
        Calculate the one-halo term in the halo model calculation.
    calculate_two_halo_term(k, component)
        Calculate the two-halo term in the halo model calculation.
    get_gas_diffuse_terms(k)
        Calculate the diffuse and diffuse halo terms for the gas component.
    get_dark_matter_gas_terms(k)
        Calculate the one-halo, two-halo, and diffuse terms for the dark matter-gas cross power spectrum.
    get_dark_matter_stars_terms(k)
        Calculate the one-halo and two-halo terms for the dark matter-stars cross power spectrum.
    get_stars_gas_terms(k)
        Calculate the one-halo, two-halo, and diffuse terms for the stars-gas cross power spectrum.
    stars_terms(k_arr)
        Calculate the one-halo and two-halo terms for the stars component.
    dark_matter_terms(k_arr)
        Calculate the one-halo and two-halo terms for the dark matter component.
    gas_terms(k_arr)
        Calculate the one-halo, two-halo, diffuse, and diffuse halo terms for the gas component.
    dark_matter_gas_terms(k_arr)
        Calculate the one-halo, two-halo, and diffuse terms for the dark matter-gas cross power spectrum.
    dark_matter_stars_terms(k_arr)
        Calculate the one-halo and two-halo terms for the dark matter-stars cross power spectrum.
    stars_gas_terms(k_arr)
        Calculate the one-halo, two-halo, and diffuse terms for the stars-gas cross power spectrum.
    dark_matter_auto_power_spectrum(k_arr)
        Calculate the auto power spectrum for the dark matter component.
    stars_auto_power_spectrum(k_arr)
        Calculate the auto power spectrum for the stars component.
    gas_auto_power_spectrum(k_arr)
        Calculate the auto power spectrum for the gas component.
    dark_matter_gas_cross_power_spectrum(k_arr)
        Calculate the cross power spectrum between the dark matter and gas components.
    dark_matter_stars_cross_power_spectrum(k_arr)
        Calculate the cross power spectrum between the dark matter and stars components.
    stars_gas_cross_power_spectrum(k_arr)
        Calculate the cross power spectrum between the stars and gas components.
    total_auto_power_spectrum(k_arr)
        Calculate the total auto power spectrum for the baryon halo model.
    total_cross_power_spectrum(k_arr)
        Calculate the total cross power spectrum for the baryon halo model.
    total_power_spectrum(k_arr)
        Calculate the total power spectrum for the baryon halo model.
    power_spectra_dict(k_arr, components)
        Calculate the power spectra for the specified components.
    individual_terms_dict(k_arr, components)
        Calculate the individual terms for the specified components.
    prefactor_dict(k_arr)
        Calculate the prefactor for the power spectra calculation.

    Attributes
    ----------
    cosmology : Cosmology object
        Cosmology object containing the cosmological parameters.
    scale_factor : float
        Scale factor at which to calculate the power spectra.
    k_array : array
        Array of wavenumbers at which to calculate the power spectra.
    halo_mass_definition : MassDef object
        Halo mass definition used in the halo model calculation.
    parameters : dict
        Dictionary containing the parameters for the power spectrum calculation.
    halo_mass_func : method
        Method to calculate the halo mass function.
    halo_bias_func : method
        Method to calculate the halo bias function.
    mass_ranges : dict
        Dictionary containing the mass ranges for the dark matter, stars, and gas components.
    densities : dict
        Dictionary containing the densities of the dark matter, stars, and gas components.
    interpolated_profiles : dict
        Dictionary containing the interpolated profiles for the dark matter, stars, and gas components.
    """

    def __init__(self, presets):
        if not isinstance(presets, Presets):
            raise TypeError("Expected a Presets instance.")
        self.cosmology = presets.cosmology
        self.scale_factor = presets.scale_factor
        self.k_array = presets.k_array
        self.halo_mass_definition = presets.halo_mass_definition
        self.parameters = presets.parameters
        self.halo_mass_func = presets.halo_model_quantities['halo_mass_function']
        self.halo_bias_func = presets.halo_model_quantities['halo_bias_function']
        self.mass_ranges = presets.mass_ranges
        self.densities = presets.densities
        self.interpolated_profiles = ProfileInterpolation(presets).interpolated_profiles()

    def _integrand_two_point_term(self, mass, k, profile_1, profile_2):
        """
        Integrand for the two-point term in the halo model calculation.

        Parameters
        -------
            mass (float): Mass of the halo
            k (float): Wavenumber
            profile_1 (method): Interpolated profile for the first component
            profile_2 (method): Interpolated profile for the second component

        Returns
        -------
            float: Value of the integrand
        """
        # Calculate the halo mass function
        hmf = self.halo_mass_func  # Halo mass function instance
        dn_dlog10m = hmf(self.cosmology, mass, self.scale_factor)
        dn_dm = dn_dlog10m / (mass * np.log(10.))
        # Calculate the profiles for the two components
        y1 = profile_1((mass, k))
        y2 = profile_2((mass, k))
        return dn_dm * mass ** 2 * y1 * y2

    def integrated_two_point_term(self, min_mass, max_mass, k, profile_1, profile_2):
        """
        Calculate the integral of the two-point term in the halo model calculation.
        Parameters
        ----------
            min_mass (float): Minimum mass for the integral
            max_mass (float): Maximum mass for the integral
            k (float): Wavenumber
            profile_1 (method): Interpolated profile_1
            profile_2 (method): Interpolated profile_2

        Returns
        -------
            float: Value of the integral
        """
        integral = integrate.quad(self._integrand_two_point_term,
                                  min_mass,
                                  max_mass,
                                  args=(k, profile_1, profile_2),
                                  epsabs=0, epsrel=1E-2, limit=1000)[0]
        return integral

    def _integrand_bias_term(self, mass, k, profile):
        """
        Integrand for the bias term in the halo model calculation.
        Parameters
        ----------
            mass (float): Mass of the halo
            k (float): Wavenumber
            profile (method): Interpolated profile

        Returns
        -------
            float: Value of the integrand
        """
        hmf = self.halo_mass_func  # Halo mass function instance
        hbf = self.halo_bias_func(self.cosmology, mass, self.scale_factor)
        dn_dlog10m = hmf(self.cosmology, mass, self.scale_factor)
        dn_dm = dn_dlog10m / (mass * np.log(10.))
        y = profile((mass, k))
        return dn_dm * mass * hbf * y

    def integrated_bias_term(self, min_mass, max_mass, k, profile):
        """
        Calculate the integral of the bias term in the halo model calculation.
        Parameters
        ----------
            min_mass (float): Minimum mass for the integral
            max_mass (float): Maximum mass for the integral
            k (float): Wavenumber
            profile (method): Interpolated profile
        Returns
        -------
            float: Value of the integral
        """
        integral = integrate.quad(self._integrand_bias_term,
                                  min_mass,
                                  max_mass,
                                  args=(k, profile),
                                  epsabs=0, epsrel=1E-2, limit=1000)[0]

        return integral

    def calculate_one_halo_term(self, k, component):
        """
        Calculate the one-halo term in the halo model calculation.
        Parameters
        ----------
            k (float): Wavenumber
            component (str): Component for which to calculate the one-halo term.
                             Options are: 'dark_matter', 'stars', 'gas'.

        Returns
        -------
            float: Value of the one-halo term
        """

        rho = self.densities[component]  # Density of the component
        min_mass = self.mass_ranges[component]['min']  # Minimum mass for the integral
        max_mass = self.mass_ranges[component]['max']  # Maximum mass for the integral
        fg = self.parameters['power_spectrum_gas']['Fg']  # Gas fraction
        bd = self.parameters['power_spectrum_gas']['bd']  # Bias for the gas component
        profile = self.interpolated_profiles[component]  # Interpolated profile for the component

        # Calculate the inverse density squared
        if component == 'gas':  # Gas component has an additional factor of fg
            inverse_density_squared = 1 / (fg ** 2 * rho ** 2)
        else:
            inverse_density_squared = 1 / (rho ** 2)  # For dark matter and stars
        integrated_two_point = self.integrated_two_point_term(min_mass, max_mass, k, profile, profile)

        # Calculate the one-halo term
        one_halo_term = inverse_density_squared * integrated_two_point

        return one_halo_term

    def calculate_two_halo_term(self, k, component):
        """
        Calculate the two-halo term in the halo model calculation.
        Parameters
        ----------
            k (float): Wavenumber
            component (str): Component for which to calculate the two-halo term.
                             Options are: 'dark_matter', 'stars', 'gas'.

        Returns
        -------
            float: Value of the two-halo term
        """
        rho = self.densities[component]  # Density of the component
        min_mass = self.mass_ranges[component]['min']  # Minimum mass for the integral
        max_mass = self.mass_ranges[component]['max']  # Maximum mass for the integral
        fg = self.parameters['power_spectrum_gas']['Fg']  # Gas fraction
        bd = self.parameters['power_spectrum_gas']['bd']  # Bias for the gas component
        profile = self.interpolated_profiles[component]  # Interpolated profile for the component

        # Calculate the linear matter power spectrum
        lin_pk = ccl.linear_matter_power(self.cosmology, k, self.scale_factor)
        # Calculate the density squared
        if component == 'gas':  # Gas component has an additional factor of fg
            density_squared = fg ** 2 * rho ** 2
        else:
            density_squared = rho ** 2  # For dark matter and stars
        # Calculate the integrated bias term
        integrated_bias = self.integrated_bias_term(min_mass, max_mass, k, profile)
        # Calculate the integrated bias squared
        integrated_bias_squared = integrated_bias ** 2

        # Calculate the two-halo term
        two_halo_term = lin_pk / density_squared * integrated_bias_squared

        return two_halo_term

    def get_gas_diffuse_terms(self, k):
        """
        Calculate the diffuse and diffuse halo terms for the gas component.
        Parameters
        ----------
            k (float): Wavenumber

        Returns
        -------
            tuple: Tuple containing the values of the diffuse and diffuse halo terms. The first element is the
                   diffuse term, and the second element is the diffuse halo term.
        """
        # Set the component to 'gas'
        component = 'gas'
        # Load the necessary parameters
        rho = self.densities[component]  # Density of the component
        fg = self.parameters['power_spectrum_gas']['Fg']  # Additional gas constant
        bd = self.parameters['power_spectrum_gas']['bd']  # Bias for the gas component
        min_mass = self.mass_ranges[component]['min']  # Minimum mass for the integral
        max_mass = self.mass_ranges[component]['max']  # Maximum mass for the integral
        profile = self.interpolated_profiles[component]  # Interpolated profile for the component

        # Calculate the linear matter power spectrum
        lin_pk = ccl.linear_matter_power(self.cosmology, k, self.scale_factor)
        # Calculate the integrated bias term
        integrated_bias = self.integrated_bias_term(min_mass, max_mass, k, profile)

        # Calculate the diffuse term
        diffuse_term = bd ** 2 * lin_pk
        # Calculate the integrated bias squared
        diffuse_halo_term = bd * lin_pk / (fg * rho) * integrated_bias

        return diffuse_term, diffuse_halo_term

    def get_dark_matter_gas_terms(self, k):
        """
        Calculate the one-halo, two-halo, and diffuse terms for the dark matter-gas cross power spectrum.
        Parameters
        ----------
            k (float): Wavenumber

        Returns
        -------
            tuple: Tuple containing the values of the one-halo, two-halo, and diffuse terms.
                   The order is: one-halo term, two-halo term, diffuse term.
        """
        # Set the components to 'dark_matter' and 'gas'
        dm = 'dark_matter'
        gas = 'gas'
        rho_dm = self.densities[dm]  # Density of the dark matter component
        rho_gas = self.densities[gas]  # Density of the gas component
        fg = self.parameters['power_spectrum_gas']['Fg']  # Additional gas constant
        bd = self.parameters['power_spectrum_gas']['bd']  # Bias for the gas component
        min_mass_dm = self.mass_ranges[dm]['min']  # Minimum mass for the dark matter integral
        max_mass_dm = self.mass_ranges[dm]['max']  # Maximum mass for the dark matter integral
        min_mass_gas = self.mass_ranges[gas]['min']  # Minimum mass for the gas integral
        max_mass_gas = self.mass_ranges[gas]['max']  # Maximum mass for the gas integral
        profile_dm = self.interpolated_profiles[dm]  # Interpolated profile for the dark matter component
        profile_gas = self.interpolated_profiles[gas]  # Interpolated profile for the gas component

        # Calculate the linear matter power spectrum
        lin_pk = ccl.linear_matter_power(self.cosmology, k, self.scale_factor)
        # Calculate the prefactor for the dark matter-gas cross power spectrum
        dm_gas_prefactor = fg * rho_dm * rho_gas
        # Calculate the integrated two-point term
        integrated_two_point = self.integrated_two_point_term(min_mass_gas,
                                                              max_mass_gas,
                                                              k,
                                                              profile_dm,
                                                              profile_gas)
        # Calculate the integrated bias terms for the dark matter and gas components
        integrated_bias_dm = self.integrated_bias_term(min_mass_dm, max_mass_dm, k, profile_dm)
        integrated_bias_gas = self.integrated_bias_term(min_mass_gas, max_mass_gas, k, profile_gas)

        # Calculate the one-halo term
        one_halo_term = 1 / dm_gas_prefactor * integrated_two_point
        # Calculate the two-halo term
        two_halo_term = lin_pk / dm_gas_prefactor * integrated_bias_dm * integrated_bias_gas

        # Calculate the diffuse term
        diffuse_term = bd * lin_pk / rho_dm * integrated_bias_dm

        return one_halo_term, two_halo_term, diffuse_term

    def get_dark_matter_stars_terms(self, k):
        """
        Calculate the one-halo and two-halo terms for the dark matter-stars cross power spectrum.
        Parameters
        ----------
            k (float): Wavenumber

        Returns
        -------
            tuple: Tuple containing the values of the one-halo and two-halo terms.
                   The order is: one-halo term, two-halo term.
        """
        # Set the components to 'dark_matter' and 'stars'
        dm = 'dark_matter'
        stars = 'stars'
        rho_dm = self.densities[dm]  # Density of the dark matter component
        rho_stars = self.densities[stars]  # Density of the stars component
        min_mass_stars = self.mass_ranges[stars]['min']  # Minimum mass for the stars integral
        max_mass_stars = self.mass_ranges[stars]['max']  # Maximum mass for the stars integral
        min_mass_dm = self.mass_ranges[dm]['min']  # Minimum mass for the dark matter integral
        max_mass_dm = self.mass_ranges[dm]['max']  # Maximum mass for the dark matter integral
        profile_dm = self.interpolated_profiles[dm]  # Interpolated profile for the dark matter component
        profile_stars = self.interpolated_profiles[stars]  # Interpolated profile for the stars component

        # Calculate the linear matter power spectrum
        lin_pk = ccl.linear_matter_power(self.cosmology, k, self.scale_factor)
        # Calculate the prefactor for the dark matter-stars cross power spectrum
        dm_stars_prefactor = rho_dm * rho_stars
        # Calculate the integrated two-point term
        integrated_two_point = self.integrated_two_point_term(min_mass_stars,
                                                              max_mass_stars,
                                                              k,
                                                              profile_dm,
                                                              profile_stars)
        # Calculate the integrated bias terms for the dark matter and stars components
        integrated_bias_dm = self.integrated_bias_term(min_mass_dm, max_mass_dm, k, profile_dm)
        integrated_bias_stars = self.integrated_bias_term(min_mass_stars, max_mass_stars, k, profile_stars)
        bias_dm_stars = integrated_bias_dm * integrated_bias_stars

        # Calculate the one-halo term
        one_halo_term = 1 / dm_stars_prefactor * integrated_two_point

        # Calculate the two-halo term
        two_halo_term = lin_pk / dm_stars_prefactor * bias_dm_stars

        return one_halo_term, two_halo_term

    def get_stars_gas_terms(self, k):
        """
        Calculate the one-halo, two-halo, and diffuse terms for the stars-gas cross power spectrum.
        Parameters
        ----------
            k (float): Wavenumber

        Returns
        -------
            tuple: Tuple containing the values of the one-halo, two-halo, and diffuse terms.
                   The order is: one-halo term, two-halo term, diffuse term.
        """
        # Set the components to 'stars' and 'gas'
        rho_stars = self.densities['stars']
        rho_gas = self.densities['gas']
        # Load the necessary parameters
        fg = self.parameters['power_spectrum_gas']['Fg']  # Additional gas constant
        bd = self.parameters['power_spectrum_gas']['bd']  # Bias for the gas component
        min_mass_stars = self.mass_ranges['stars']['min']  # Minimum mass for the stars integral
        max_mass_stars = self.mass_ranges['stars']['max']  # Maximum mass for the stars integral
        min_mass_gas = self.mass_ranges['gas']['min']  # Minimum mass for the gas integral
        max_mass_gas = self.mass_ranges['gas']['max']  # Maximum mass for the gas integral
        profile_stars = self.interpolated_profiles['stars']  # Interpolated profile for the stars component
        profile_gas = self.interpolated_profiles['gas']  # Interpolated profile for the gas component

        # Calculate the linear matter power spectrum
        lin_pk = ccl.linear_matter_power(self.cosmology, k, self.scale_factor)
        # Calculate the prefactor for the stars-gas cross power spectrum
        stars_gas_prefactor = fg * rho_stars * rho_gas
        # Calculate the integrated two-point term
        integrated_two_point_gas_stars = self.integrated_two_point_term(min_mass_gas,
                                                                        max_mass_stars,
                                                                        k,
                                                                        profile_gas,
                                                                        profile_stars)
        # Calculate the one-halo terms for the stars and gas components
        integrated_bias_gas = self.integrated_bias_term(min_mass_gas, max_mass_gas, k, profile_gas)
        integrated_bias_stars = self.integrated_bias_term(min_mass_stars, max_mass_stars, k, profile_stars)
        # Calculate the one-halo term
        bias_stars_gas = integrated_bias_gas * integrated_bias_stars

        # Calculate the one-halo term
        one_halo_term = 1 / stars_gas_prefactor * integrated_two_point_gas_stars
        # Calculate the two-halo term
        two_halo_term = lin_pk / stars_gas_prefactor * bias_stars_gas
        # Calculate the diffuse term
        gas_diffuse_term = bd * lin_pk / rho_stars * integrated_bias_stars

        return one_halo_term, two_halo_term, gas_diffuse_term

    def stars_terms(self, k_arr):
        """
        Calculate the one-halo and two-halo terms for the stars component.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            tuple: Tuple containing the values of the one-halo and two-halo terms.
                   The order is: one-halo term, two-halo term.
        """
        # Initialize arrays to store results
        two_halo = np.zeros(len(k_arr))
        one_halo = np.zeros(len(k_arr))

        # Iterate over each k value in the k_arr
        for i in range(len(k_arr)):
            k = k_arr[i]  # Wavenumber
            # Call the wrappers for each scalar k value
            one_halo[i] = self.calculate_one_halo_term(k, 'stars')
            two_halo[i] = self.calculate_two_halo_term(k, 'stars')

        return one_halo, two_halo

    def dark_matter_terms(self, k_arr):
        """
        Calculate the one-halo and two-halo terms for the dark matter component.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            tuple: Tuple containing the values of the one-halo and two-halo terms.
                   The order is: one-halo term, two-halo term.
        """
        # Initialize arrays to store results
        two_halo = np.zeros(len(k_arr))
        one_halo = np.zeros(len(k_arr))

        # Iterate over each k value in the k_arr
        for i in range(len(k_arr)):
            k = k_arr[i]  # Wavenumber
            # Call the wrappers for each scalar k value
            one_halo[i] = self.calculate_one_halo_term(k, 'dark_matter')
            two_halo[i] = self.calculate_two_halo_term(k, 'dark_matter')

        return one_halo, two_halo

    def gas_terms(self, k_arr):
        """
        Calculate the one-halo, two-halo, diffuse, and diffuse halo terms for the gas component.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            tuple: Tuple containing the values of the one-halo, two-halo, diffuse, and diffuse halo terms.
                   The order is: one-halo term, two-halo term, diffuse term, diffuse halo term.
        """
        # Initialize arrays to store results
        two_halo = np.zeros(len(k_arr))
        one_halo = np.zeros(len(k_arr))
        diffuse_term = np.zeros(len(k_arr))
        diffuse_halo_term = np.zeros(len(k_arr))

        # Iterate over each k value in the k_arr
        for i in range(len(k_arr)):
            k = k_arr[i]  # Wavenumber
            # Call the wrappers for each scalar k value
            one_halo[i] = self.calculate_one_halo_term(k, 'gas')
            two_halo[i] = self.calculate_two_halo_term(k, 'gas')
            diffuse_term[i], diffuse_halo_term[i] = self.get_gas_diffuse_terms(k)

        return one_halo, two_halo, diffuse_term, diffuse_halo_term

    def dark_matter_gas_terms(self, k_arr):
        """
        Calculate the one-halo, two-halo, and diffuse terms for the dark matter-gas cross power spectrum.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            tuple: Tuple containing the values of the one-halo, two-halo, and diffuse terms.
                   The order is: one-halo term, two-halo term, diffuse term.
        """
        # Initialize arrays to store results
        one_halo = np.zeros(len(k_arr))
        two_halo = np.zeros(len(k_arr))
        diffuse_term = np.zeros(len(k_arr))

        # Iterate over each k value in the k_arr
        for i in range(len(k_arr)):
            k = k_arr[i]  # Wavenumber
            one_halo[i], two_halo[i], diffuse_term[i] = self.get_dark_matter_gas_terms(k)

        return one_halo, two_halo, diffuse_term

    def dark_matter_stars_terms(self, k_arr):
        """
        Calculate the one-halo and two-halo terms for the dark matter-stars cross power spectrum.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            tuple: Tuple containing the values of the one-halo and two-halo terms.
                   The order is: one-halo term, two-halo term.
        """
        # Initialize arrays to store results
        one_halo = np.zeros(len(k_arr))
        two_halo = np.zeros(len(k_arr))

        # Iterate over each k value in the k_arr
        for i in range(len(k_arr)):
            k = k_arr[i]  # Wavenumber
            one_halo[i], two_halo[i] = self.get_dark_matter_stars_terms(k)

        return one_halo, two_halo

    def stars_gas_terms(self, k_arr):
        """
        Calculate the one-halo, two-halo, and diffuse terms for the stars-gas cross power spectrum.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            tuple: Tuple containing the values of the one-halo, two-halo, and diffuse terms.
                   The order is: one-halo term, two-halo term, diffuse term.
        """
        # Initialize arrays to store results
        one_halo = np.zeros(len(k_arr))
        two_halo = np.zeros(len(k_arr))
        diffuse_term = np.zeros(len(k_arr))

        # Iterate over each k value in the k_arr
        for i in range(len(k_arr)):
            k = k_arr[i]  # Wavenumber
            one_halo[i], two_halo[i], diffuse_term[i] = self.get_stars_gas_terms(k)

        return one_halo, two_halo, diffuse_term

    def dark_matter_auto_power_spectrum(self, k_arr):
        """
        Calculate the auto power spectrum for the dark matter component.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            array: Auto power spectrum for the dark matter component (one-halo + two-halo).
        """
        one_halo, two_halo = self.dark_matter_terms(k_arr)
        return one_halo + two_halo

    def stars_auto_power_spectrum(self, k_arr):
        """
        Calculate the auto power spectrum for the stars component.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            array: Auto power spectrum for the stars component (one-halo + two-halo).
        """
        one_halo, two_halo = self.stars_terms(k_arr)
        return one_halo + two_halo

    def gas_auto_power_spectrum(self, k_arr):
        """
        Calculate the auto power spectrum for the gas component.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            array: Auto power spectrum for the gas component.
        """
        fg = self.parameters['power_spectrum_gas']['Fg']  # Gas additional constant
        # Define the prefactors
        prefactor_1 = fg ** 2
        prefactor_2 = (1 - fg) ** 2
        prefactor_3 = 2 * fg * (1 - fg)
        # Load the terms
        one_halo, two_halo, diffuse_term, diffuse_halo_term = self.gas_terms(k_arr)
        # Calculate the auto power spectrum
        pk_gas = prefactor_1 * (one_halo + two_halo) + prefactor_2 * diffuse_term + prefactor_3 * diffuse_halo_term

        return pk_gas

    def dark_matter_gas_cross_power_spectrum(self, k_arr):
        """
        Calculate the cross power spectrum for the dark matter-gas components.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            array: Cross power spectrum for the dark matter-gas components.
        """
        fg = self.parameters['power_spectrum_gas']['Fg']  # Gas additional constant
        # Load the terms
        one_halo, two_halo, diffuse_term = self.dark_matter_gas_terms(k_arr)

        # Calculate the cross power spectrum
        pk_dm_gas = (1 - fg) * diffuse_term + fg * (one_halo + two_halo)

        return pk_dm_gas

    def dark_matter_stars_cross_power_spectrum(self, k_arr):
        """
        Calculate the cross power spectrum for the dark matter-stars components.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            array: Cross power spectrum for the dark matter-stars components.
        """
        # Load the terms
        one_halo, two_halo = self.dark_matter_stars_terms(k_arr)

        # Calculate the cross power spectrum
        pk_dm_stars = one_halo + two_halo

        return pk_dm_stars

    def stars_gas_cross_power_spectrum(self, k_arr):
        """
        Calculate the cross power spectrum for the stars-gas components.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            array: Cross power spectrum for the stars-gas components.
        """
        fg = self.parameters['power_spectrum_gas']['Fg']  # Gas additional constant

        # Load the terms
        one_halo, two_halo, diffuse_term = self.stars_gas_terms(k_arr)

        # Calculate the cross power spectrum
        pk_stars_gas = (1 - fg) * diffuse_term + fg * (one_halo + two_halo)

        return pk_stars_gas

    def total_auto_power_spectrum(self, k_arr):
        """
        Calculate the total auto power spectrum for the baryon halo model.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            array: Total auto power spectrum for the baryon halo model.
                   It consists of the auto power spectra for the dark matter, stars, and gas components.
        """
        # Load the densities
        rho_dm = self.densities['dark_matter']  # Dark matter density
        rho_star = self.densities['stars']  # Stars density
        rho_g = self.densities['gas']  # Gas density
        rho_m = self.densities['matter']  # Matter density

        # Load the auto power spectra for each component
        dm_auto_pk = self.dark_matter_auto_power_spectrum(k_arr)  # Dark matter auto power spectrum
        star_auto_pk = self.stars_auto_power_spectrum(k_arr)  # Stars auto power spectrum
        gas_auto_pk = self.gas_auto_power_spectrum(k_arr)  # Gas auto power spectrum

        # Calculate the prefactors
        dm_prefactor = (rho_dm / rho_m) ** 2
        star_prefactor = (rho_star / rho_m) ** 2
        gas_prefactor = (rho_g / rho_m) ** 2

        # Calculate the total auto power spectrum
        total_auto_pk = dm_prefactor * dm_auto_pk + star_prefactor * star_auto_pk + gas_prefactor * gas_auto_pk

        return total_auto_pk

    def total_cross_power_spectrum(self, k_arr):
        """
        Calculate the total cross power spectrum for the baryon halo model.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            array: Total cross power spectrum for the baryon halo model.
                   It consists of the cross power spectra for the dark matter-gas,
                   dark matter-stars, and stars-gas components.
        """
        # Load the densities
        rho_dm = self.densities['dark_matter']  # Dark matter density
        rho_star = self.densities['stars']  # Stars density
        rho_g = self.densities['gas']  # Gas density
        rho_m = self.densities['matter']  # Matter density

        # Load the cross power spectra for each component
        dm_star_cross_pk = self.dark_matter_stars_cross_power_spectrum(k_arr)
        dm_gas_cross_pk = self.dark_matter_gas_cross_power_spectrum(k_arr)
        star_gas_cross_pk = self.stars_gas_cross_power_spectrum(k_arr)

        # Calculate the total cross power spectrum
        addend_1 = 2 * rho_dm * rho_star * dm_star_cross_pk
        addend_2 = 2 * rho_g * rho_star * star_gas_cross_pk
        addend_3 = 2 * rho_dm * rho_g * dm_gas_cross_pk
        numerator = addend_1 + addend_2 + addend_3
        denominator = rho_m ** 2

        # Calculate the total cross power spectrum
        total_cross_pk = numerator / denominator

        return total_cross_pk

    def total_power_spectrum(self, k_arr):
        """
        Calculate the total power spectrum for the baryon halo model.
        It consists of the total auto and cross power spectra for the dark matter, stars, and gas components.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers

        Returns
        -------
            array: Total power spectrum for the baryon halo model.
        """
        # Load the total auto and cross power spectra
        total_auto_pk = self.total_auto_power_spectrum(k_arr)
        total_cross_pk = self.total_cross_power_spectrum(k_arr)

        total_pk = total_auto_pk + total_cross_pk

        return total_pk

    def power_spectra_dict(self, k_arr, component):
        """
        Returns the power spectrum for the specified component.
        Parameters
            k_arr (array): Array of wavenumbers
            component (str): Component for which to calculate the power spectrum. Options are:
                            'dark_matter', 'stars', 'gas', 'dark_matter_gas', 'dark_matter_stars',
                            'stars_gas', 'auto', 'cross', 'total'

        Returns
            array: Power spectrum for the specified component
        """
        # Define the dictionary
        pk_dict = {
            'dark_matter': self.dark_matter_auto_power_spectrum,
            'stars': self.stars_auto_power_spectrum,
            'gas': self.gas_auto_power_spectrum,
            'dark_matter_gas': self.dark_matter_gas_cross_power_spectrum,
            'dark_matter_stars': self.dark_matter_stars_cross_power_spectrum,
            'stars_gas': self.stars_gas_cross_power_spectrum,
            'auto': self.total_auto_power_spectrum,
            'cross': self.total_cross_power_spectrum,
            'total': self.total_power_spectrum
        }

        return pk_dict[component](k_arr)

    def individual_terms_dict(self, k_arr, component):
        """
        Returns the individual terms for the specified component.
        For example, the one-halo and two-halo terms for the dark matter component.
        Parameters
        ----------
            k_arr (array): Array of wavenumbers
            component (str): Component for which to calculate the individual terms. Options are:
                            'dark_matter', 'stars', 'gas',
                            'dark_matter_gas', 'dark_matter_stars', 'stars_gas'.

        Returns
        -------
            tuple: Tuple containing the values of the individual terms (depends on the component).
        """
        # Define the dictionary
        terms_dict = {
            'dark_matter': self.dark_matter_terms,
            'stars': self.stars_terms,
            'gas': self.gas_terms,
            'dark_matter_gas': self.dark_matter_gas_terms,
            'dark_matter_stars': self.dark_matter_stars_terms,
            'stars_gas': self.stars_gas_terms
        }

        return terms_dict[component](k_arr)

    def prefactor_dict(self):
        """
        Calculate the prefactors for the auto and cross power spectra.
        This is useful for plotting the power spectra (the resulting component can be multiplied
        by the corresponding prefactor).
        Returns
        -------
            dict: Dictionary containing the prefactors for the auto and cross power spectra.
        """
        # Load the densities
        rho_dm = self.densities['dark_matter']
        rho_stars = self.densities['stars']
        rho_gas = self.densities['gas']
        rho_m = self.densities['matter']

        # Calculate the prefactors
        prefactor_dict = {
            'dark_matter': (rho_dm / rho_m) ** 2,
            'stars': (rho_stars / rho_m) ** 2,
            'gas': (rho_gas / rho_m) ** 2,
            'dark_matter_gas': 2 * rho_dm * rho_gas / (rho_m ** 2),
            'dark_matter_stars': 2 * rho_dm * rho_stars / (rho_m ** 2),
            'stars_gas': 2 * rho_stars * rho_gas / (rho_m ** 2),
        }

        return prefactor_dict
