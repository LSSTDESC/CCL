import numpy as np
from .presets import Presets
from scipy import integrate
from scipy.special import erf


class MassFraction:

    def __init__(self, presets):
        if not isinstance(presets, Presets):
            raise TypeError("Expected a Presets instance.")

        self.cosmology = presets.cosmology
        self.scale_factor = presets.scale_factor
        self.halo_mass_definition = presets.halo_mass_definition
        self.parameters = presets.parameters
        self.densities = presets.densities
        self.mass_ranges = presets.mass_ranges
        self.halo_model_quantities = presets.halo_model_quantities

    def gas_mass_fraction(self, mass):
        m_0 = self.parameters['gas_mass_fraction']['m_0']
        sigma = self.parameters['gas_mass_fraction']['sigma']
        omega_b = self.cosmology['Omega_b']
        omega_m = self.cosmology['Omega_m']

        omega_ratio = omega_b / omega_m
        mass_ratio = np.atleast_1d(mass) / m_0
        gas_mass_frac = np.zeros_like(mass_ratio)

        # Apply the original conditional logic
        valid_indices = mass_ratio >= 1  # Only apply erf calculation to valid mass ratios
        gas_mass_frac[valid_indices] = omega_ratio * erf((np.log10(mass_ratio[valid_indices])) / sigma)

        # Return scalar if input was scalar
        if np.isscalar(mass):
            return gas_mass_frac[0]
        return gas_mass_frac

    def stellar_mass_fraction(self, mass):
        # Retrieve constants and necessary data
        rho_star = self.densities['stars']
        m_0 = self.parameters['stellar_mass_fraction']['m_0']
        sigma = self.parameters['stellar_mass_fraction']['sigma']
        mass_min = self.mass_ranges['stars']['min']
        mass_max = self.mass_ranges['stars']['max']

        # Use the halo mass function from the pre-defined quantities in the class
        halo_mass_function = self.halo_model_quantities['halo_mass_function']

        # Define the integrand function using the halo mass function
        def smf_integrand(m):
            mass_dex = (m * np.log(10))
            mf = halo_mass_function(self.cosmology, m, self.scale_factor) / mass_dex
            return m * np.exp(-(np.log10(m / m_0)) ** 2 / (2 * sigma ** 2)) * mf

        # Integration using scipy.integrate.quad for normalization
        integral, error = integrate.quad(smf_integrand, mass_min, mass_max, epsabs=0, epsrel=1E-3, limit=5000)

        # Compute A
        A = rho_star / integral

        # Compute f_star for the provided mass array
        star_mass_frac = A * np.exp(-(np.log10(mass / m_0)) ** 2 / (2 * sigma ** 2))

        return star_mass_frac

    def dark_matter_mass_fraction(self, mass):
        omega_b = self.cosmology['Omega_b']
        omega_m = self.cosmology['Omega_m']

        omega_ratio = omega_b / omega_m
        # My mass frac is 1 - omega_ratio
        mass_ratio = np.atleast_1d(mass)
        dark_matter_frac = np.ones_like(mass_ratio) - omega_ratio

        # Return scalar if input was scalar
        if np.isscalar(mass):
            return dark_matter_frac[0]
        return dark_matter_frac

    def mass_fraction_dict(self, mass):
        """
        Compute the mass fraction of each component for a given mass.
        Parameters
        ----------
            mass : float or array_like
            The mass of the halo

        Returns
        -------
            A dictionary containing the mass fractions of each component.
            Components are: stars, gas, dark_matter.
        """
        mass_frac_dict = {
            "dark_matter": self.dark_matter_mass_fraction(mass),
            "gas": self.gas_mass_fraction(mass),
            "stars": self.stellar_mass_fraction(mass),
        }

        return mass_frac_dict
