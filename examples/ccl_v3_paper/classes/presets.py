import cmasher as cmr
import numpy as np
import pyccl as ccl


class Presets:
    """
    Class to define the preset parameters for the baryon halo model.
    Methods are used to define the densities of the ingredients, the parameters of the ingredients,
    the mass ranges for each ingredient, the halo model quantities, and the interpolation grid.
    Parameters
    ----------
    cosmology : dict
        Dictionary of cosmological parameters.
    k_array : array_like
        Array of wavenumbers.
    scale_factor : float
        Scale factor.
    halo_mass_definition : ccl.halos.MassDef, optional.
        If not provided, the default is used.Default is the pyCCl default one.

    Methods
    -------
    define_halo_model_quantities()
        Define the halo model quantities.
    define_ingredient_densities()
        Define the densities of the ingredients.
    define_ingredient_parameters()
        Define the parameters of the ingredients.
    define_mass_ranges()
        Define the mass ranges for each ingredient.
    define_interpolation_grid()
        Define the interpolation grid.
    interpolation_vectors()
        Return the interpolation grid.
    get_colors()
        Get a list of colors from a colormap.
    """

    def __init__(self,
                 cosmology,
                 k_array,
                 scale_factor,
                 halo_mass_definition=None):

        self.cosmology = cosmology
        self.k_array = k_array
        self.scale_factor = scale_factor

        # If no mass_definition is provided, use the default definition
        if halo_mass_definition is None:
            self.halo_mass_definition = ccl.halos.MassDef(200, 'matter')
        else:
            self.halo_mass_definition = halo_mass_definition

        self.densities = self.define_ingredient_densities()
        self.parameters = self.define_ingredient_parameters()
        self.mass_ranges = self.define_mass_ranges()
        self.halo_model_quantities = self.define_halo_model_quantities()
        self.interpolation_grid = self.define_interpolation_grid()

    def define_halo_model_quantities(self):
        """
        Define the halo model quantities, necessary for the halo model calculations.

        Returns
        -------
        halo_dict : dict
            Dictionary containing the halo model quantities.
            Quantities are: halo_mass_function, halo_bias_function, concentration_mass_relation,
            nfw_profile, halo_model_calculator.
        """

        halo_mass_definition = self.halo_mass_definition

        halo_dict = {
            "halo_mass_function": ccl.halos.MassFuncTinker08(mass_def=halo_mass_definition,
                                                             mass_def_strict=False),
            "halo_bias_function": ccl.halos.HaloBiasTinker10(mass_def=halo_mass_definition,
                                                             mass_def_strict=False),
            "concentration_mass_relation": ccl.halos.ConcentrationDuffy08(mass_def=halo_mass_definition),
            "nfw_profile": ccl.halos.HaloProfileNFW(mass_def=halo_mass_definition,
                                                    concentration=ccl.halos.ConcentrationDuffy08(
                                                        mass_def=halo_mass_definition)),
            "halo_model_calculator": ccl.halos.HMCalculator(
                mass_function=ccl.halos.MassFuncTinker08(mass_def=halo_mass_definition, mass_def_strict=False),
                halo_bias=ccl.halos.HaloBiasTinker10(mass_def=halo_mass_definition, mass_def_strict=False),
                mass_def=halo_mass_definition)
        }

        return halo_dict

    def define_ingredient_densities(self):
        """
        Get the densities of the ingredients relevant for the halo model.

        Returns:
            rho_dict: dict containing the densities of the ingredients.
            Ingredients are: star, matter, dark_matter, and gas.
        """
        # Extract parameters from the cosmology
        rho_crit = ccl.physical_constants.RHO_CRITICAL  # Critical density of the universe
        omega_m = self.cosmology['Omega_m']  # Total matter density parameter
        omega_c = self.cosmology['Omega_c']  # Dark matter density parameter
        h = self.cosmology['h']  # Dimensionless Hubble parameter

        # Constants for stellar density calculation, based on some external assumptions or measurements
        rho_star_coefficient = 7E8  # Coefficient for stellar density, specific to this model's calibration

        # Calculate densities for each component
        rho_star = rho_star_coefficient * h ** 2
        rho_matter = omega_m * rho_crit * h ** 2
        rho_dark_matter = omega_c * rho_crit * h ** 2
        # Gas is whatever is left from total matter after accounting for stars and dark matter
        rho_gas = rho_matter - rho_star - rho_dark_matter

        # Create dictionary of densities
        rho_dict = {
            "stars": rho_star,
            "matter": rho_matter,
            "dark_matter": rho_dark_matter,
            "gas": rho_gas
        }

        return rho_dict

    def define_ingredient_parameters(self):
        """"
        Define the parameters for the ingredients in the baryon halo model.
        """

        cosmology = self.cosmology
        h = cosmology['h']

        baryonic_params = {
            "stellar_mass_fraction": {
                "m_0": 5E12 / h,
                "sigma": 1.2
            },
            "stellar_profile": {
                "x_delta": 1. / 0.03
            },
            "gas_mass_fraction": {
                "m_0": 1E12 / h,
                "sigma": 3.
            },
            "gas_profile": {
                "beta": 2.9
            },
            "power_spectrum_gas": {
                "Fg": 0.05,  # Example value; should be << 1
                "bd": 0.85  # From Fedeli
            },
            "reference_halo_mass": {
                "m": 1E14 / h
            }
        }

        return baryonic_params

    def define_mass_ranges(self):

        mass_range_dict = {
            "dark_matter": {
                "min": 1E6,
                "max": 1E16
            },
            "gas": {
                "min": self.parameters["gas_mass_fraction"]["m_0"],
                "max": 1E16
            },
            "stars": {
                "min": 1E10,
                "max": 1E15
            },
            "matter": np.geomspace(1E6, 1E16, 128)  # Based on DM mass range
        }

        return mass_range_dict

    def define_interpolation_grid(self):
        mass = self.mass_ranges
        num_mass = 500
        k_min = -4
        k_max = 2
        num_k = 1500

        k_vector = np.logspace(k_min, k_max, num=num_k)

        def get_mass_vector(mass_type):
            return np.logspace(np.log10(mass[mass_type]['min']),
                               np.log10(mass[mass_type]['max']),
                               num=num_mass)

        interpolation_dict = {
            'dark_matter': {
                'k': k_vector,
                'mass': get_mass_vector('dark_matter')
            },
            'gas': {
                'k': k_vector,
                'mass': get_mass_vector('gas')
            },
            'stars': {
                'k': k_vector,
                'mass': get_mass_vector('stars')
            }
        }
        return interpolation_dict

    def interpolation_vectors(self):
        return self.define_interpolation_grid()

    def get_colors(self, n_colors, cmap='cmr.pride', cmap_range=(0.2, 0.9)):
        """
        Get a list of colors from a colormap uaing cmasher.
        Parameters
        ----------
        n_colors (int): int
            Number of colors to get from the colormap
        cmap (str): str
            Name of the colormap to use. Default is 'cmr.pride'. Use cmasher or matplotlib colormaps.
        cmap_range (tuple): tuple
            Range of the colormap to use. Default is (0.2, 0.9).
        Returns
        -------
        colors (list): list
            List of colors in HEX format.
        """
        # Take n colors from rainforest in [0.15, 0.85] range in HEX
        colors = cmr.take_cmap_colors(cmap, n_colors, cmap_range=cmap_range, return_fmt='hex')
        return colors
