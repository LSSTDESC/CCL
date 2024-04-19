from .mass_fraction import MassFraction
import numpy as np
from .presets import Presets
import pyccl as ccl


class StellarHaloProfile(ccl.halos.HaloProfile):
    """
    Class to compute the stellar profile of a halo. Follows  Fedeli (2014), arXiv:1401.2997.
    We use $\\rho_*(x|m)=\\frac{\rho_t}{x}\\exp(-x^\\alpha)$, with $x:=r/r_t$.
    The truncation radius is $r_t=r_s/x_\\delta$,
    where $r_s$ is the scale radius and $x_\\delta$ is a parameter.
    Methods
    -------
    _rs : float or array_like
        Compute the scale radius of the stellar profile.
    _real : float or array_like
        Compute the real-space stellar profile.
    Attributes
    ----------
    cosmology : object
        Cosmology (pyccl object)
    k_array : array_like
        Array of wavenumbers in units of h/Mpc
    scale_factor : float
        Scale factor
    halo_mass_definition : object
        Halo mass definition (pyccl object)
    densities : dict
        Densities of the ingredients relevant for the halo model
    parameters : dict
        Parameters for the stellar profile
    mass_ranges : dict
        Mass ranges for the ingredients relevant for the halo model
    halo_model_quantities : dict
        Quantities relevant for the halo model
    mass_frac : object
        MassFraction instance
    """

    def __init__(self, presets):
        if not isinstance(presets, Presets):
            raise TypeError("Expected a Presets instance.")

        # Call the constructor of the superclass with the default mass definition
        super().__init__(mass_def=presets.halo_mass_definition)

        # Initialize object attributes based on the input stream
        self.cosmology = presets.cosmology
        self.k_array = presets.k_array
        self.scale_factor = presets.scale_factor
        self.halo_mass_definition = presets.halo_mass_definition
        self.densities = presets.densities
        self.parameters = presets.parameters
        self.mass_ranges = presets.mass_ranges
        self.halo_model_quantities = presets.halo_model_quantities
        self.mass_frac = MassFraction(presets)

    def _rs(self, cosmo, M, a):
        """
        Compute the scale radius of the stellar profile.
        Parameters
        ----------
            cosmo (object): Cosmology (pyccl object)
            M (float or array_like): Halo mass in units of Msun
            a (float): Scale factor

        Returns
        -------
            float or array_like: Scale radius in units of Mpc/h
        """
        # Generate 1D array by default
        if a is None:
            a = self.scale_factor
        if cosmo is None:
            cosmo = self.cosmology
        # Compute scale radius
        radius = self.halo_mass_definition.get_radius(cosmo, M, a)
        return radius / a

    def _real(self, cosmo, r, M, a):
        """
        Compute the real-space stellar profile.
        Parameters
        ----------
            cosmo (object): Cosmology (pyccl object)
            r (float or array_like): Radius in units of Mpc/h
            M (float or array_like): Halo mass in units of Msun
            a (float): Scale factor

        Returns
        -------
            float or array_like: Real-space profile in units of Msun/Mpc^3/h
        """
        # Generate 1D array by default
        if a is None:
            a = self.scale_factor
        if cosmo is None:
            cosmo = self.cosmology

        # Load parameters
        x_delta = self.parameters['stellar_profile']['x_delta']

        # Generate 2D array by default
        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        # Compute profile
        r_delta = self._rs(cosmo, M_use, a)  # Scale radius
        r_t = r_delta / x_delta  # Truncation radius
        x = r_use / r_t[:, None]  # Dimensionless radius
        rho_t = M_use * self.mass_frac.stellar_mass_fraction(M_use) / (4 * np.pi * r_t ** 3)  # Truncation density
        prof = rho_t[:, None] * np.exp(-x) / x  # Profile

        # Return scalar if input was scalar
        if np.ndim(r) == 0:
            prof = np.squeeze(prof, axis=-1)
        if np.ndim(M) == 0:
            prof = np.squeeze(prof, axis=0)

        return prof
