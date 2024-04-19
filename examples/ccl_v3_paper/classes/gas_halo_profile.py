from .mass_fraction import MassFraction
import numpy as np
from .presets import Presets
import pyccl as ccl


class GasHaloProfile(ccl.halos.HaloProfile):
    """
    Class to calculate the gas profile of a halo using the formula:
    ρ_g(r) = ρ_g,0 / [(1 + u) ^ β * (1 + v^2) ^ ((7 - β) / 2)], where u = r / r_co and v = r / r_ej.
    For more information, see Fedeli (2014), arXiv:1401.2997.
    Methods
    -------
    _rs(cosmo, M, a)
        Calculate the scaled radius of the halo.
    _rint(Rd, r)
        Calculate the radial integrand used for the normalization of the gas profile.
    _norm(M, Rd)
        Calculate the normalization constant for the gas profile density.
    _real(cosmo, r, M, a)
        Calculate the gas profile density at given radii for specified halo masses.
    Attributes
    ----------
    cosmology : Cosmology object
        Cosmology object from pyCCL used for calculations.
    scale_factor : float
        Scale factor at which to evaluate the profile.
    halo_mass_definition : MassDef object
        Mass definition used for the halo profile calculations.
    parameters : dict
        Dictionary containing the parameters for the gas profile calculation.
    mass_frac : MassFraction object
        MassFraction object used for gas mass fraction calculations.
    """

    def __init__(self, presets):
        """
        Initialize the GasHaloProfile class with the provided presets.
        Parameters
        ----------
        presets : Presets instance
            Presets object containing the cosmology and other parameters.
        """
        if not isinstance(presets, Presets):
            raise TypeError("Expected a Presets instance.")
        super().__init__(mass_def=presets.halo_mass_definition)
        # Set the instance attributes based on the provided presets
        self.cosmology = presets.cosmology
        self.scale_factor = presets.scale_factor
        self.halo_mass_definition = presets.halo_mass_definition
        self.parameters = presets.parameters
        self.mass_frac = MassFraction(presets)

    def _rs(self, cosmo=None, M=None, a=None):
        """
        Calculate the scaled radius of the halo, defaulting to instance attributes if
        certain parameters are not provided.

        Parameters:
        cosmo : Cosmology object, optional
            The cosmology used to calculate the radius. Defaults to self.cosmology if None.
        M : float or array_like
            Mass of the halo. This parameter is required.
        a : float, optional
            Scale factor. Defaults to self.scale_factor if None.

        Returns:
        float or array_like
            The scaled radius of the halo.
        """
        if cosmo is None:
            cosmo = self.cosmology
        if a is None:
            a = self.scale_factor
        if M is None:
            raise ValueError("Mass 'M' must be provided for radius calculation.")

        radius = self.halo_mass_definition.get_radius(cosmo, M, a) / a
        return radius

    def _rint(self, Rd, r):
        """
        Calculate the radial integrand used for the normalization of the gas profile, based on the formula:
        ρ_g(r) = ρ_g,0 / [(1 + u) ^ β * (1 + v^2) ^ ((7 - β) / 2)],
        where u = r / r_co and v = r / r_ej.

        Parameters:
        Rd : float
            Characteristic radius of the halo, typically derived from halo properties.
        r : float or array_like
            Radial distance at which to evaluate the integrand.

        Returns:
        float or array_like
            Value of the integrand at radius r.
        """
        beta = self.parameters['gas_profile']['beta']
        r_co = 0.1 * Rd  # Define core radius as 10% of the characteristic radius
        r_ej = 4.5 * Rd  # Define envelope radius as 450% of the characteristic radius

        # Define u and v according to the provided expressions
        u = r / r_co
        v = r / r_ej

        # Calculate the integrand for the gas profile using the defined variables
        radial_distance = r ** 2 / ((1 + u) ** beta * (1 + v ** 2) ** ((7 - beta) / 2))
        return radial_distance

    def _norm(self, M, Rd):
        """
        Calculate the normalization constant for the gas profile density using trapezoidal integration.

        Parameters:
        M : float or array_like
            Mass of the halo.
        Rd : float or array_like
            Characteristic radii of the halos.

        Returns:
        numpy.ndarray
            Normalization constants for each halo mass.
        """
        f_gas = self.mass_frac.gas_mass_fraction(M)
        r_grid = np.linspace(1E-3, 50, 500)  # Create a radial grid for integration
        r_integrands = np.array([self._rint(rd, r_grid) for rd in Rd])  # Compute integrands for each Rd

        # Perform trapezoidal integration over the radial grid for each characteristic radius
        integrals = np.trapz(r_integrands, r_grid, axis=1)

        norm = f_gas * M / (4 * np.pi * integrals)
        return norm

    def _real(self, cosmo=None, r=None, M=None, a=None):
        """
        Calculate the gas profile density at given radii for specified halo masses.

        Parameters:
        cosmo : Cosmology object, optional
            The cosmology used for calculations. Defaults to self.cosmology if None.
        r : float or array_like, optional
            Radial distances to evaluate the profile. Must be provided.
        M : float or array_like, optional
            Mass of the halo. Must be provided.
        a : float, optional
            Scale factor. Defaults to self.scale_factor if None.

        Returns:
        numpy.ndarray
            Gas profile densities for each (r, M) combination.
        """
        if cosmo is None:
            cosmo = self.cosmology
        if a is None:
            a = self.scale_factor
        if r is None or M is None:
            raise ValueError("Both radial distances 'r' and masses 'M' must be provided.")

        r_use = np.atleast_1d(r)
        M_use = np.atleast_1d(M)

        r_delta = self._rs(cosmo, M_use, a)
        r_co = 0.1 * r_delta
        r_ej = 4.5 * r_delta

        beta = self.parameters['gas_profile']['beta']
        norm = self._norm(M_use, r_delta)

        # Define u and v for readability
        u = r_use[None, :] / r_co[:, None]
        v = r_use[None, :] / r_ej[:, None]

        # Compute the profile using defined u and v
        prof = norm[:, None] / ((1 + u) ** beta * (1 + v ** 2) ** ((7 - beta) / 2))

        # Reshape output to match the dimensions of the inputs
        if np.isscalar(r):
            prof = np.squeeze(prof, axis=-1)
        if np.isscalar(M):
            prof = np.squeeze(prof, axis=0)
        return prof
