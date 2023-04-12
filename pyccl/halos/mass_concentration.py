from ..base import CCLAutoRepr
from .massdef import MassDef, convert_concentration
from .halo_model_base import Concentration


__all__ = ("MassConcentration",)


class MassConcentration(CCLAutoRepr):
    """
    """
    __repr_attrs__ = __eq_attrs__ = ("mass_def", "concentration",)

    def __init__(self, *, mass_def, concentration=None):
        self.mass_def = mass_def
        if (concentration is not None
            and not isinstance(concentration, Concentration)):
            raise TypeError("concentration must be Concentration")
        self.concentration = concentration

    @classmethod
    def from_specs(cls, *, mass_def, concentration=None):
        """
        """
        mass_def = MassDef.create_instance(mass_def)
        if concentration is not None:
            concentration = Concentration.create_instance(concentration,
                                                          mass_def=mass_def)
        return cls(mass_def=mass_def, concentration=concentration)

    @property
    def rho_type(self):
        """Helper to access ``mass_def.rho_type``."""
        return self.mass_def.rho_type

    def get_Delta(self, cosmo, a):
        """Helper to access ``mass_def.get_Delta``."""
        return self.mass_def.get_Delta(cosmo, a)

    def get_radius(self, cosmo, M, a):
        """Helper to access ``mass_def.get_radius``."""
        return self.mass_def.get_radius(cosmo, M, a)

    def get_mass(self, cosmo, R, a):
        """Helper to access ``mass_def.get_mass.``"""
        return self.mass_def.get_mass(cosmo, R, a)

    def get_concentration(self, cosmo, M, a):
        """Helper to call ``concentration``."""
        return self.concentration(cosmo, M, a)

    def get_comoving_virial_radius(self, cosmo, M, a):
        """Compute the comoving virial radius given the mass."""
        R = self.get_radius(cosmo, M, a) / a
        c = self.get_concentration(cosmo, M, a)
        return R/c

    def mass_translator(self, other):
        """Translate between different mass definitions, using the internal
        concentration and assuming an NFW profile.
        """

        def translate(cosmo, M, a):
            if self == other:
                return M

            c_this = self.get_concentration(cosmo, M, a)
            Om_this = cosmo.omega_x(a, self.rho_type)
            D_this = self.get_Delta(cosmo, a) * Om_this
            R_this = self.get_radius(cosmo, M, a)

            Om_other = cosmo.omega_x(a, other.rho_type)
            D_other = other.get_Delta(cosmo, a) * Om_other
            c_other = convert_concentration(
                cosmo, c_old=c_this, Delta_old=D_this, Delta_new=D_other)
            R_other = R_this * c_other/c_this
            return other.get_mass(cosmo, R_other, a)

        return translate
