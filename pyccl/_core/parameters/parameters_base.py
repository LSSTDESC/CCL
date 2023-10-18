__all__ = (
    "CCLParameters", "DefaultParams", "SplineParams", "GSLParams",
    "PhysicalConstants", "spline_params", "gsl_params", "physical_constants",
    "DEFAULT_POWER_SPECTRUM",)

from ... import lib

DEFAULT_POWER_SPECTRUM = "delta_matter:delta_matter"


class DefaultParams:
    """Default cosmological parameters used throughout the library.
    """
    #: Mean CMB temperature in Kelvin.
    T_CMB = 2.7255
    #: Non-CDM temperature in units of ``T_CMB``.
    T_ncdm = 0.71611


class CCLParameters:
    """Base for classes holding global CCL parameters and their values.

    Subclasses contain a reference to the C-struct with the collection
    of parameters (and their values) (via SWIG). All subclasses act as proxies
    to the CCL parameters at the C-level.

    Subclasses automatically store a backup of the initial parameter state
    to enable ad hoc reloading.
    """

    def __init_subclass__(cls, *, instance=None, factory=None, freeze=False):
        """Routine for subclass initialization.

        Parameters
        ----------
        instance : :obj:`pyccl.ccllib`
            Reference to the instance where the parameters are implemented.
        factory : :class:`pyccl.ccllib`
            The SWIG factory class where the parameters are stored.
        freeze : bool
            Disable parameter mutation.
        """
        super().__init_subclass__()
        if (instance, factory) == (None, None):
            raise ValueError(
                "Provide either the instance, or an instance factory.")
        cls._instance = instance
        cls._factory = factory
        cls._frozen = freeze

    def __init__(self):
        # Emulate abstraction so that base class cannot be instantiated.
        if not (hasattr(self, "_instance") or hasattr(self, "_factory")):
            name = type(self).__name__
            raise TypeError(f"Can't instantiate {name} with no set "
                            "`instance` or `factory`.")
        # Create a new instance if a factory is provided.
        if self._factory:
            object.__setattr__(self, "_instance", self._factory())
        # Keep a copy of the default parameters.
        object.__setattr__(self, "_bak", CCLParameters.get_params_dict(self))

    def __getattribute__(self, name):
        get = object.__getattribute__
        try:
            return get(get(self, "_instance"), name)
        except AttributeError:
            return get(self, name)

    def __setattr__(self, key, value):
        if self._frozen:
            name = self.__class__.__name__
            raise AttributeError(f"Instances of {name} are frozen.")
        if not hasattr(self._instance, key):
            raise KeyError(f"Parameter {key} does not exist.")
        object.__setattr__(self._instance, key, value)

    __getitem__ = __getattribute__

    __setitem__ = __setattr__

    def __repr__(self):
        out = self._bak.copy()
        for par in out:
            out[par] = getattr(self, par)
        return repr(out)

    def reload(self):
        """Reload the C-level default CCL parameters."""
        frozen = self._frozen
        if frozen:
            object.__setattr__(self, "_frozen", False)
        for param, value in self._bak.items():
            setattr(self, param, value)
        object.__setattr__(self, "_frozen", frozen)

    def freeze(self):
        """Freeze an instance of ``CCLParameters``."""
        object.__setattr__(self, "_frozen", True)

    def unfreeze(self):
        """Unfreeze an instance of ``CCLParameters``."""
        object.__setattr__(self, "_frozen", False)

    @classmethod
    def get_params_dict(cls, name):
        """Get a dictionary of the current parameters.

        Arguments:
            name (str or :obj:`CCLParameters`):
                Name or instance of the parameters to look up.
        """
        pars = eval(name) if isinstance(name, str) else name
        out = {}
        for par in dir(pars):
            if not par.startswith("_") and par not in ["this", "thisown"]:
                out[par] = getattr(pars, par)
        return out


class SplineParams(CCLParameters, instance=lib.cvar.user_spline_params):
    """Instances of this class hold the spline parameters."""

    def __setattr__(self, key, value):
        if key == "A_SPLINE_MAX" and value != 1.0:
            raise ValueError("A_SPLINE_MAX is fixed to 1.")
        super().__setattr__(key, value)


class GSLParams(CCLParameters, instance=lib.cvar.user_gsl_params):
    """Instances of this class hold the gsl parameters."""


class PhysicalConstants(CCLParameters, instance=lib.cvar.constants,
                        freeze=True):
    """Instances of this class hold the physical constants."""


spline_params = SplineParams()
gsl_params = GSLParams()
physical_constants = PhysicalConstants()
