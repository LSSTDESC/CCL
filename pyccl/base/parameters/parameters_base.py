"""Base class for all kinds of parameters handled by CCL."""

__all__ = (
    "CCLParameters", "DefaultParams", "SplineParams", "GSLParams",
    "PhysicalConstants", "spline_params", "gsl_params", "physical_constants",
    "DEFAULT_POWER_SPECTRUM",)

import warnings

from ... import CCLDeprecationWarning, lib

DEFAULT_POWER_SPECTRUM = "delta_matter:delta_matter"
"""Name of the default power spectrum."""


class DefaultParams:
    """Default cosmological parameters used throughout the library."""
    T_CMB = 2.725
    T_ncdm = 0.71611

    warnings.warn(
        "The default CMB temperaHaloProfileCIBture (T_CMB) will change in "
        "CCLv3.0.0, from 2.725 to 2.7255 (Kelvin).", CCLDeprecationWarning)


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
        instance : :mod:`~pyccl.ccllib`
            Reference to the instance where the parameters are implemented.
        factory : :mod:`~pyccl.ccllib`
            The SWIG factory class where the parameters are stored.
        freeze : bool
            Disable parameter mutation.
        """
        super().__init_subclass__()
        if not (bool(instance) ^ bool(factory)):  # XNOR
            raise ValueError(
                "Provide either the instance, or an instance factory.")
        cls._instance = instance
        cls._factory = factory
        cls._frozen = freeze

        def _new_setattr(self, key, value):
            # Make instances of the SWIG-level class immutable
            # so that everything is handled through this interface.
            # SWIG only assigns `this` via the low level `_ccllib`;
            # we therefore disable all other direct assignments.
            if key == "this":
                return object.__setattr__(self, key, value)
            name = self.__class__.__name__
            # TODO: Deprecation cycle for fully immutable Cosmology objects.
            # raise AttributeError(f"Direct assignment in {name} not supported.")  # noqa
            warnings.warn(
                f"Direct assignment of {name} is deprecated "
                "and an error will be raised in the next CCL release. "
                f"Set via `pyccl.{name}.{key}` before instantiation.",
                CCLDeprecationWarning)
            object.__setattr__(self, key, value)

        # Replace C-level `__setattr__`.
        class_ = cls._factory if cls._factory else cls._instance.__class__
        class_.__setattr__ = _new_setattr

    def __init__(self):
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
            name (str or :class:`CCLParameters`):
                Name or instance of the parameters to look up.
        """
        pars = eval(name) if isinstance(name, str) else name
        out = {}
        for par in dir(pars):
            if not par.startswith("_") and par not in ["this", "thisown"]:
                out[par] = getattr(pars, par)
        return out


class SplineParams(CCLParameters, instance=lib.cvar.user_spline_params):
    """Instances of this class hold the spline parameters.

    These are module-level variables, used throughout the library.
    """

    def __setattr__(self, key, value):
        if key == "A_SPLINE_MAX" and value != 1.0:
            raise ValueError("A_SPLINE_MAX is fixed to 1.")
        super().__setattr__(key, value)


class GSLParams(CCLParameters, instance=lib.cvar.user_gsl_params):
    """Instances of this class hold the gsl parameters.

    These are module-level variables, used throughout the library.
    """


class PhysicalConstants(CCLParameters, instance=lib.cvar.constants,
                        freeze=True):
    """Instances of this class hold the physical constants.

    These are module-level variables, used throughout the library.
    """


spline_params = SplineParams()
gsl_params = GSLParams()
physical_constants = PhysicalConstants()
