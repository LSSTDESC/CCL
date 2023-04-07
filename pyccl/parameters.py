import warnings
from . import ccllib as lib
from .errors import CCLDeprecationWarning


__all__ = ("CCLParameters", "SplineParams", "spline_params", "GSLParams",
           "gsl_params", "PhysicalConstants", "physical_constants",
           "CosmologyParams",)


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


class CosmologyParams(CCLParameters, factory=lib.parameters):
    """Instances of this class hold cosmological parameters."""

    def __getattribute__(self, key):
        if key == "m_nu":
            N_nu_mass = self._instance.N_nu_mass
            nu_masses = lib.parameters_get_nu_masses(self._instance, N_nu_mass)
            return nu_masses.tolist()
        return super().__getattribute__(key)

    def __setattr__(self, key, value):
        if key == "m_nu":
            return lib.parameters_m_nu_set_custom(self._instance, value)
        if key == "mgrowth":
            return lib.parameters_mgrowth_set_custom(self._instance, *value)
        super().__setattr__(key, value)


spline_params = SplineParams()
gsl_params = GSLParams()
physical_constants = PhysicalConstants()


class FFTLogParams:
    """Objects of this class store the FFTLog accuracy parameters."""
    padding_lo_fftlog = 0.1   # | Anti-aliasing: multiply the lower boundary.
    padding_hi_fftlog = 10.   # |                multiply the upper boundary.

    n_per_decade = 100        # Samples per decade for the Hankel transforms.
    extrapol = "linx_liny"     # Extrapolation type.

    padding_lo_extra = 0.1    # Padding for the intermediate step of a double
    padding_hi_extra = 10.    # transform. Doesn't have to be as precise.
    large_padding_2D = False  # If True, high precision intermediate transform.

    plaw_fourier = -1.5       # Real <--> Fourier transforms.
    plaw_projected = -1.0     # 2D projected & cumulative density profiles.

    @property
    def params(self):
        return ["padding_lo_fftlog", "padding_hi_fftlog", "n_per_decade",
                "extrapol", "padding_lo_extra", "padding_hi_extra",
                "large_padding_2D", "plaw_fourier", "plaw_projected"]

    def to_dict(self):
        return {param: getattr(self, param) for param in self.params}

    def __getitem__(self, name):
        return getattr(self, name)

    def __setattr__(self, name, value):
        raise AttributeError("FFTLogParams can only be updated via "
                             "`updated_parameters`.")

    def __repr__(self):
        return repr(self.to_dict())

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.to_dict() == other.to_dict()

    def update_parameters(self, **kwargs):
        """Update the precision of FFTLog for the Hankel transforms.

        Arguments
        ---------
        padding_lo_fftlog, padding_hi_fftlog : float
            Multiply the lower and upper boundary of the input range
            to avoid aliasing. The defaults are 0.1 and 10.0, respectively.
        n_per_decade : float
            Samples per decade for the Hankel transforms.
            The default is 100.
        extrapol : {'linx_liny', 'linx_logy'}
            Extrapolation type when FFTLog has narrower output support.
            The default is 'linx_liny'.
        padding_lo_extra, padding_hi_extra : float
            Padding for the intermediate step of a double Hankel transform.
            Used to compute the 2D projected profile and the 2D cumulative
            density, where the first transform goes from 3D real space to
            Fourier, then from Fourier to 2D real space. Usually, it doesn't
            have to be as precise as ``padding_xx_fftlog``.
            The defaults are 0.1 and 10.0, respectively.
        large_padding_2D : bool
            Override ``padding_xx_extra`` in the intermediate transform,
            and use ``padding_xx_fftlog``. The default is False.
        plaw_fourier, plaw_projected : float
            FFTLog pre-whitens its arguments (makes them flatter) to avoid
            aliasing. The ``plaw`` parameters describe the tilt of the profile,
            :math:`P(r) \\sim r^{\\mathrm{tilt}}`, between real and Fourier
            transforms, and between 2D projected and cumulative density,
            respectively. Subclasses of ``HaloProfile`` may obtain finer
            control via ``_get_plaw_[fourier | projected]``, and some level of
            experimentation with these parameters is recommended.
            The defaults are -1.5 and -1.0, respectively.
        """
        for name, value in kwargs.items():
            if name not in self.params:
                raise AttributeError(f"Parameter {name} does not exist.")
            object.__setattr__(self, name, value)
