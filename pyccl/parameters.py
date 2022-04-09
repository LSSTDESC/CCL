from . import ccllib as lib


class Singleton(type):
    """Implements a singleton type."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(Singleton, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class CCLParameters:
    """Base for singletons holding global CCL parameters and their values."""

    def __init_subclass__(cls, ctype=None, cinstance=None, **kwargs):
        cls._ctype = ctype
        cls._cinstance = cinstance
        super().__init_subclass__(**kwargs)

    def __init__(self):
        for attribute in dir(self._ctype):
            if (not attribute.startswith("_")
                    and attribute not in ["this", "thisown"]):
                value = getattr(self._cinstance, attribute)
                super.__setattr__(self, attribute, value)

    def __setattr__(self, key, value):
        if not hasattr(self._ctype, key):
            raise KeyError(f"Parameter {key} does not exist.")
        setattr(self._cinstance, key, value)
        super.__setattr__(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return repr(self.__dict__)

    @classmethod
    def from_cosmo(cls, cosmo):
        return {**spline_params.__dict__.copy(), **gsl_params.__dict__.copy()}


class SplineParams(CCLParameters,
                   ctype=lib.spline_params,
                   cinstance=lib.cvar.user_spline_params,
                   metaclass=Singleton):
    """The singleton instance of this class holds the spline parameters."""
    pass


class GSLParams(CCLParameters,
                ctype=lib.gsl_params,
                cinstance=lib.cvar.user_gsl_params,
                metaclass=Singleton):
    """The singleton instance of this class holds the gsl parameters."""
    pass


class PhysicalConstants(CCLParameters,
                        ctype=lib.physical_constants,
                        cinstance=lib.cvar.constants,
                        metaclass=Singleton):
    """The singleton instance of this class holds the physical constants."""
    pass


spline_params = SplineParams()
gsl_params = GSLParams()
physical_constants = PhysicalConstants()
