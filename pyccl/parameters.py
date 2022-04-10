from . import ccllib as lib


class CCLParameters:
    """Base for singletons holding global CCL parameters and their values.

    Subclasses contain a pointer to the C-struct with the collection
    of parameters and their values (via SWIG), as well as a Python-level
    copy of every parameter and value. These are managed simultaneously
    for the life of the singleton's instance.

    Subclasses automatically store a backup of the initial parameter state
    to enable ad hoc reloading.
    """
    _instances = {}

    def __init_subclass__(cls, ctype=None, cinstance=None,
                          freeze=False, **kwargs):
        """Routine for subclass initialization.

        Parameters:
            ctype (``type``):
                Pointer to the definition of the C-struct. In SWIG,
                this is the class whose instance is a parameter collection.
            cinstance (``instance``):
                The instance where the default parameters are implemented.
                ``cinstance`` should be an instance of ``ctype``.
            freeze (``bool``):
                Disallow mutation of the parameters.
        """
        cls._ctype = ctype
        cls._cinstance = cinstance
        cls._frozen = freeze
        super().__init_subclass__(**kwargs)

    def __new__(cls, *args, **kwargs):
        # Convert all subclasses to singletons.
        if cls not in CCLParameters._instances:
            instance = super().__new__(cls, *args, **kwargs)
            CCLParameters._instances[cls] = instance
        return CCLParameters._instances[cls]

    def __init__(self):
        for attribute in dir(self._ctype):
            if (not attribute.startswith("_")
                    and attribute not in ["this", "thisown"]):
                value = getattr(self._cinstance, attribute)
                super.__setattr__(self, attribute, value)
        self.__class__._params_bak = self.__dict__.copy()

    def __setattr__(self, key, value):
        if self._frozen:
            name = self.__class__.__name__
            raise AttributeError(f"Instances of {name} are frozen.")
        if not hasattr(self._ctype, key):
            raise KeyError(f"Parameter {key} does not exist.")
        setattr(self._cinstance, key, value)
        super.__setattr__(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    __setitem__ = __setattr__

    def __repr__(self):
        return repr(self.__dict__)

    def reload(self):
        """Reload the C-level default CCL parameters."""
        for param, value in self.__class__._params_bak.items():
            setattr(self._cinstance, param, value)
            super.__setattr__(self, param, value)

    @classmethod
    def from_cosmo(cls, cosmo):
        """Return a dictionary of accuracy parameters and their values.

        Arguments:
            cosmo (``pyccl.ccllib.cosmology``):
                Input cosmology via SWIG.
        """
        out = {}
        for param_set in ["spline_params", "gsl_params"]:
            for param in globals()[param_set].__dict__:  # access module vars
                value = getattr(getattr(cosmo, param_set), param)
                out[param] = value if isinstance(value, (int, float)) else None
        return out


class SplineParams(CCLParameters,
                   ctype=lib.spline_params,
                   cinstance=lib.cvar.user_spline_params):
    """The singleton instance of this class holds the spline parameters."""
    pass


class GSLParams(CCLParameters,
                ctype=lib.gsl_params,
                cinstance=lib.cvar.user_gsl_params):
    """The singleton instance of this class holds the gsl parameters."""
    pass


class PhysicalConstants(CCLParameters,
                        ctype=lib.physical_constants,
                        cinstance=lib.cvar.constants,
                        freeze=True):
    """The singleton instance of this class holds the physical constants."""
    pass


spline_params = SplineParams()
gsl_params = GSLParams()
physical_constants = PhysicalConstants()
