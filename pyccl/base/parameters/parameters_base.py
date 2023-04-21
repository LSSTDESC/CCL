__all__ = ("Parameters",)

import warnings
from functools import cached_property

from ... import CCLDeprecationWarning
from .. import ObjectLock


class Parameters:
    """Base for classes holding global CCL parameters and their values.

    Subclasses are coupled to the C-struct with the collection of parameters
    they represent (and their values), via SWIG.
    """
    _allowed_keys = ("_object_lock", "_instance", "_index",)

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
        cls._freeze = freeze

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

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)          # create a new object
        obj._object_lock = ObjectLock()     # assign a lock to it
        if cls._factory:
            obj._instance = cls._factory()  # instantiate C struct
        if cls._freeze:
            obj._object_lock.lock()         # lock it if needed
        return obj

    def __init__(self):
        if self._factory:
            return self.set_parameter_names()
        self.reload()

    @cached_property
    def _parameters(self):
        return [par for par in dir(self) if self._is_parameter(par)]

    def _is_parameter(self, par) -> bool:
        # Check if `par` is a parameter: exclude private and callables.
        return not (par.startswith(("_", "this"))
                    or callable(getattr(self, par, None)))

    def __setattr__(self, key, value):
        if key not in self._allowed_keys:
            if self._object_lock.locked:
                name = self.__class__.__name__
                raise AttributeError(f"Instances of {name} are frozen.")
            if key not in self:
                raise AttributeError(f"Parameter {key} does not exist.")

        object.__setattr__(self, key, value)  # update Python
        if self._is_parameter(key):
            object.__setattr__(self._instance, key, value)  # update C

    def __getitem__(self, key):
        return getattr(self, key)

    __setitem__ = __setattr__

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        for par in self:
            if getattr(self, par) != getattr(other, par):
                return False
        return True

    def __repr__(self):
        name = self.__class__.__name__
        pars = {par: self[par] for par in self}
        return f"<{name}>\n\t" + "\n\t".join(repr(pars).split(","))

    def __contains__(self, key):
        return key in self._parameters

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        self._index += 1
        return self._parameters[self._index-1]

    def freeze(self):
        """Freeze the parameters to make them immutable."""
        self._object_lock.lock()

    def unfreeze(self):
        """Unfreeze the parameters to mutate them."""
        self._object_lock.unlock()

    def copy(self):
        """Create a copy of the parameters."""
        out = type(self)()
        out.reload(source=self)
        return out

    def reload(self, source=None):
        """Reload the original values of the parameters."""
        source = self.__class__ if source is None else source
        for par in self:
            value = getattr(source, par)
            object.__setattr__(self, par, value)
            object.__setattr__(self._instance, par, value)

    def set_parameter_names(self):
        """Set the parameter names from the C library."""
        for par in dir(self._instance):
            if self._is_parameter(par):
                object.__setattr__(self, par, getattr(self._instance, par))
