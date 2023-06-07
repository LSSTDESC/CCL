"""
====================================================
Parameters Base (:mod:`pyccl._core.parameters_base`)
====================================================

Base class for all kinds of parameters handled by CCL.
"""

from __future__ import annotations

__all__ = ("Parameters",)

from dataclasses import dataclass, fields
from functools import cached_property
from typing import Any, Optional


class Parameters:
    """Base for classes holding parameters and their values.

    Subclasses are data classes. If the stored parameters are coupled with the
    C library, `instance` or `factory` have to be provided,

    Subclasses are data classes and are coupled to the C-struct with the
    parameter collection they represent (and their values), via SWIG.

    Parameters
    ----------
    instance : :mod:`pyccl.ccllib`, optional
        Reference to the instance where the parameters are implemented.
    factory : :mod:`pyccl.ccllib`, optional
        The SWIG factory class where the parameters are stored.
    frozen : bool, optional
        Disable parameter mutation. The default is False.
    """
    _allowed_keys = ("_instance", "_frozen", "_index",)
    _instance: Any = None
    _factory: Any = None
    _frozen: bool = False
    _coupled_with_C: bool

    def __init_subclass__(cls, *, instance=None, factory=None, frozen=False):
        super().__init_subclass__()
        cls._instance = instance
        cls._factory = factory
        cls._coupled_with_C = not (instance, factory) == (None, None)
        if frozen:
            cls.__post_init__ = lambda self: self.freeze()  # noqa

        # Make subclasses data classes.
        cls = dataclass(init=True, eq=True, repr=True, unsafe_hash=True)(cls)

    def __new__(cls, **kwargs):
        obj = super().__new__(cls)  # create a new object
        if cls._factory is not None:
            obj._instance = cls._factory()  # instantiate C struct
        return obj

    @cached_property
    def _parameters(self) -> list:
        return [field.name for field in fields(self)]

    def __setattr__(self, name, value):
        if name not in self._allowed_keys:
            if name not in self:
                raise AttributeError(f"Parameter {name} does not exist.")
            if self._frozen:
                raise AttributeError(f"{type(self).__name__} is frozen.")

        object.__setattr__(self, name, value)  # update Python
        if self._coupled_with_C and name in self:
            object.__setattr__(self._instance, name, value)  # update C

    def __getitem__(self, name):
        return getattr(self, name)

    __setitem__ = __setattr__

    def __contains__(self, name):
        return name in self._parameters

    def __len__(self):
        return len(self._parameters)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        self._index += 1
        return self._parameters[self._index - 1]

    def freeze(self) -> None:
        """Freeze the instance."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the instance."""
        self._frozen = False

    def reload(self, *, source: Optional[Parameters] = None) -> None:
        """Reload the original values.

        Arguments
        ---------
        source
            Where to reload the parameters from. This is an implementation
            detail which enables copying. :meth:`reload` is normally called
            with no arguments.
        """
        source = type(self) if source is None else source
        frozen, self._frozen = self._frozen, False  # unfreeze to reload
        for param in self:
            setattr(self, param, getattr(source, param))
        self._frozen = frozen  # reset original frozen state

    def copy(self) -> Parameters:
        """Create a copy of the instance."""
        out = type(self)()
        out.reload(source=self)
        return out
