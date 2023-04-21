__all__ = ("ObjectLock",
           "UnlockInstance", "unlock_instance",
           "CustomRepr", "CustomEq",
           "CCLObject", "CCLAutoRepr", "CCLNamedClass",)

import functools
from abc import ABC, abstractmethod
from inspect import signature
from _thread import RLock

import numpy as np


class ObjectLock:
    """Control the lock state (immutability) of a ``CCLObject``."""
    _locked: bool = False
    _lock_id: int = None

    def __repr__(self):
        return f"{self.__class__.__name__}(locked={self.locked})"

    @property
    def locked(self):
        """Check if the object is locked."""
        return self._locked

    @property
    def active(self):
        """Check if an unlocking context manager is active."""
        return self._lock_id is not None

    def lock(self):
        """Lock the object."""
        self._locked = True
        self._lock_id = None

    def unlock(self, manager_id=None):
        """Unlock the object."""
        self._locked = False
        if manager_id is not None:
            self._lock_id = manager_id


class UnlockInstance:
    """Context manager that temporarily unlocks an immutable instance
    of ``CCLObject``.

    Parameters:
        instance (``CCLObject``):
            Instance of ``CCLObject`` to unlock within the scope
            of the context manager.
        mutate (``bool``):
            If the enclosed function mutates the object, the stored
            representation is automatically deleted.
    """

    def __init__(self, instance, *, mutate=True):
        self.instance = instance
        self.mutate = mutate
        # Define these attributes for easy access.
        self.id = id(self)
        self.thread_lock = RLock()
        # We want to catch and exit if the instance is not a CCLObject.
        # Hopefully this will be caught downstream.
        self.check_instance = isinstance(instance, CCLObject)
        if self.check_instance:
            self.object_lock = instance._object_lock

    def __enter__(self):
        if not self.check_instance:
            return

        with self.thread_lock:
            # Prevent simultaneous enclosing of a single instance.
            if self.object_lock.active:
                # Context manager already active.
                return

            # Unlock and store the fingerprint of this context manager so that
            # only this context manager is allowed to run on the instance.
            self.object_lock.unlock(manager_id=self.id)

    def __exit__(self, type, value, traceback):
        if not self.check_instance:
            return

        # If another context manager is running,
        # do nothing; otherwise reset.
        if self.id != self.object_lock._lock_id:
            return

        # Lock the instance on exit.
        # self.object_lock.lock()  # TODO: Uncomment for CCLv3.

    @classmethod
    def unlock_instance(cls, func=None, *, name=None, mutate=True):
        """Decorator that temporarily unlocks an instance of CCLObject.

        Arguments:
            func (``function``):
                Function which changes one of its ``CCLObject`` arguments.
            name (``str``):
                Name of the parameter to unlock. Defaults to the first one.
                If not a ``CCLObject`` the decorator will do nothing.
            mutate (``bool``):
                If after the function ``instance_old != instance_new``, the
                instance is mutated. If ``True``, the representation of the
                object will be reset.
        """
        if func is None:
            # called with parentheses
            return functools.partial(cls.unlock_instance, name=name,
                                     mutate=mutate)

        if not hasattr(func, "__signature__"):
            # store the function signature
            func.__signature__ = signature(func)
        names = list(func.__signature__.parameters.keys())
        name = names[0] if name is None else name  # default name
        if name not in names:
            # ensure the name makes sense
            raise NameError(f"{name} does not exist in {func.__name__}.")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = func.__signature__.bind(*args, **kwargs)
            with UnlockInstance(bound.arguments[name], mutate=mutate):
                return func(*args, **kwargs)
        return wrapper

    @classmethod
    def Funlock(cls, cl, name, mutate: bool):
        """Allow an instance to change or mutate when `name` is called."""
        func = vars(cl).get(name)
        if func is not None:
            newfunc = cls.unlock_instance(mutate=mutate)(func)
            setattr(cl, name, newfunc)


unlock_instance = UnlockInstance.unlock_instance


def is_equal(this, other):
    """Powerful helper for equivalence checking."""
    try:
        np.testing.assert_equal(this, other)
        return True
    except AssertionError:
        return False


class _CustomMethod:
    """Subclasses specifying a method string control whether the custom
    method is used in subclasses of ``CCLObject``.
    """

    def __init_subclass__(cls, *, method):
        super().__init_subclass__()
        if method not in vars(cls):
            raise ValueError(
                f"Subclass must contain a default {method} implementation.")
        cls._method = method
        cls._enabled: bool = True
        cls._classes: dict = {}

    @classmethod
    def register(cls, cl):
        """Register class to the dictionary of classes with custom methods."""
        if cls._method in vars(cls):
            cls._classes[cl] = getattr(cl, cls._method)

    @classmethod
    def enable(cls):
        """Enable the custom methods if they exist."""
        for cl, method in cls._classes.items():
            setattr(cl, cls._method, method)
        cls._enabled = True

    @classmethod
    def disable(cls):
        """Disable custom methods and fall back to Python defaults."""
        for cl in cls._classes.keys():
            default = getattr(cls, cls._method)
            setattr(cl, cls._method, default)
        cls._enabled = False


class CustomEq(_CustomMethod, method="__eq__"):
    """Controls the usage of custom ``__eq__`` for ``CCLObjects``."""

    def __eq__(self, other):
        # Default `eq`.
        return self is other


class CustomRepr(_CustomMethod, method="__repr__"):
    """Controls the usage of custom ``__repr__`` for ``CCLObjects``."""

    def __repr__(self):
        # Default `repr`.
        return object.__repr__(self)


class CCLObject(ABC):
    """Base for CCL objects.

    All CCL objects inherit ``__eq__`` and ``__hash__`` methods from here.
    Both methods rely on ``__repr__`` uniqueness. This aims to homogenize
    equivalence checking, and to standardize the use of hash.

    Overview
    --------
    ``CCLObjects`` inherit ``__hash__``, which consistently hashes the
    representation string. They also inherit ``__eq__`` which checks for
    representation equivalence.

    In the implemented scheme, each ``CCLObject`` may have its own, specialized
    ``__repr__`` method overloaded. Object representations have to be unique
    for equivalent objects. If no ``__repr__`` is provided, the default from
    ``object`` is used.

    Mutation
    --------
    ``CCLObjects`` are by default immutable. This aims to provide a failsafe
    mechanism, where, changing attributes has to trigger a re-computation
    of something else inside of the instance, rather than simply doing a value
    change.

    This immutability mechanism can be safely bypassed if a subclass defines an
    ``update_parameters`` method. ``CCLObjects`` temporarily unlock whenever
    this method is called.

    Internal State vs. Mutation
    ---------------------------
    Other methods that use ``setattr`` can only do that if they are decorated
    with ``@unlock_instance`` or if the particular code block that makes the
    change is enclosed within the ``UnlockInstance`` context manager.
    If neither is provided, an exception is raised.

    If such methods only change the instance's internal state, the decorator
    may be called with ``@unlock_instance(mutate=False)`` (or equivalently
    for the context manager ``UnlockInstance(..., mutate=False)``). Otherwise,
    the instance is assumed to have mutated.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # 1. Store the initialization signature on import.
        cls.__signature__ = signature(cls.__init__)

        # 2. Register subclasses with custom dunder method implementations.
        CustomEq.register(cls)
        CustomRepr.register(cls)

        # 3. Unlock instance on specific methods.  # TODO: Uncomment for CCLv3.
        # UnlockInstance.Funlock(cls, "__init__", mutate=False)
        # UnlockInstance.Funlock(cls, "update_parameters", mutate=True)

    def __new__(cls, *args, **kwargs):
        # Populate every instance with an `ObjectLock` as attribute.
        instance = super().__new__(cls)
        object.__setattr__(instance, "_object_lock", ObjectLock())
        return instance

    def __setattr__(self, name, value):
        if self._object_lock.locked:
            raise AttributeError("CCL objects can only be updated via "
                                 "`update_parameters`, if implemented.")
        object.__setattr__(self, name, value)

    def update_parameters(self, **kwargs):
        name = self.__class__.__name__
        raise NotImplementedError(f"{name} objects are immutable.")

    def __repr__(self):
        # By default we use `__repr__` from `object`.
        return object.__repr__(self)

    def __hash__(self):
        # `__hash__` makes use of the `repr` of the object,
        # so we have to make sure that the `repr` is unique.
        return hash(repr(self))

    def __eq__(self, other):
        # Exit early if it is the same object.
        if self is other:
            return True
        # Two same-type objects are equal if their representations are equal.
        if type(self) is not type(other):
            return False
        # Compare the attributes listed in `__eq_attrs__`.
        if hasattr(self, "__eq_attrs__"):
            for attr in self.__eq_attrs__:
                if not is_equal(getattr(self, attr), getattr(other, attr)):
                    return False
            return True
        return False


class CCLAutoRepr(CCLObject):
    """Base for objects with automatic representation. Representations
    for instances are built from a list of attribute names specified as
    a class variable in ``__repr_attrs__`` (acting as a hook).

    Example:
        The representation (also hash) of instances of the following class
        is built based only on the attributes specified in ``__repr_attrs__``:

        >>> class MyClass(CCLAutoRepr):
            __repr_attrs__ = ("a", "b", "other")
            def __init__(self, a=1, b=2, c=3, d=4, e=5):
                self.a = a
                self.b = b
                self.c = c
                self.other = d + e

        >>> repr(MyClass(6, 7, 8, 9, 10))
            <__main__.MyClass>
                a = 6
                b = 7
                other = 19
    """

    def __repr__(self):
        # Build string from specified `__repr_attrs__` or use Python's default.
        # Subclasses overriding `__repr__`, stop using `__repr_attrs__`.
        if hasattr(self.__class__, "__repr_attrs__"):
            from .repr_ import build_string_from_attrs
            return build_string_from_attrs(self)
        return object.__repr__(self)


class CCLNamedClass(CCLObject):
    """Base for objects that contain methods ``from_name()`` and
    ``create_instance()``.

    Implementation
    --------------
    Subclasses must define a ``name`` class attribute which allows the tree to
    be searched to retrieve the particular model, using its name.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Class attribute denoting the name of the model."""

    @classmethod
    def _subclasses(cls):
        # This helper returns a set of all subclasses.
        direct_subs = cls.__subclasses__()
        deep_subs = [sub for cl in direct_subs for sub in cl._subclasses()]
        return set(direct_subs).union(deep_subs)

    @classmethod
    def from_name(cls, name):
        """Obtain particular model."""
        mod = {p.name: p for p in cls._subclasses() if hasattr(p, "name")}
        if name not in mod:
            raise KeyError(f"Invalid model {name}.")
        return mod[name]

    @classmethod
    def create_instance(cls, input_, **kwargs):
        """Process the input and generate an object of the class.
        Input can be an instance of the class, or a name string.
        Optional ``**kwargs`` may be passed.
        """
        if isinstance(input_, cls):
            return input_
        if isinstance(input_, str):
            class_ = cls.from_name(input_)
            return class_(**kwargs)
        good, bad = cls.__name__, type(input_).__name__
        raise TypeError(f"Expected {good} or str but received {bad}.")
