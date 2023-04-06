from _thread import RLock
from abc import ABC, abstractproperty
from inspect import signature
import functools


__all__ = ("ObjectLock", "UnlockInstance", "unlock_instance", "FancyRepr",
           "CCLObject", "CCLAutoRepr", "CCLNamedClass",)


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

        with self.thread_lock:
            # Reset `repr` if the object has been mutated.
            if self.mutate:
                try:
                    delattr(self.instance, "_repr")
                    delattr(self.instance, "_hash")
                except AttributeError:
                    # Object mutated but none of these exist.
                    pass

            # Lock the instance on exit.
            self.object_lock.lock()

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

    # TODO: Remove for CCLv3 & replace with Funlock.
    @classmethod
    def Funlock2(cls, cl, name):
        """Replacement for Funlock incompatibility with signature.bind."""

        def lock_on_exit(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                out = func(self, *args, **kwargs)
                init_name = self.__class__.__init__.__qualname__
                this_name = func.__qualname__
                if this_name is init_name:  # skip this if called via `super()`
                    self._object_lock.lock()
                return out
            return wrapper

        func = vars(cl).get(name)
        if func is not None:
            newfunc = lock_on_exit(func)
            setattr(cl, name, newfunc)


unlock_instance = UnlockInstance.unlock_instance


class FancyRepr:
    """Controls the usage of fancy ``__repr__` for ``CCLObjects."""
    _enabled: bool = True
    _classes: dict = {}

    @classmethod
    def add(cls, cl):
        """Add class to the internal dictionary of fancy-repr classes."""
        cls._classes[cl] = cl.__repr__

    @classmethod
    def enable(cls):
        """Enable fancy representations if they exist."""
        for cl, method in cls._classes.items():
            FancyRepr.bind_and_replace(cl, method)
        cls._enabled = True

    @classmethod
    def disable(cls):
        """Disable fancy representations and fall back to Python defaults."""
        for cl in cls._classes.keys():
            cl.__repr__ = object.__repr__
        cls._enabled = False

    @classmethod
    def bind_and_replace(cls, cl, method):
        """Bind ``method`` to class ``cl``, and replace original with default.
        This helper only works for binding and replacing ``__repr__`` methods
        for ``CCLObjects``.
        """
        # If the class defines a custom `__repr__`, this will be the new
        # `_repr` (which is cached). Decorator `cached_property` requires
        # that `__set_name__` is called on it.
        bmethod = functools.cached_property(method)
        cl._repr = bmethod
        bmethod.__set_name__(cl, "_repr")
        # Fall back to using `__ccl_repr__` from `CCLObject`.
        cl.__repr__ = cl.__ccl_repr__


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

        # 1. Store the signature of the constructor on import.
        cls.__signature__ = signature(cls.__init__)

        # 2. Replace repr (if implemented) with its cached version.
        if "__repr__" in vars(cls):
            FancyRepr.add(cls)
            FancyRepr.bind_and_replace(cls, cls.__repr__)

        # 3. Unlock instance on specific methods.
        # UnlockInstance.Funlock(cls, "__init__", mutate=False)
        UnlockInstance.Funlock2(cls, "__init__")  # TODO: Replace in CCLv3.
        UnlockInstance.Funlock(cls, "update_parameters", mutate=True)

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
        name = self.__class__.__qualname__
        raise NotImplementedError(f"{name} objects are immutable.")

    @functools.cached_property
    def _repr(self):
        # By default we use `__repr__` from `object`.
        return object.__repr__(self)

    @functools.cached_property
    def _hash(self):
        # `__hash__` makes use of the `repr` of the object,
        # so we have to make sure that the `repr` is unique.
        return hash(repr(self))

    def __ccl_repr__(self):
        # The custom `__repr__` is converted to a
        # cached property and is replaced by this method.
        return self._repr

    __repr__ = __ccl_repr__

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        # Two same-type objects are equal if their representations are equal.
        if self.__class__ is not other.__class__:
            return False
        return repr(self) == repr(other)


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


def _subclasses(cls):
    # This helper returns a set of all subclasses.
    direct_subs = cls.__subclasses__()
    deep_subs = [sub for cl in direct_subs for sub in cl._subclasses()]
    return set(direct_subs).union(deep_subs)


def from_name(cls, name):
    """Obtain particular model."""
    mod = {p.name: p for p in cls._subclasses() if hasattr(p, "name")}
    return mod[name]


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
    good, bad = cls.__name__, input_.__class__.__name__
    raise TypeError(f"Expected {good} or str but received {bad}.")


class CCLNamedClass(CCLObject):
    """Base for objects that contain methods ``from_name()`` and
    ``create_instance()``.

    Implementation
    --------------
    Subclasses must define a ``name`` class attribute which allows the tree to
    be searched to retrieve the particular model, using its name.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._subclasses = classmethod(_subclasses)
        if not hasattr(cls, "from_name"):
            cls.from_name = classmethod(from_name)
        cls.create_instance = classmethod(create_instance)

    @abstractproperty
    def name(self) -> str:
        """Class attribute denoting the name of the model."""
