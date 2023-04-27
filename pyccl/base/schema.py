"""
=================================
Schema (:mod:`pyccl.base.schema`)
=================================

Base class for all CCL objects, and functionality related to it.
"""

__all__ = ("Lock", "Unlock", "unlock",
           "CustomRepr", "CustomEq", "is_equal",
           "update",
           "CCLObject", "CCLNamedClass",)

import functools
from abc import ABC, abstractmethod
from inspect import signature, Parameter
from _thread import RLock
from typing import Any, Callable, Optional, Sequence, Type, Union

import numpy as np


class Lock:
    """Control the lock state (immutability) of an instance.

    This lock may be injected to instances during construction and, with
    :meth:`__setattr__` modified to check the :attr:`locked` property of the
    lock, object immutabilty (and controlled mutability) can be achieved.

    Example
    -------

    .. code-block:: python

        class Immutable:

            def __new__(cls):
                # create a new instance
                instance = super().__new__(cls)
                # inject a new lock
                object.__setattr__(instance, "_lock", Lock())
                return instance

            def __setattr__(self, name, value):
                if self._lock.locked:  # check if locked
                    raise AttributeError("Immutable class.")
                object.__setattr__(self, name, value)

        obj = Immutable()
        obj._lock.lock()  # lock the object
        obj.my_attr = 1  # AttributeError

    See Also
    --------
    :class:`~Unlock` : Temporarily unlock an immutable instance.
    """
    _locked: bool = False
    _lock_id: int = None

    def __repr__(self):
        return f"{self.__class__.__name__}(locked={self.locked})"

    @property
    def locked(self) -> bool:
        """Check if the object is locked."""
        return self._locked

    @property
    def active(self) -> bool:
        """Check if an unlocking context manager is active."""
        return self._lock_id is not None

    def lock(self) -> None:
        """Lock the object."""
        self._locked = True
        self._lock_id = None

    def unlock(self, manager_id: Optional[Any] = None) -> None:
        """Unlock the object.

        Arguments
        ---------
        manager_id : Any, optional
            Unique identifier of unlocker. May be used to prevent double-
            unlocking.
        """
        self._locked = False
        if manager_id is not None:
            self._lock_id = manager_id


class Unlock:
    """Context manager to temporarily unlock instances that contain a
    :class:`~Lock`.

    Parameters
    ----------
    instance : object
        Instance to unlock within the context manager block.
    lock_name : str
        Name of the attribute containing the :class:`~Lock`.
        The default is `'_lock'`.

    Example
    -------
    We instantiate class `Immutable` which is defined in the docs of
    :class:`~Lock`.

    .. code-block:: python

        obj = Immutable()
        obj._lock.lock()  # lock the object

        obj.my_attr = 1  # raises AttributeError

        with Unlock(obj):
            obj.my_attr = 1  # this works
    """

    def __init__(self, instance: object, *, lock_name: str = "_lock"):
        self.instance = instance
        # Define these attributes for easy access.
        self.id = id(self)
        self.thread_lock = RLock()
        # This context manager only unlocks objects containing an `Lock`.
        # We exit early if the object us not unlockable.
        attr = getattr(instance, lock_name, None)
        self.check_instance = isinstance(attr, Lock)
        if self.check_instance:
            self.object_lock = instance._lock

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
    def unlock(cls, func: Callable = None, *,
                        name: Optional[str] = None) -> Callable:
        """Wrapper to temporarily unlock a locked instance.

        Arguments
        ---------
        func : function
            Function which changes one of its locked arguments.
        name : str, optional
            Name of the parameter to unlock. If the parameter does not contain
            a lock the decorator will do nothing. The default is the first
            argument (which is usually `self`).

        Returns
        -------
        new_func : function
            Wrapped function which unlocks argument `name`.

        Raises
        ------
        NameError
            If `name` does not exist in the signature of `func`.

        Example
        -------
        We work with the `Immutable` class which is defined in the docs of
        :class:`~Lock`.

        .. code-block:: python

            @unlock(name="a")  # unlock argument `a` in `func`
            def func(a: Immutable):
                a.my_attr = 1
                a.my_other_attr = 2

        If the first argument is to be unlocked, `name` can be skipped.
        """
        if func is None:
            # called with parentheses
            return functools.partial(cls.unlock, name=name)

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
            with cls(bound.arguments[name]):
                return func(*args, **kwargs)
        return wrapper

    @classmethod
    def Funlock(cls, cl, *, funcname: str, argname: str = "self") -> None:
        """Helper which wraps a method in a class with :meth:`~unlock`
        to allow for mutation of a locked instance.

        Arguments
        ---------
        cl : class
            Class whose method will be wrapped and replaced.
        funcname : str
            Name of the method to wrap.
        argname : str, optional
            Name of the argument that changes in `funcname`. The default is
            `'self'`: the implicit first argument of instance methods.

        Example
        -------
        We subclass `Immutable` which is defined in the docs of
        :class:`~Lock`. We do not want to have to wrap with
        `@unlock` the :meth:`__init__` of all subclasses.

        .. code-block:: python

            class AutoImmutable(Immutable):

                def __init_subclass__(cls):
                    Unlock.Funlock(cls, funcname='__init__')

            class MyClass(AutoImmutable):

                def __init__(self):  # automatically wrapped to unlock `self`
                    self.my_attr = 1
        """
        func = vars(cl).get(funcname)
        if func is not None:
            newfunc = cls.unlock(name=argname)(func)
            setattr(cl, funcname, newfunc)


unlock = Unlock.unlock


def is_equal(this, other) -> bool:
    """Powerful helper for equivalence checking.

    See `numpy.assert_equal <https://numpy.org/doc/stable/reference/generated/
    numpy.testing.assert_equal.html>`_.
    """
    try:
        np.testing.assert_equal(this, other)
        return True
    except AssertionError:
        return False


class _CustomMethod:
    """Subclasses work as registers for classes that define a custom method
    implementation. These may then be turned on/off upon request.
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
    def register(cls, cl) -> None:
        """Register class to the dictionary of classes with custom methods."""
        if cls._method in vars(cls):
            cls._classes[cl] = getattr(cl, cls._method)

    @classmethod
    def enable(cls) -> None:
        """Enable the custom methods if they exist."""
        for cl, method in cls._classes.items():
            setattr(cl, cls._method, method)
        cls._enabled = True

    @classmethod
    def disable(cls) -> None:
        """Disable custom methods and fall back to Python defaults."""
        for cl in cls._classes.keys():
            default = getattr(cls, cls._method)
            setattr(cl, cls._method, default)
        cls._enabled = False


class CustomEq(_CustomMethod, method="__eq__"):
    """Control the usage of custom :meth:`__eq__` for all registered classes.

    Custom :meth:`__eq__` may be enabled/disabled in exchange for Python's
    default id-checking. Enable with `pyccl.CustomEq.enable()`. Disable with
    `pyccl.CustomEq.disable()`.

    Example
    -------

    .. code-block:: python

        class MyClass:

            def __init__(self, a=1):
                self.a = a

            def __eq__(self, other):
                return self.a == other.a

        CustomEq.register(MyClass)  # register the class

        obj1, obj2 = MyClass(), MyClass()
        print(obj1 == obj2)  # True - uses custom `__eq__()`

        CustomEq.disable()
        print(obj1 == obj2)  # False - uses Python's default `self is other`
    """

    def __eq__(self, other):
        # Default `eq`.
        return self is other


class CustomRepr(_CustomMethod, method="__repr__"):
    """Control the usage of custom :meth:`__repr__` for all registered classes.

    Custom :meth:`__repr__` may be enabled/disabled in exchange for Python's
    default repr. Enable with `pyccl.CustomRepr.enable()`. Disable with
    `pyccl.CustomRepr.disable()`.

    Example
    -------

    .. code-block:: python

        class MyClass:

            def __repr__(self):
                return "my_repr"

        CustomRepr.register(MyClass)  # register the class

        obj = MyClass()
        print(obj)  # 'my_repr'

        CustomRepr.disable()
        print(obj)  # <MyClass at 0x7f3bf530db90>
    """

    def __repr__(self):
        # Default `repr`.
        return object.__repr__(self)


def update(func: Callable = None, *, names: Sequence[str]) -> Callable:
    """Wrapper to automatically update model parameters.

    Extend the signature of a function to accept new keyword arguments, and
    update the corresponding instance attributes if the value of the parameter
    is not None.

    Arguments
    ---------
    func : function
        Function to wrap.
    names : Sequence
        Extend the original signature to make these parameters updatable.

    Returns
    -------
    new_func : function
        Function with extended signature as specified by `names`.

    Example
    -------
    The following versions are equivalent

    .. code-block:: python

        @update(names=["a",])
        def update_parameters(self) -> None:
            ...

        def update_parameters(self, *, a=None) -> None:
            if a is not None:
                self.a = a
    """
    if func is None:
        return functools.partial(update, names=names)

    # Extend the original signature.
    sig = signature(func)
    params = list(sig.parameters.values())
    for name in names:
        params.append(Parameter(name, Parameter.KEYWORD_ONLY, default=None))
    func.__signature__ = sig.replace(parameters=params)

    @functools.wraps(func)
    def wrapper(self, **kwargs):
        new = {param: v for param, v in kwargs.items() if param in names}
        old = {param: v for param, v in kwargs.items() if param not in names}
        for param, value in new.items():
            if value is not None:
                setattr(self, param, value)
        return func(self, **old)

    return wrapper


class CCLObject(ABC):
    """Base for CCL objects.

    Provide a framework for immutability, homogeneous representation and
    hashing (used for caching), and comparison.

    Comparison
    ----------
    Subclasses that override :meth:`__eq__` are registered, and the custom
    methods can be turned off (replaced with Python's default) or on through
    :class:`~CustomEq`.

    If comparison can be achieved by comparing a list of instance attributes, a
    sequence of their names may be defined in class variable `__eq_attrs__`,
    which provides a shortcut for automatic :meth:`__eq__` creation.

    The following class definitions are equivalent:

    .. code-block:: python

        class MyClass(CCLObject):
            __eq_attrs__ = ('a', 'b',)

            def __init__(self, a, b):
                self.a = a
                self.b = b

        class MyClass(CCLObject):

            def __init__(self, a, b):
                self.a = a
                self.b = b

            def __eq__(self, other):
                if self is other:
                    return True
                if type(self) is not type(other):
                    return False
                return self.a == other.a and self.b == other.b

    .. note::

        Subclasses overriding :meth:`__eq__` need to define :meth:`__repr__`.

    Representations
    ---------------
    As with comparisons, subclasses with custom :meth:`__repr__` are also
    registered and controlled by :class:`~CustomRepr`.

    Defining the class variable `__repr_attrs__` provides a shortcut for
    automatic :meth:`__repr__` creation.

    The following class definitions are equivalent:

    .. code-block:: python

        class MyClass(CCLObject):
            __repr_attrs__ = ('a', 'b',)

            def __init__(self, a, b):
                self.a = a
                self.b = b

        class MyClass(CCLObject):

            def __init__(self, a, b):
                self.a = a
                self.b = b

            def __repr__(self):
                sep = '\n\t'
                return f'<MyClass>{sep}a = {self.a}{sep}b = {self.b}'

        print(MyClass(a=1, b=2))
        <MyClass>
          a = 1
          b = 2

    Mutation
    --------
    Instances of this class are by default immutable. This provides a failsafe
    mechanism, where, changing attributes has to trigger a re-computation
    of something else inside of the instance, rather than simply doing a value
    change.

    This immutability mechanism can be bypassed if a subclass defines
    :meth:`update_parameters`. Instances temporarily unlock when this method
    is called. To temporarily unlock an instance in other methods, use the
    :func:`~unlock` decorator, or enclose the code block in an
    :func:`~Unlock` context manager.

    For simple updating of an instance attribute that only requires a value
    change (i.e. no re-computation of something internal), the :func:`~update`
    wrapper is provided.

    Raises
    ------
    AttributeError
        Trying to set an attribute of an immutable instance.
    NotImplementedError
        Trying to update the parameters of an immutable instance.

    See Also
    --------
    :class:`~CCLNamedClass` : Abstract subclass for named classes.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # 1. Store the initialization signature on import.
        cls.__signature__ = signature(cls.__init__)

        # 2. Register subclasses with custom dunder method implementations.
        CustomEq.register(cls)
        CustomRepr.register(cls)

        # 3. Unlock instance on specific methods.  # TODO: Uncomment for CCLv3.
        # Unlock.Funlock(cls, "__init__")
        # Unlock.Funlock(cls, "update_parameters")

    def __new__(cls, *args, **kwargs):
        # Populate every instance with an `Lock` as attribute.
        instance = super().__new__(cls)
        object.__setattr__(instance, "_lock", Lock())
        return instance

    def __setattr__(self, name, value):
        if self._lock.locked:
            raise AttributeError("CCL objects can only be updated via "
                                 "`update_parameters`, if implemented.")
        object.__setattr__(self, name, value)

    def update_parameters(self, **kwargs) -> None:
        """Override to allow parameter udpating. This is the default which
        raises a `NotImplementedError`.
        """
        name = self.__class__.__name__
        raise NotImplementedError(f"{name} objects are immutable.")

    def __repr__(self):
        # Build string from specified `__repr_attrs__` or use Python's default.
        # Subclasses overriding `__repr__`, stop using `__repr_attrs__`.
        if hasattr(self, "__repr_attrs__"):
            from .repr_ import build_string_from_attrs
            return build_string_from_attrs(self)
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

    @classmethod
    def _subclasses(cls) -> set:
        """Helper that returns a set of subclasses."""
        direct_subs = cls.__subclasses__()
        deep_subs = [sub for cl in direct_subs for sub in cl._subclasses()]
        return set(direct_subs).union(deep_subs)


class CCLNamedClass(CCLObject):
    """Base for objects with :meth:`from_name` and :meth:`create_instance`.

    Implementation
    --------------
    * Subclasses must define a `name` class attribute which allows the tree
      to be searched to retrieve the particular model, using its name.

    Example
    -------
    To avoid name collision, we typically subclass into an abstract class which
    covers all the individual models.

    .. code-block:: python

        class MassFunc(CCLNamedClass):  # abstract class
            ...

        class MassFuncTinker10(MassFunc):
            name = 'Tinker10'

        class HaloBias(CCLNamedClass):  # another abstract class
            ...

        class HaloBiasTinker10(HaloBias):
            name = 'Tinker10'  # shares the name with another concrete class

        print(MassFunc.from_name('Tinker10'))
        # <class 'MassFuncTinker10'>

        print(HaloBias.from_name('Tinker10'))
        # <class 'HaloBiasTinker10'>
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Class attribute denoting the name of the model."""

    @classmethod
    def from_name(cls, name: str) -> Type:
        """Obtain particular model.

        Arguments
        ---------
        name : str
            Name of model.

        Returns
        -------
        subclass : Type
            Subclass of `cls` with the particular name.

        Raises
        ------
        KeyError
            If subclass with `name` does not exist.

        See Also
        --------
        :meth:`~create_instance` : Instantiate the model directly.
        """
        mod = {p.name: p for p in cls._subclasses() if hasattr(p, "name")}
        if name not in mod:
            raise KeyError(f"Invalid model {name}.")
        return mod[name]

    @classmethod
    def create_instance(cls, input_: Union[object, str], **kwargs) -> object:
        """Process the input and generate an instance of the class. Input can
        be an instance or a name string. Optional `**kwargs` may be passed.

        Arguments
        ---------
        input_ : object or str
            If an `object`, return it. If a string, instantiate the subclass
            with that name.
        **kwargs
            Keyword arguments to be passed to the model constructor.

        Returns
        -------
        instance : object
            Instantiated model.

        Raises
        ------
        TypeError
            If the `input_` is neither a model name nor an instance of
            `cls`.

        Example
        -------

        .. code-block:: python

            class Concentration(CCLNamedClass):  # abstract class
                ...

            class ConcentrationDuffy08(Concentration):
                name = 'Duffy08'

                def __init__(self, mass_def):
                    ...

            print(Concentration.create_instance('Duffy08', mass_def='200c'))
            # <ConcentrationDuffy08 object at 0x7f3bf530db90>

        See Also
        --------
        :meth:`~from_name` : Obtain the model class without instantiating it.
        """
        if isinstance(input_, cls):
            return input_
        if isinstance(input_, str):
            class_ = cls.from_name(input_)
            return class_(**kwargs)
        good, bad = cls.__name__, type(input_).__name__
        raise TypeError(f"Expected {good} or str but received {bad}.")
