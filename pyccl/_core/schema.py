"""
==================================
Schema (:mod:`pyccl._core.schema`)
==================================

Base class for all CCL objects, and functionality related to it.
"""

__all__ = (
    "CustomEq", "CustomRepr",
    "unlock_instance", "funlock", "update", "is_equal",
    "Immutable", "CCLObject", "CCLNamedClass",)

import functools
from abc import ABC, abstractmethod
from contextlib import nullcontext
from inspect import signature, Parameter
from typing import Callable, Optional, Sequence, Type, Union, final

import numpy as np


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


@final
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


@final
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


@final
class _UnlockContext:
    """Temporarily unlock instances of :class:`~Immutable`.

    .. warning::

        This context manager is coupled to :class:`~Immutable`. It should not
        be used directly. Instead, use :meth:`~Immutable.unlock`.

    .. warning::

        Do not nest more than one context manager unlocking the same instance.

    Parameters
    ----------
    instance
        Instance to unlock with the context manager protocol. Works as a
        :class:`~nullcontext` if not specified.
    """
    instance: Optional["Immutable"]

    def __init__(self, instance: Optional["Immutable"] = None):
        self.instance = instance

    def __enter__(self):
        if self.instance is not None:
            self.instance._locked = False

    def __exit__(self, type, value, traceback):
        if self.instance is not None:
            self.instance._locked = True


def unlock_instance(
        func: Callable = None,
        *,
        name: Optional[str] = None
) -> Callable:
    """Wrapper to temporarily unlock a locked instance.

    Arguments
    ---------
    func
        Function which changes one of its locked arguments.
    name
        Name of the parameter to unlock. If the parameter is not an instance of
        :class:`~Immutable` the decorator will do nothing. The default is the
        first argument (which is usually `self`).

    Returns
    -------

        Wrapped function which unlocks argument `name`.

    Raises
    ------
    NameError
        If `name` does not exist in the signature of `func`.

    Example
    -------

    .. code-block:: python

        from pyccl import Immutable

        @unlock_instance(name="a")  # unlock argument `a` in `func`
        def func(a: Immutable):
            a.my_attr = 1
            a.my_other_attr = 2

    If the first argument is to be unlocked, `name` can be omitted.

    See Also
    --------
    :meth:`~Immutable.unlock`
        Temporarily unlock an instance within a context manager.
    :func:`~funlock`
        Helper to unlock a specific method in a class.
    """
    if func is None:
        # called with parentheses
        return functools.partial(unlock_instance, name=name)

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
        arguments = func.__signature__.bind(*args, **kwargs).arguments
        obj = arguments[name]
        with obj.unlock() if isinstance(obj, Immutable) else nullcontext():
            return func(*args, **kwargs)
    return wrapper


def funlock(cls: Type, name: str) -> None:
    """Helper which wraps a method in a class with :meth:`~unlock_instance` to
    allow for mutation of a locked instance.

    Arguments
    ---------
    cls
        Class whose method will be wrapped and replaced.
    name
        Name of the method to wrap.

    Example
    -------
    We want to unlock `my_method` in all subclasses.

    .. code-block:: python

        class MyImmutable(Immutable):

            def __init_subclass__(cls):
                funlock(cls, 'my_method')

        class MyClass(MyImmutable):

            def my_method(self):  # automatically wrapped to unlock `self`
                self.my_attr = 1

    See Also
    --------
    :meth:`~Immutable.unlock`
        Temporarily unlock an instance within a context manager.
    :func:`~unlock_instance`
        Decorator to unlock an instance within the body of a function.
    """
    func = vars(cls).get(name)
    if func is not None:
        setattr(cls, name, unlock_instance(func))


def update(func: Callable = None, *, names: Sequence[str]) -> Callable:
    """Wrapper to automatically update model parameters.

    Extend the signature of a function to accept new keyword arguments, and
    update the corresponding instance attributes if the value of the parameter
    is not None.

    Arguments
    ---------
    func
        Function to wrap.
    names
        Extend the original signature to make these parameters updatable.

    Returns
    -------

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


class Immutable(ABC):
    r"""Implementation of an immutable object.

    Instances of this class are by default immutable, providing a failsafe
    mechanism, where, changing attributes has to trigger a re-computation of
    somethnig else inside the instance, rather than simply a value change.

    Subclasses which define :meth:`__init__` or :meth:`update_parameters`
    automatically unlock temporarily when these methods are called.

    Raises
    ------
    AttributeError
        Trying to set an attribute of an immutable instance.
    NotImplementedError
        Trying to update the parameters of an immutable instance.

    Example
    -------

    .. code-block:: python

        obj = Immutable()
        obj.my_attr = 1  # AttributeError

    See Also
    --------
    :func:`~update`
        Decorator to extend the signature of
        :meth:`~Immutable.update_parameters` for simple parameter updating
        (i.e. no re-computation of something internal).
    """
    _locked: bool = False  # TODO: Change to True for CCLv3.

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Unlock instance on specific methods.  # TODO: Uncomment for CCLv3.
        # funlock(cls, "__init__")
        # funlock(cls, "update_parameters")

    @final
    def unlock(self) -> _UnlockContext:
        """Context manager to temporarily unlock the instance.

        Example
        -------

        .. code-block::

            from pyccl import Immutable

            obj = Immutable()
            with obj.unlock():
                obj.my_attr = 1

        See Also
        --------
        :func:`~unlock_instance`
            Decorator to unlock an instance within the body of a function.
        :func:`~funlock`
            Helper to unlock a specific method in a class.
        """
        instance = self if self._locked else None
        return _UnlockContext(instance)

    def __setattr__(self, name, value):
        if self._locked and name != "_locked":
            tp = type(self).__name__
            raise AttributeError(f"Instances of {tp} can only be updated via "
                                 "`update_parameters()`, if implemented.")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if self._locked:
            raise AttributeError("Can't delete attributes of a locked object.")
        super().__delattr__(name)

    def update_parameters(self, **kwargs) -> None:
        """Override to allow parameter udpating. This is the default which
        raises :exc:`NotImplementedError`.
        """
        name = self.__class__.__name__
        raise NotImplementedError(f"{name} objects are immutable.")


class CCLObject(Immutable):
    r"""Base for CCL objects.

    Provide a framework representation and hashing (used for caching) and
    comparisons.

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

        Subclasses overriding :meth:`__eq__` need to define :meth:`__hash__`
        because Python discards it.

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
        # <MyClass>
        #   a = 1
        #   b = 2

    See Also
    --------
    :class:`~CCLNamedClass` : Abstract subclass for named classes.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__signature__ = signature(cls)  # store the signature

        # Register subclasses with custom dunder method implementations.
        CustomEq.register(cls)
        CustomRepr.register(cls)

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
        # Different type objects are unequal.
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

    Example
    -------
    To avoid name collision, we typically subclass into an abstract class which
    covers all related models.

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
        name
            Name of model.

        Returns
        -------

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
    def create_instance(cls, spec: Union[object, str], **kwargs) -> object:
        """Process the input and generate an instance of the class. Input can
        be an instance or a name string. Optional `**kwargs` may be passed.

        Arguments
        ---------
        spec
            If an `object`, return it. If a string, instantiate the subclass
            with that name.
        **kwargs
            Keyword arguments to be passed to the model constructor.

        Returns
        -------

            Instantiated model.

        Raises
        ------
        TypeError
            If `spec` is neither a model name nor an instance of `cls`.

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
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, str):
            class_ = cls.from_name(spec)
            return class_(**kwargs)
        good, bad = cls.__name__, type(spec).__name__
        raise TypeError(f"Expected {good} or str but received {bad}.")
