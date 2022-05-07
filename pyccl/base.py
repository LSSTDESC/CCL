import sys
import functools
from collections import OrderedDict
import numpy as np
import warnings
from inspect import signature, Parameter
from _thread import RLock
from abc import ABC


def _to_hashable(obj):
    """Make unhashable objects hashable in a consistent manner."""

    if isinstance(obj, (int, float, str)):
        # Strings and Numbers are hashed directly.
        return obj

    elif hasattr(obj, "__iter__"):
        # Encapsulate all the iterables to quickly discard as needed.

        if isinstance(obj, np.ndarray):
            # Numpy arrays: Convert the data buffer to a byte string.
            return obj.tobytes()

        elif isinstance(obj, dict):
            # Dictionaries: Build a tuple from key-value pairs,
            # where all values are converted to hashables.
            out = dict.fromkeys(obj)
            for key, value in obj.items():
                out[key] = _to_hashable(value)
            # Sort unordered dictionaries for hash consistency.
            if isinstance(obj, OrderedDict):
                return tuple(obj.items())
            return tuple(sorted(obj.items()))

        else:
            # Iterables: Build a tuple from values converted to hashables.
            out = [_to_hashable(item) for item in obj]
            return tuple(out)

    elif hasattr(obj, "__hash__"):
        # Hashables: Just return the object.
        return obj

    # NotImplemented: Can't hash safely, so raise TypeError.
    raise TypeError(f"Hashing for {type(obj)} not implemented.")


def hash_(obj):
    """Generic hash method, which changes between processes."""
    digest = hash(repr(_to_hashable(obj))) + sys.maxsize + 1
    return digest


class _ClassPropertyMeta(type):
    """Implement `property` to a `classmethod`."""
    # TODO: in py39+ decorators `classmethod` and `property` can be combined
    @property
    def maxsize(cls):
        return cls._maxsize

    @maxsize.setter
    def maxsize(cls, value):
        if value < 0:
            raise ValueError(
                "`maxsize` should be larger than zero. "
                "To disable caching, use `Caching.disable()`.")
        cls._maxsize = value
        for func in cls._cached_functions:
            func.cache_info.maxsize = value

    @property
    def policy(cls):
        return cls._policy

    @policy.setter
    def policy(cls, value):
        if value not in cls._policies:
            raise ValueError("Cache retention policy not recognized.")
        if value == "lfu" != cls._policy:
            # Reset counter if we change policy to lfu
            # otherwise new objects are prone to being discarded immediately.
            # Now, the counter is not just used for stats,
            # it is part of the retention policy.
            for func in cls._cached_functions:
                for item in func.cache_info._caches.values():
                    item.reset()
        cls._policy = value
        for func in cls._cached_functions:
            func.cache_info.policy = value


class Caching(metaclass=_ClassPropertyMeta):
    """Infrastructure to hold cached objects.

    Caching is used for pre-computed objects that are expensive to compute.

    Attributes:
        maxsize (``int``):
            Maximum number of caches to store. If the dictionary is full, new
            caches are assigned according to the set cache retention policy.
        policy (``'fifo'``, ``'lru'``, ``'lfu'``):
            Cache retention policy.
    """
    _enabled: bool = True
    _policies: list = ['fifo', 'lru', 'lfu']
    _default_maxsize: int = 128   # class default maxsize
    _default_policy: str = 'lru'  # class default policy
    _maxsize = _default_maxsize   # user-defined maxsize
    _policy = _default_policy     # user-defined policy
    _cached_functions: list = []

    @classmethod
    def _get_key(cls, func, *args, **kwargs):
        """Calculate the hex hash from arguments and keyword arguments."""
        # get a dictionary of default parameters
        params = func.cache_info._signature.parameters
        # get a dictionary of the passed parameters
        passed = {**dict(zip(params, args)), **kwargs}
        # discard the values equal to the default
        defaults = {param: value.default for param, value in params.items()}
        return hex(hash_({**defaults, **passed}))

    @classmethod
    def _get(cls, dic, key, policy):
        """Get the cached object container
        under the implemented caching policy.
        """
        obj = dic[key]
        if policy == "lru":
            dic.move_to_end(key)
        # update stats
        obj.increment()
        return obj

    @classmethod
    def _pop(cls, dic, policy):
        """Remove one cached item as per the implemented caching policy."""
        if policy == "lfu":
            keys = list(dic)
            idx = np.argmin([item.counter for item in dic.values()])
            dic.move_to_end(keys[idx], last=False)
        dic.popitem(last=False)

    @classmethod
    def _decorator(cls, func, maxsize, policy):
        # assign caching attributes to decorated function
        func.cache_info = CacheInfo(func, maxsize=maxsize, policy=policy)
        func.clear_cache = func.cache_info._clear_cache
        cls._cached_functions.append(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not cls._enabled:
                # Cache emulators even when caching is disabled.
                if not func.__name__ == "_load_emu":
                    return func(*args, **kwargs)

            key = cls._get_key(func, *args, **kwargs)
            # shorthand access
            caches = func.cache_info._caches
            maxsize = func.cache_info.maxsize
            policy = func.cache_info.policy

            with RLock():
                if key in caches:
                    # output has been cached; update stats and return it
                    out = cls._get(caches, key, policy)
                    func.cache_info.hits += 1
                    return out.item

            with RLock():
                while len(caches) >= maxsize:
                    # output not cached and no space available, so remove
                    # items as per the caching policy until there is space
                    cls._pop(caches, policy)

            # cache new entry and update stats
            out = CachedObject(func(*args, **kwargs))
            caches[key] = out
            func.cache_info.misses += 1
            return out.item

        return wrapper

    @classmethod
    def cache(cls, func=None, *, maxsize=_maxsize, policy=_policy):
        """Cache the output of the decorated function, using the input
        arguments as a proxy to build a hash key.

        Arguments:
            func (``function``):
                Function to be decorated.
            maxsize (``int``):
                Maximum cache size for the decorated function.
            policy (``'fifo'``, ``'lru'``, ``'lfu'``):
                Cache retention policy. When the storage reaches maxsize
                decide which cached object will be deleted. Default is 'lru'.\n
                'fifo': first-in-first-out,\n
                'lru': least-recently-used,\n
                'lfu': least-frequently-used.
        """
        if maxsize < 0:
            raise ValueError(
                "`maxsize` should be larger than zero. "
                "To disable caching, use `Caching.disable()`.")
        if policy not in cls._policies:
            raise ValueError("Cache retention policy not recognized.")

        if func is None:
            # `@cache` with parentheses
            return functools.partial(
                cls._decorator, maxsize=maxsize, policy=policy)
        # `@cache()` without parentheses
        return cls._decorator(func, maxsize=maxsize, policy=policy)

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def reset(cls):
        cls.maxsize = cls._default_maxsize
        cls.policy = cls._default_policy

    @classmethod
    def clear_cache(cls):
        [func.clear_cache() for func in cls._cached_functions]


cache = Caching.cache


class CacheInfo:
    """Cache info container.
    Assigned to cached function as ``function.cache_info``.

    Parameters:
        func (``function``):
            Function in which an instance of this class will be assigned.
        maxsize (``Caching.maxsize``):
            Maximum number of caches to store.
        policy (``Caching.policy``):
            Cache retention policy.

    .. note ::

        To assist in deciding an optimal ``maxsize`` and ``policy``, instances
        of this class contain the following attributes:
            - ``hits``: number of times the function has been bypassed
            - ``misses``: number of times the function has computed something
            - ``current_size``: current size of the cache dictionary
    """

    def __init__(self, func, maxsize=Caching.maxsize, policy=Caching.policy):
        # we store the signature of the function on import
        # as it is the most expensive operation (~30x slower)
        self._signature = signature(func)
        self._caches = OrderedDict()
        self.maxsize = maxsize
        self.policy = policy
        self.hits = self.misses = 0

    @property
    def current_size(self):
        return len(self._caches)

    def __repr__(self):
        s = f"<{self.__class__.__name__}>"
        for par, val in self.__dict__.items():
            if not par.startswith("_"):
                s += f"\n\t {par} = {val!r}"
        s += f"\n\t current_size = {self.current_size!r}"
        return s

    def _clear_cache(self):
        self._caches = OrderedDict()
        self.hits = self.misses = 0


class CachedObject:
    """A cached object container.

    Attributes:
        counter (``int``):
            Number of times the cached item has been retrieved.
    """
    counter: int = 0

    def __init__(self, obj):
        self.item = obj

    def __repr__(self):
        s = f"CachedObject(counter={self.counter})"
        return s

    def increment(self):
        self.counter += 1

    def reset(self):
        self.counter = 0


def auto_assign(func, sig=None):
    """Decorator to automatically assign all parameters as instance attributes.
    This ought to be applied on constructor methods.

    Arguments:
        func (``function``):
            Function which takes the instance as its first argument.
            All function arguments will be assigned as attributes of the
            instance.
        sig (``inspect.Signature``, optional):
            A signature may be provided externally for speed.
    """
    sig = signature(func).parameters if sig is None else sig.parameters
    _, *params = [n for n in sig]
    _, *defaults = [p.default for p in sig.values()]

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # collect all input in one dictionary
        dic = {**dict(zip(params, args)), **kwargs}

        # assign the declared parameters
        for param, value in dic.items():
            setattr(self, param, value)

        # assign the undeclared parameters with default values
        for param, default in zip(reversed(params), reversed(defaults)):
            if not hasattr(self, param):
                setattr(self, param, default)

        # func may now override the attributes we just set
        func(self, *args, **kwargs)

    return wrapper


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

    def __init__(self, instance, mutate=True):
        self.instance = instance
        self.mutate = mutate
        # Define these attributes for easy access.
        self.id = id(self)
        self.lock = RLock()

    def check_instance(self):
        # We want to catch and exit if the instance is not a CCLObject.
        # Hopefully this will be caught downstream.
        return isinstance(self.instance, CCLObject)

    def __enter__(self):
        if not self.check_instance():
            return

        with self.lock:
            # Prevent simultaneous enclosing of a single instance.
            if self.instance._lock_id is not None:
                # Context manager already active.
                return

            # Unlock and store the fingerprint of this context manager so that
            # only this context manager is allowed to run on the instance.
            object.__setattr__(self.instance, "_locked", False)
            self.instance._lock_id = self.id

    def __exit__(self, type, value, traceback):
        if not self.check_instance():
            return

        # If another context manager is running,
        # do nothing; otherwise reset.
        if self.id != self.instance._lock_id:
            return

        with self.lock:
            # Reset `repr` if the object has been mutated.
            if self.mutate:
                try:
                    delattr(self.instance, "_repr")
                    delattr(self.instance, "_hash")
                except AttributeError:
                    # Object mutated but none of these exist.
                    pass

            # Lock the instance on exit.
            self.instance._lock_id = None
            self.instance._locked = True


def unlock_instance(func=None, *, argv=0, mutate=True):
    """Decorator that temporarily unlocks an instance of CCLObject.

    Arguments:
        func (``function``):
            Function which changes one of its ``CCLObject`` arguments.
        argv (``int``):
            Which argument should be unlocked. Defaults to the first argument.
        mutate (``bool``):
            If after the function ``instance_old != instance_new``, the
            instance is mutated. If ``True``, the representation of the
            object will be reset.
    """
    if func is None:
        # called with parentheses
        return functools.partial(unlock_instance, argv=argv, mutate=mutate)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Pick argument from list of `args` or `kwargs` as needed.
        size = len(args)
        arg = args[argv] if size > argv else list(kwargs.values())[argv-size]
        with UnlockInstance(arg, mutate=mutate):
            out = func(*args, **kwargs)
        return out
    return wrapper


# +==========================================================================+
# |  The following decorators are used to notify users about deprecations.   |
# +==========================================================================+
def deprecated(new_function=None):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used. If there is a replacement function,
    pass it as `new_function`.
    """
    def decorator(func):
        from .errors import CCLDeprecationWarning

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            s = f"The function {func.__qualname__} is deprecated."
            if new_function:
                s += f" Use {new_function.__qualname__} instead."
            warnings.warn(s, CCLDeprecationWarning)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def warn_api(func=None, *, pairs=[], reorder=[]):
    """ This decorator translates old API to new API for:
      - functions/methods whose arguments have been ranamed,
      - functions/methods with changed argument order,
      - constructors in the ``halos`` sub-package where ``cosmo`` is removed,
      - functions/methods where ``normprof`` is now a required argument.

    Parameters:
        pairs : list of pairs, optional
            List of renaming pairs ``('old', 'new')``.
        reorder : list, optional
            List of the **previous** order of the arguments whose order
            has been changed, under their **new** name.

    Example:
        We have the legacy constructor:

        >>> def __init__(self, cosmo, a, b, c=0, d=1, normprof=False):
                # do something
                return a, b, c, d, normprof

        and we want to change the API to

        >>> def __init__(self, a, *, see=0, bee, d=1, normprof=None):
                # do the same thing
                return a, bee, see, d, normprof

        Then, adding this decorator to our new function would preserve API

        >>> @warn_api(pairs=[('b', 'bee'), ('c', 'see')],
                      reorder=['bee', 'see'])

        - ``cosmo`` is automatically detected for all constructors in ``halos``
        - ``normprof`` is automatically detected for all decorated functions.
    """
    if func is None:
        # called with parentheses
        return functools.partial(warn_api, pairs=pairs, reorder=reorder)

    name = func.__qualname__
    plural = lambda expr: "" if not len(expr)-1 else "s"  # noqa: final 's'
    params = signature(func).parameters
    POK = Parameter.POSITIONAL_OR_KEYWORD
    KWO = Parameter.KEYWORD_ONLY
    pos_names = [k for k, v in params.items() if v.kind == POK]
    kwo_names = [k for k, v in params.items() if v.kind == KWO]
    npos = len(pos_names)
    rename = dict(pairs)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from .errors import CCLDeprecationWarning

        # API compatibility with `cosmo` as a first argument in `halos`.
        from .core import Cosmology
        catch_cosmo = args[1] if len(args) > 1 else kwargs.get("cosmo")
        if ("pyccl.halos" in func.__module__
                and func.__name__ == "__init__"
                and isinstance(catch_cosmo, Cosmology)):
            warnings.warn(
                f"Use of argument `cosmo` has been deprecated in {name}. "
                "This will trigger an exception in the future.",
                CCLDeprecationWarning)
            # `cosmo` may be in `args` or in `kwargs`, so we check both.
            args = tuple(
                item for item in args if not isinstance(item, Cosmology))
            kwargs.pop("cosmo", None)

        # API compatibility for reordered positionals in `fourier_2pt`.
        from .halos.profiles import HaloProfile
        first_arg = args[1] if len(args) > 1 else None
        if (func.__name__ == "fourier_2pt"
                and isinstance(first_arg, HaloProfile)):
            api = dict(zip(["prof", "cosmo", "k", "M", "a"], args[1: 6]))
            args = (args[0],) + args[6:]  # discard args [1-5]
            kwargs.update(api)            # they are now kwargs
            warnings.warn(
                "API for Profile2pt.fourier_2pt has changed. "
                "Argument order (prof, cosmo, k, M, a) has been replaced by "
                "(cosmo, k, M, a, prof).", CCLDeprecationWarning)

        # API compatibility for renamed arguments.
        warn_names = set(kwargs) - set(params)
        if warn_names:
            s = plural(warn_names)
            warnings.warn(
                f"Use of argument{s} {list(warn_names)} is deprecated "
                f"in {name}. Pass the new name{s} of the argument{s} "
                f"{[rename[k] for k in warn_names]}, respectively.",
                CCLDeprecationWarning)
            for param in warn_names:
                kwargs[rename[param]] = kwargs.pop(param)

        # API compatibility for star operator.
        if len(args) > npos:
            # API compatibility for shuffled order.
            if reorder:
                # Pick up the positions of the common elements.
                mask = [param in reorder for param in kwo_names]
                start = mask.index(True)
                stop = start + len(reorder)
                # Sort the reordered part of `kwo_names` by `reorder` indexing.
                kwo_names[start: stop] = sorted(kwo_names[start: stop],
                                                key=reorder.index)
            extras = dict(zip(kwo_names, args[npos:]))
            kwargs.update(extras)
            s = plural(extras)
            warnings.warn(
                f"Use of argument{s} {list(extras)} as positional is "
                f"deprecated in {func.__qualname__}.", CCLDeprecationWarning)

        # API compatibility for `normprof` as a required argument.
        if "normprof" in set(params) - set(kwargs):
            kwargs["normprof"] = False
            warnings.warn(
                "Halo profile normalization `normprof` has become a required "
                f"argument in {name}. Not specifying it will trigger an "
                "exception in the future", CCLDeprecationWarning)

        # Collect what's remaining and sort to preserve signature order.
        pos = dict(zip(pos_names, args))
        kwargs.update(pos)
        kwargs = {param: kwargs[param]
                  for param in sorted(kwargs, key=list(params).index)}

        return func(**kwargs)
    return wrapper


def deprecate_attr(getter=None, *, pairs=[]):
    """This decorator can be used to deprecate attributes,
    warning users about it and pointing them to the new attribute.

    Parameters
    ----------
    getter : slot wrapper ``__getattribute__``
        This is the getter method to be decorated.
    pairs : list of pairs
        List of renaming pairs ``('old', 'new')``.

    Example
    -------
    We have the legacy attribute ``old_name`` which we want to rename
    to ``new_name``. To achieve this we decorate the ``__getattribute__``
    method of the parent class in the main class body to retrieve the
    ``__getattr__`` method for the main class, like so:

    >>>  __getattr__ = deprecate_attr([('old_name', 'new_name')])(
             super.__getattribute__)

    Now, every time the attribute is called via its old name, the user will
    be warned about the renaming, and the attribute value will be returned.

    .. note:: Make sure that you bind ``__getattr__`` to the decorator,
              rather than ``__getattribute__``, because ``__getattr__``
              provides the fallback mechanism we want to use. Otherwise,
              an infinite recursion will initiate.

    """
    if getter is None:
        return functools.partial(deprecate_attr, pairs=pairs)

    rename = dict(pairs)

    @functools.wraps(getter)
    def wrapper(cls, name):
        from .errors import CCLDeprecationWarning

        if name in rename:
            new_name = rename[name]
            class_name = cls.__class__.__name__
            warnings.warn(
                f"Attribute {name} is deprecated in {class_name}. "
                f"Pass the new name {new_name}.", CCLDeprecationWarning)
            name = new_name

        return cls.__getattribute__(name)
    return wrapper


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
    # *** Information regarding the state of the CCLObject ***
    # Immutability lock. Disables `setattr`. (see `unlock_instance`)
    _locked: bool = False
    # Memory address of the unlocking context manager. (see `UnlockInstance`)
    _lock_id: int = None
    # Have all the arguments in the constructor been assigned as instance
    # attributes? (see `auto_assign`)
    _init_attrs_state: bool = False

    def __init_subclass__(cls, init_attrs=None, **kwargs):
        """Subclass initialization routine.

        Parameters:
            init_attrs (``bool``):
                If ``True``, assign all arguments of the constructor
                as instance attributes. (see ``~pyccl.base.auto_assign``)
        """
        # Store the signature of the constructor on import.
        cls._init_signature = signature(cls.__init__)

        if init_attrs is None:
            # If not specified, get from current state
            # because a parent class might have toggled it.
            init_attrs = cls._init_attrs_state

        if init_attrs and hasattr(cls, "__init__"):
            # Decorate the __init__ method with the auto-assigner.
            cls.__init__ = auto_assign(cls.__init__, sig=cls._init_signature)
            # Make sure this is inherited.
            cls._init_attrs_state = True

        if "__repr__" in vars(cls):
            # If the class defines a custom `__repr__`, this will be the new
            # `_repr` (which is cached). Decorator `cached_property` requires
            # that `__set_name__` is called on it.
            bmethod = functools.cached_property(cls.__repr__)
            cls._repr = bmethod
            bmethod.__set_name__(cls, "_repr")
            # Fall back to using `__ccl_repr__` from `CCLObject`.
            cls.__repr__ = cls.__ccl_repr__

        def Funlock(cl, name, mutate):
            # Allow instance to change or mutate if method `name` is called.
            func = vars(cl).get(name)
            if func is not None:
                newfunc = unlock_instance(mutate=mutate)(func)
                setattr(cl, name, newfunc)

        Funlock(cls, "__init__", False)
        Funlock(cls, "update_parameters", True)

        super().__init_subclass__(**kwargs)

    def __setattr__(self, name, value):
        if self._locked:
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


class CCLHalosObject(CCLObject, init_attrs=True):
    """Base for halo objects. Automatically assign all ``__init__``
    parameters as attributes.
    """

    def __repr__(self):
        # If all the passed parameters have been assigned as instance
        # attributes during construction, we can use these parameters
        # to build a unique string for each instance.
        from ._repr import _build_string_from_init_attrs
        return _build_string_from_init_attrs(self)

    # Decorate the default `__getattribute__` to preserve API following
    # the name changes of the specified instance attrbiutes.
    # TODO: remove for CCLv3.
    __getattr__ = deprecate_attr(
        pairs=[('mdef', 'mass_def'),
               ('_mdef', 'mass_def'),
               ('cM', 'c_m_relation'),
               ('_massfunc', 'mass_function'),
               ('_hbias', 'halo_bias')]
    )(super.__getattribute__)


def link_abstractmethods(cls=None, *, methods: list):
    """Abstract class decorator, (used together with ``@abstractmethod``)
    that links multiple abstract methods. Subclasses that define either of
    the linked methods will satisfy the abstraction requirement. Propagated
    via inheritance using the ``__linked_abstractmethods__`` hook.

    Example:
        Subclasses of the following class can be instantiated if either
        ``method1`` or ``method2`` are defined. Otherwise, it falls back
        to normal ``abc.ABCMeta`` behavior:

        >>> @link_abstractmethods(methods=['method1', 'method2'])
            class MyClass(metaclass=ABCMeta):
                @abstractmethod
                def method1(self):
                    ...
                @abstractmethod
                def method2(self):
                    ...
                @abstractmethod
                def another_method(self):
                    ...

        This subclass can be instantiated:

        >>> class MySubclass(MyClass):
                def method1(self):
                    ...
                def another_method(self):
                    ...

        This subclass can't be instantiated:

        >>> class MySubclass(MyClass):
                def another_method(self):
                    ...
    """
    if cls is None:
        # Avoid doubly-nested decorator factory.
        return functools.partial(link_abstractmethods, methods=methods)

    if not hasattr(cls, "__linked_abstractmethods__"):
        # Save the linked abstract methods as a hook.
        cls.__linked_abstractmethods__ = frozenset(methods)

    def is_abstract(cls, method):
        # Return True if a method is an abstract method.
        return getattr(getattr(cls, method), "__isabstractmethod__", False)

    def __new__(cl, *args, **kwargs):
        # Tap into instance creation and remove all linked abstract methods
        # from the `__abstractmethods__` hook.
        linked = cl.__linked_abstractmethods__
        if not all([is_abstract(cl, method) for method in linked]):
            # If not all are abstract, it means that at least one is defined.
            abstracts = (set(cl.__abstractmethods__) - set(linked))
            cl.__abstractmethods__ = frozenset(abstracts)

        return super(cls, cl).__new__(cl)

    cls.__new__ = __new__
    return cls
