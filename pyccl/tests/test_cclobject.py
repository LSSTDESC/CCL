import pytest
import pyccl as ccl
import functools


def all_subclasses(cls):
    """Get all subclasses of ``cls``. NOTE: Used in ``conftest.py``."""
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in all_subclasses(c)])


def test_method_control_raises():
    # All method control subclasses must contain a default implementation.
    with pytest.raises(ValueError):
        class MyMethodControl(ccl._core.schema._CustomMethod, method="name"):
            pass


def test_repr_control():
    # Test custom repr controls.
    cosmo = ccl.CosmologyVanillaLCDM()

    ccl.CustomRepr.disable()
    assert repr(cosmo) == object.__repr__(cosmo)

    ccl.CustomRepr.enable()
    assert repr(cosmo) != object.__repr__(cosmo)


def test_eq_control():
    # Test custom eq controls.
    cosmo = [ccl.CosmologyVanillaLCDM() for _ in range(2)]
    assert id(cosmo[0]) != id(cosmo[1])

    ccl.CustomEq.disable()
    assert cosmo[0] != cosmo[1]

    ccl.CustomEq.enable()
    assert cosmo[0] == cosmo[1]


def check_eq_repr_hash(self, other, *, equal=True):
    # Helper to ensure `__eq__`, `__repr__`, `__hash__` are consistent.
    if equal:
        return (self == other
                and repr(self) == repr(other)
                and hash(self) == hash(other))
    return (self != other
            and repr(self) != repr(other)
            and hash(self) != hash(other))


def test_CCLObject_immutability():
    # These tests check the behavior of immutable objects, i.e. instances
    # of classes where `Funlock` or `unlock_instance` is not used.
    # test `CCLObject` lock
    obj = ccl.CCLObject()
    obj._object_lock.unlock()
    assert "locked=False" in repr(obj._object_lock)
    obj._object_lock.lock()
    assert "locked=True" in repr(obj._object_lock)

    # `update_parameters` not implemented.
    cosmo = ccl.CosmologyVanillaLCDM()
    # with pytest.raises(AttributeError):  # TODO: Uncomment for CCLv3.
    #     cosmo.my_attr = "hello_world"
    with pytest.raises(NotImplementedError):
        cosmo.update_parameters(A_SPLINE_NA=120)

    # `update_parameters` implemented.
    prof = ccl.halos.HaloProfilePressureGNFW(mass_def="200c", mass_bias=0.5)
    # with pytest.raises(AttributeError):  # TODO: Uncomment for CCLv3.
    #     prof.mass_bias = 0.7
    assert prof.mass_bias == 0.5
    prof.update_parameters(mass_bias=0.7)
    assert prof.mass_bias == 0.7


def test_CCLObject_default_behavior():
    # Test that if `__repr__` is not defined the fall back is safe.
    MyType = type("MyType", (ccl.CCLObject,), {"test": 0})
    instances = [MyType() for _ in range(2)]
    assert check_eq_repr_hash(*instances, equal=False)

    # Test that all subclasses of ``CCLAutoRepr`` use Python's default
    # ``repr`` if no ``__repr_attrs__`` has been defined.
    instances = [ccl.CCLAutoRepr() for _ in range(2)]
    assert check_eq_repr_hash(*instances, equal=False)

    MyType = type("MyType", (ccl.CCLAutoRepr,), {"test": 0})
    instances = [MyType() for _ in range(2)]
    assert instances[0] != instances[1]


def test_named_class_raises():
    # Test that an error is raised if `create_instance` gets the wrong type.
    with pytest.raises(AttributeError):
        ccl.halos.MassDef.create_instance(1)


def test_ccl_parameters_abstract():
    # Test that the Parameters base class is abstract and cannot instantiate
    # if `instance` or `factory` are not specified.
    with pytest.raises(TypeError):
        ccl.CCLParameters()
    with pytest.raises(ValueError):
        class MyPars(ccl.CCLParameters):
            pass


# +==========================================================================+
# | The following functions are used by `conftest.py` to check correct setup.|
# +==========================================================================+


def init_decorator(func):
    """Check that all attributes listed in ``__repr_attrs__`` are defined in
    the constructor of all subclasses of ``CCLAutoRepr``.
    NOTE: Used in ``conftest.py``.
    """

    def in_mro(self):
        """Determine if `__repr_attrs__` is defined somewhere in the MRO."""
        # NOTE: This helper function makes sure that an AttributeError is not
        # raised when `super().__init__` is called inside another `__init__`.
        mro = self.__class__.__mro__[1:]  # MRO excluding this class
        for cls in mro:
            if hasattr(cls, "__repr_attrs__"):
                return True
        return False

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        if not hasattr(self, "__repr_attrs__"):
            # If `__repr_attrs__` is not specified, use local repr or inherit.
            return

        flag = [attr for attr in self.__repr_attrs__
                if not (hasattr(self, attr) or in_mro(self))]
        if flag:
            # NOTE: Set the attributes before calling `super`.
            raise AttributeError(f"{self.__class__.__name__}: attribute(s) "
                                 f"{flag} not set in __init__.")

    return wrapper


def test_unlock_instance_errors():
    # Test that unlock_instance gives the correct errors.

    # 1. Developer error
    with pytest.raises(NameError):
        @ccl.unlock_instance(name="hello")
        def func1(item, pk, a0=0, *, a1=None, a2):
            return

    # 2. User error
    @ccl.unlock_instance(name="pk")
    def func2(item, pk, a0=0, *, a1=None, a2):
        return

    with pytest.raises(TypeError):
        func2()

    # 3. Doesn't do anything if instance is not CCLObject.
    with ccl.UnlockInstance(True, mutate=False):
        pass
