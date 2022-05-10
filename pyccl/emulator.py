from .base import CCLObject, cache, unlock_instance
from .pk2d import Pk2D
import numpy as np
from abc import abstractmethod


class Bounds(CCLObject):
    """Convenience class for storing dictionaries of emulator bounds.

    Parameters:
        bounds (dict):
            Dictionary of parameters and their bounds (``vmin, vmax``).
    """

    def __init__(self, bounds):
        self.bounds = bounds
        for par, vals in self.bounds.items():
            vmin, vmax = vals
            if not vmin <= vmax:
                raise ValueError(f"Malformed bounds for parameter {par}. "
                                 "Should be [min, max].")
            if not isinstance(vals, list):
                self.bounds[par] = list(vals)

    def __repr__(self):
        return repr(self.bounds)


class EmulatorObject:
    """Convenience class storing loaded instances of external emulators,
    and attaching them to their bounds within a single object.

    Parameters:
        model (external):
            Under normal use, this is the expensive part of the emulator
            which we don't want to reload every time.
        bounds (``dict``, ``callable``, or ``None``):
            Used for paramameter validation of the ``model``. Accepts one of:

            * Dictionary of the bounds ``{'param': [vmin, vmax]}``.

            * Callable doing the bound-checking, given a parameter dictionary.

            * ``None`` to fully rely on the emulator's internal routines.
    """

    def __init__(self, model, bounds=None):
        self.model = model
        if bounds is None:
            self.bounds = NotImplemented
            self.check_bounds = NotImplemented
        elif isinstance(bounds, dict):
            self.bounds = Bounds(bounds)
        elif callable(bounds):
            self.check_bounds = bounds
            self.bounds = NotImplemented
        else:
            raise ValueError("Unrecognized `bounds` input.")

    def check_bounds(self, proposal):
        """Check a dictionary of proposal parameters against the bounds.

        Arguments:
            proposal (dict):
                Dictionary of proposal parameters and values for the emulator.
        """
        for par, val in proposal.items():
            if par not in self.bounds.bounds:
                continue
            vmin, vmax = self.bounds.bounds[par]
            if not (vmin <= val <= vmax):
                raise ValueError(f"Parameter {par} out of bounds "
                                 "for current emulator configuration.")


class Emulator(CCLObject):
    """Abstract base class acting as hook for emulators.

    Specific implementations must subclass from here.

    Notes:
      * The only requirement is for a method ``_load_emu`` to return a
        ``pyccl.emulator.EmulatorObject``. The output is automatically cached
        and subsequent calls to ``_load_emu`` will return the cached object.

      * Method ``_load_emu`` is implicitly a class method and will be converted
        to one. It is not a requirement that you define it as a class method.

      * If the implemented emulator can be called with multiple configurations
        you may add your own **kwargs in ``_load_emu`` to provide model
        specifications. Different configurations will be cached separately.

      * You may need a method that translates any CCL parameters to a set
        of parameters the emulator can understand. To that end, if the method
        is named ``_build_parameters``, CCL will automatically allow changing
        of the instance via ``setattr`` inside of that method.
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def Funlock(cl, name, mutate):
            # Allow instance to change or mutate if method `name` is called.
            func = vars(cl).get(name)
            if func is not None:
                newfunc = unlock_instance(mutate=mutate)(func)
                setattr(cl, name, newfunc)

        Funlock(cls, "_build_parameters", False)

        # Subclasses with `_load_emu` methods are emulator implementations.
        # Automatically cache the result, and convert it to class method.
        if hasattr(cls, "_load_emu"):
            if getattr(cls._load_emu, "__isabstractmethod__", False):
                cls._load_emu = classmethod(cache(maxsize=8)(cls._load_emu))

    @abstractmethod
    def _load_emu(cls, **kwargs) -> EmulatorObject:
        """Implicit class method that loads the emulator (and its bounds)
        and returns an instance of ``EmulatorObject``. The configuration
        parameters in ``kwargs`` are used to cache the emulator.
        """

    def _build_parameters(self, **kwargs) -> None:
        """Emulator implementations that use this method have the instance
        ``self`` automatically unlocked so that ``setattr`` works as needed.
        Alternatively, because ``CCLObjects`` are immutable, the context
        manager ``UnlockInstance`` or the decorator ``unlock_instance``
        may be used.
        """


class PowerSpectrumEmulator(Emulator):
    """Base class for power spectrum emulators.

    This class uses subclassed power spectrum emulator implementations
    to build `Pk2D` object containing the power spectrum.

    Specific implementations should have at least one of the following methods:
        * ``_get_pk_linear(cosmo) -> a, k pka``
        * ``_get_pk_nonlin(cosmo) -> a, k, pka``
        * ``_get_nonlin_boost(cosmo) -> a, k, pka``
        * ``_get_baryon_boost(cosmo) -> a, k, pka``

    Further information for general use of the emulators can be found
    in the docs of ``pyccl.emulators.Emulator``.
    """

    @classmethod
    def from_name(cls, name):
        """Return an emulator instance from a name string."""
        pspec_emus = {p.name: p for p in cls.__subclasses__()}
        if name in pspec_emus:
            return pspec_emus[name]
        raise ValueError(f"Power spectrum emulator {name} not implemented.")

    def get_pk_linear(self, cosmo):
        """Linear matter power spectrum"""
        if not hasattr(self, "_get_pk_linear"):
            raise NotImplementedError(
                f"Emulator {self.name} does not have a method "
                "`_get_pk_linear` to compute the linear "
                "matter power spectrum.")

        a, k, pk = self._get_pk_linear(cosmo)
        pk2d = Pk2D(lk_arr=np.log(k), a_arr=a, pk_arr=np.log(pk))
        return pk2d

    def get_pk_nonlin(self, cosmo):
        """Non-linear matter power spectrum, given a model name."""
        if hasattr(self, "_get_pk_nonlin"):
            a, k, pk = self._get_pk_nonlin(cosmo)
            pk2d = Pk2D(lk_arr=np.log(k), a_arr=a, pk_arr=np.log(pk))
        elif hasattr(self, "_get_nonlin_boost"):
            # query the emulator
            a, k, pk = self._get_pk_linear(cosmo)
            anl, knl, fknl = self._get_nonlin_boost(cosmo)
            # construct Pk2D objects
            pk2d_lin = Pk2D(lk_arr=np.log(k), a_arr=a, pk_arr=np.log(pk))
            pk2d_nonlin_boost = Pk2D(lk_arr=np.log(knl), a_arr=anl,
                                     pk_arr=np.log(fknl))
            # multiply
            pk2d = pk2d_nonlin_boost * pk2d_lin
        else:
            raise NotImplementedError(
                f"Emulator {self.name} does not have any of the methods "
                "`_get_pk_nonlin` or `_get_nonlin_boost` to compute "
                "the non-linear matter power spectrum.")

        return pk2d

    def apply_nonlin_model(self, cosmo, pk_linear):
        if hasattr(self, "_get_nonlin_boost"):
            anl, knl, fknl = self._get_nonlin_boost(cosmo)
            pk2d_nonlin_boost = Pk2D(lk_arr=np.log(knl), a_arr=anl,
                                     pk_arr=np.log(fknl))
            pk2d = pk2d_nonlin_boost * pk_linear
        elif (hasattr(self, "_get_pk_linear") and
              hasattr(self, "_get_pk_nonlin")):
            # In this case we calculate the non-linear boost using
            # the ratio of nonlin/linear.
            # query the emulator
            al, kl, pkl = self._get_pk_linear(cosmo)
            anl, knl, pknl = self._get_pk_nonlin(cosmo)
            # construct Pk2D objects and take their ratio
            pk2d_lin = Pk2D(lk_arr=np.log(kl), a_arr=al, pk_arr=np.log(pkl))
            pk2d_nl = Pk2D(lk_arr=np.log(knl), a_arr=anl, pk_arr=np.log(pknl))
            pk2d_nonlin_boost = pk2d_nl * pk2d_lin**(-1)
            # multiply
            pk2d = pk2d_nonlin_boost * pk_linear
        else:
            raise NotImplementedError(
                f"Emulator {self.name} does not have any of the methods "
                "`_get_pk_linear`, `_get_pk_nonlin`, or `get_nonlin_boost` "
                "to apply the matter power spectrum correction.")

        return pk2d

    def include_baryons(self, cosmo, pk_in):
        if hasattr(self, "_get_baryon_boost"):
            a, k, pk = self._get_baryon_boost(cosmo)
            pk2d_baryon = Pk2D(lk_arr=np.log(k), a_arr=a, pk_arr=np.log(pk))
            pk2d = pk2d_baryon * pk_in
        else:
            # Here, we can't safely infer the baryon correction from
            # a ratio of power spectra because the baryon correction
            # could have been applied in any power spectrum.
            raise NotImplementedError(
                f"Emulator {self.name} does not have a method "
                "`_get_baryon_boost` to compute the baryon correction.")

        return pk2d
