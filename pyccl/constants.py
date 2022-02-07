"""This file exposes constants present in CCL."""
# flake8: noqa
from . import ccllib as lib
from .ccllib import (
    CCL_ERROR_CLASS, CCL_ERROR_INCONSISTENT, CCL_ERROR_INTEG,
    CCL_ERROR_LINSPACE, CCL_ERROR_MEMORY, CCL_ERROR_ROOT, CCL_ERROR_SPLINE,
    CCL_ERROR_SPLINE_EV, CCL_CORR_FFTLOG, CCL_CORR_BESSEL, CCL_CORR_LGNDRE,
    CCL_CORR_GG, CCL_CORR_GL, CCL_CORR_LP, CCL_CORR_LM)



class ParamStruct(object):
    """Dict-like structure with frozen keys and mutable values.

    Parameters:
        dic (``dict``):
            Dictionary of parameters and values

    Attributes:
        _locked (``bool``):
            Switch to make the keys immutable.
        _dic_init (``dict``):
            Store the original dictionary; used by the ``reload`` method.
        _names (``list`` of ``str``):
            Names of the parameters.
    """
    _locked = False

    def __init__(self, dic):
        self._dic_init = dic  # store defaults from the C layer
        self._names = list(self._dic_init.keys())
        self._setup()

    def _setup(self):
        for param, value in self._dic_init.items():
            setattr(self, param, value)
        self._locked = True

    def reload(self):
        self._setup()

    def __setattr__(self, param, value):
        if self._locked and not hasattr(self, param):
            raise KeyError(
                f"CCL global parameter {param} does not exist.")
        if "SPLINE_TYPE" in param:
            return
        if (param == "A_SPLINE_MAX") and (value != 1.0):
            # repeat the message from ccl_core.i
            raise RuntimeError(
                "A_SPLINE_MAX is fixed to 1.0 and is not mutable.")
        object.__setattr__(self, param, value)

    def __getitem__(self, param):
        return getattr(self, param)

    def __setitem__(self, param, value):
        setattr(self, param, value)

    def __repr__(self):
        params = self._dic_init.keys()
        s = ""
        for i, param in enumerate(params):
            s += "{" if i == 0 else " "
            s += f"'{param}': {getattr(self, param)}"
            s += "}" if i == len(params)-1 else ",\n"
        return s


class DefaultParameters(object):
    """Container class with methods to manipulate ``ParamStruct`` dicts.

    Parameters:
        None (only class method functionality implemented)
    """

    def __init__(self):
        pass

    @classmethod
    def from_struct(cls, name):
        """Construct a ``ParamStruct`` dict containing CCL parameters
        sourced from the C-layer.

        Arguments:
            name (``str``):
                Name of the SWIG-generated function which passes the
                parameters defined at the C-layer struct.

        Returns:
            ``ParamStruct``:
                Frozen dictionary of default parameters.
        """
        struct = getattr(lib, name)
        dir_ = dir(struct)
        params = dict()
        for param in dir_:
            if param.isupper():
                params[param] = getattr(struct, param)
        return ParamStruct(params)

    @classmethod
    def populate(cls, cosmo):
        """Populate the parameters of ``Cosmology.cosmo`` with those
        stored in this instance of pyccl's config:
            - ``pyccl.gsl_params``
            - ``pyccl.spline_params``

        Arguments:
            cosmo (``SWIG``: ``ccl_cosmology``):
                The ``ccl_cosmology`` struct via ``SWIG``.
        """
        from . import gsl_params, spline_params
        for param in gsl_params._names:
            value = getattr(gsl_params, param)
            setattr(cosmo.gsl_params, param, value)
        for param in spline_params._names:
            if "SPLINE_TYPE" in param:
                continue
            if param == "A_SPLINE_MAX":
                continue
            value = getattr(spline_params, param)
            setattr(cosmo.spline_params, param, value)
