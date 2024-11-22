__all__ = ("CCLError", "CCLWarning", "CCLDeprecationWarning",
           "warnings", "update_warning_verbosity")

import warnings as warnings_builtin


_warning_importance = {'high': 10, 'low': 1}
_verbosity_thresholds = {'none': 100, 'low': 10, 'high': 1}


class warnings:
    _CCL_WARN_THRESHOLD = 10  # Equivalent to "low" verbosity

    def warn(*args, **kwargs):
        category = kwargs.get('category')
        importance = _warning_importance[kwargs.pop('importance', 'low')]

        if ((category in (CCLWarning, CCLDeprecationWarning)) and
                (importance < warnings._CCL_WARN_THRESHOLD)):
            return

        warnings_builtin.warn(*args, **kwargs)


def update_warning_verbosity(verbosity):
    """ Update the level of verbosity of the CCL warnings. Available
    levels are "none", "low", and "high". More warning messages will
    be output for higher verbosity levels. If "none", no CCL-level
    warnings will be shown. The default verbosity is "low". Note that
    unless the verbosity level is "high", all C-level warnings will
    be omitted.

    Args:
        verbosity (str): one of ``'none'``, ``'low'`` or ``'high'``.
    """

    if not (verbosity in ['none', 'low', 'high']):
        raise KeyError("`verbosity` must be one of {'none', 'low', 'high'}")
    warnings._CCL_WARN_THRESHOLD = _verbosity_thresholds[verbosity]

    # Remove C-level warnings
    from . import debug_mode

    if verbosity == 'high':
        debug_mode(True)
    else:
        debug_mode(False)


class CCLError(RuntimeError):
    """A CCL-specific RuntimeError"""
    def __repr__(self):
        return "pyccl.CCLError(%r)" % (str(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


class CCLWarning(RuntimeWarning):
    """A CCL-specific warning"""
    def __repr__(self):
        return "pyccl.CCLWarning(%r)" % (str(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


class CCLDeprecationWarning(DeprecationWarning):
    """A CCL-specific deprecation warning."""
    def __repr__(self):
        return "pyccl.CCLDeprecationWarning(%r)" % (str(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))

    @classmethod
    def enable(cls):
        warnings_builtin.simplefilter("always")

    @classmethod
    def disable(cls):
        warnings_builtin.filterwarnings(action="ignore", category=cls)
