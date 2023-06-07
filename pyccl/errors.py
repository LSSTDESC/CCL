__all__ = ("CCLError", "CCLWarning", "CCLDeprecationWarning",)

import warnings


class CCLError(RuntimeError):
    """Generic error."""


class CCLWarning(RuntimeWarning):
    """Generic warning."""


class CCLDeprecationWarning(DeprecationWarning):
    """Warning for deprecated features."""

    @classmethod
    def enable(cls):
        warnings.simplefilter("always")

    @classmethod
    def disable(cls):
        warnings.filterwarnings(action="ignore", category=cls)
