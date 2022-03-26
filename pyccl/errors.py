from .base import CCLObject


class CCLError(CCLObject, RuntimeError):
    """A CCL-specific RuntimeError"""
    def __repr__(self):
        return 'pyccl.CCLError(%r)' % (str(self))


class CCLWarning(CCLObject, RuntimeWarning):
    """A CCL-specific warning"""
    def __repr__(self):
        return 'pyccl.CCLWarning(%r)' % (str(self))


class CCLDeprecationWarning(CCLObject, FutureWarning):
    """A CCL-specific deprecation warning."""
    def __repr__(self):
        return 'pyccl.CCLDeprecationWarning(%r)' % (str(self))

    @classmethod
    def enable(cls):
        import warnings
        warnings.simplefilter("always")

    @classmethod
    def disable(cls):
        import warnings
        warnings.filterwarnings(action="ignore", category=cls)
