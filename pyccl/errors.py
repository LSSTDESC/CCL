from .base import CCLObject


class CCLError(CCLObject, RuntimeError):
    """A CCL-specific RuntimeError"""
    def __repr__(self):
        return 'pyccl.CCLError(%r)' % (str(self))


class CCLWarning(CCLObject, RuntimeWarning):
    """A CCL-specific warning"""
    def __repr__(self):
        return 'pyccl.CCLWarning(%r)' % (str(self))
