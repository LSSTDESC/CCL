import warnings


class CCLError(RuntimeError):
    """A CCL-specific RuntimeError"""
    def __repr__(self):
        return 'pyccl.CCLError(%r)' % (str(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


class CCLWarning(RuntimeWarning):
    """A CCL-specific warning"""
    def __repr__(self):
        return 'pyccl.CCLWarning(%r)' % (str(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))


class CCLDeprecationWarning(FutureWarning):
    """A CCL-specific deprecation warning."""
    def __repr__(self):
        return 'pyccl.CCLDeprecationWarning(%r)' % (str(self))

    def __eq__(self, other):
        return repr(self) == repr(other)

    def __hash__(self):
        return hash(repr(self))

    @classmethod
    def enable(cls):
        warnings.simplefilter("always")

    @classmethod
    def disable(cls):
        warnings.filterwarnings(action="ignore", category=cls)
