__all__ = ("BaccoemuBaryons",)

from warnings import warn

from . import BaryonsBaccoemu


class BaccoemuBaryons(BaryonsBaccoemu):
    name = 'BaccoemuBaryons'

    def __init__(self, *args, **kwargs):
        """This throws a deprecation warning on initialization."""
        warn(f'Class {self.__class__.__name__} will be deprecated. ' +
             f'Please use {BaryonsBaccoemu.__name__} instead.',
             DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)
    pass
