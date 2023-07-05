__all__ = ('EmulatorPk', )

from abc import abstractmethod


class EmulatorPk(object):

    @abstractmethod
    def _get_pk_at_a(self, a, cosmo):
        """Get k vector and uninterpolated power spectrum at given a.

        Args:
            a (:obj:`float` or `array`):
                Scale factor.
            cosmo (:class:`~pyccl.cosmology.Cosmology`):
                Cosmological parameters.

        Returns:
            :tuple: k and pk arrays.
        """

    def get_pk_at_a(self, a, cosmo):
        """Get k vector and uninterpolated power spectrum at given a.

        Args:
            a (:obj:`float` or `array`):
                Scale factor.
            cosmo (:class:`~pyccl.cosmology.Cosmology`):
                Cosmological parameters.

        Returns:
            :tuple: k and pk arrays.
        """
        return self._get_pk_at_a(a, cosmo)

    @abstractmethod
    def _get_pk2d(self, cosmo):
        """Get a 2D interpolator for the power of k and a.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`):
                Cosmological parameters.

        Returns:
            :obj:`~pyccl.pk2d.Pk2D` object.
        """

    def get_pk2d(self, cosmo):
        """Get a 2D interpolator for the power of k and a.

        Args:
            cosmo (:class:`~pyccl.cosmology.Cosmology`):
                Cosmological parameters.

        Returns:
            :obj:`~pyccl.pk2d.Pk2D` object.
        """
        return self._get_pk2d(cosmo)
