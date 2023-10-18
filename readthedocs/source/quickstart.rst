.. _quickstart:

**********
Quickstart
**********

CCL is structured around :class:`~pyccl.cosmology.Cosmology` objects which hold the cosmological
parameters and any tabulated data associated with a given cosmology. The
library then provides functions to compute specific quantities of interest.
See the full API documentation through the :mod:`pyccl` module and submodules for more details.

Here are some of the generic conventions followed by CCL:

  - **All** units are **non**-h-inverse (e.g., Mpc as opposed to Mpc/h).
  - Distances are in units of Mpc, masses are in :math:`M_\odot` units.
  - The scale factor ``a`` is preferred over redshift ``z`` as a time label
    (although this is not always respected!).
  - The :class:`~pyccl.cosmology.Cosmology` object always comes first in
    most function calls that require it.
  - Argument ordering for power spectra is ``(k, a)``.
  - Argument ordering for halo model functions is ``(M, a)`` or ``(k, M, a)`` (when both
    wavenumbers and masses are required).

This example computes the comoving distance, HALOFIT non-linear
power spectrum using the BBKS transfer function, and the cross-power
spectrum between cmb lensing and a sample of galaxies around :math:`z=1`.

.. code-block:: python

   >>> import pyccl as ccl
   >>> import numpy as np
   >>> cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05,
                             h=0.7, n_s=0.95, sigma8=0.8,
                             transfer_function='bbks')
   >>> cosmo.sigma8()  # get sigma8
   0.8
   >>> z = 1
   >>> ccl.comoving_radial_distance(cosmo, 1/(1+z))  # comoving distance to z=1 in Mpc
   3303.5260243050056
   >>> ccl.nonlin_matter_power(cosmo, k=1, a=1/(1+z))  # HALOFIT P(k) at k,z = 1,1
   array(143.65934042)
   >>> zs = np.linspace(0, 2, 512)
   >>> ells = np.array([10, 100, 1000])
   >>> gals = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(zs, np.exp(-((zs-z)/0.1)**2)),
                                     bias=(zs, np.ones_like(zs)))
   >>> cmbk = ccl.CMBLensingTracer(cosmo, z_source=1100)
   >>> ccl.angular_cl(cosmo, gals, cmbk, ells)
   array([1.71356558e-07, 1.80803491e-07, 7.51805386e-09])

See :ref:`models` for more details on the supported models for various cosmological
quantities and the specification of the cosmological parameters.

A comprehensive set of examples showcasing the different types of functionality
implemented in CCL can be found `here <https://github.com/LSSTDESC/CCLX>`_.
