**********************
Core Cosmology Library
**********************

The Core Cosmology Library (CCL) is a standardized library of routines to
calculate cosmological observables. It is the theory pipeline of LSST
Dark Energy Science Collaboration (DESC).

The core functions of this package include:
 * Background quantities (Hubble parameter :math:`H(z)`, distances etc.).
 * Linear growth factor :math:`D(z)` and growth rate :math:`f(z)`.
 * Linear matter power spectra :math:`P(k)` from Boltzmann codes (CLASS, CAMB)
   and approximate fitting functions.
 * Non-linear matter power power spectra using HaloFit and the Mira-Titan
   Emulator.
 * Baryonic modifications to the matter power spectrum
   :math:`\Delta^2_{\rm baryons}`
 * Angular power spectra :math:`C_\ell` and correlation functions :math:`\xi`
   for arbitrary combinations of tracers including number counts, shear and CMB
   lensing, as well as custom-made tracers.
 * A comprehensive halo model framework able to combine different models of the
   halo mass function :math:`{\rm d}n/{\rm d}M`, halo bias :math:`b(M)`,
   concentration-mass relation :math:`c(M)`, mass definitions, and halo
   profiles, as well as to provide predictions for the halo-model power
   spectrum, 1-halo trispectrum, and the super-sample covariance (SSC)
   effective trispectrum of arbitrary quantities.
 * Support for :math:`\rm \Lambda CDM`, and :math:`w_0-w_a` CDM cosmologies
   with curvature, as well as the :math:`\mu-\Sigma` modified gravity
   extension.
 * Correlations using Eulerian perturbation theory (with FAST-PT).
 * A calculator mode where custom background, growth, linear & non-linear
   matter power spectra can be input to compute cosmological observables.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   source/intro_installation
   source/intro_quickstart
   source/intro_citation

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :name: apiref

   source/ref_core_concepts
   Documentation<api/modules>
   source/ref_changelog

.. toctree::
   :maxdepth: 1
   :caption: Contributor's Guide
   :name: devguide

   source/dev_scope
   source/dev_installation
   source/dev_workflow
   source/dev_guidelines
   source/dev_python_c_interface
