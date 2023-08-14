****************************************
Core Cosmology Library
****************************************

The Core Cosmology Library (CCL) is a standardized library of routines to calculate basic observables used in cosmology. It will be the standard analysis package used by the LSST Dark Energy Science Collaboration (DESC).

The core functions of this package include:

 * Background quantities (Hubble parameter :math:`H(z)`, distances etc.).
 * Linear growth factor :math:`D(z)` and growth rate :math:`f(z)`.
 * Linear matter power spectra :math:`P(k)` from Boltzmann codes (CLASS, CAMB) and approximate fitting functions.
 * Non-linear matter power power spectra using HaloFit and the Mira-Titan Emulator.
 * Approximate baryonic modifications to the matter power spectra :math:`\Delta^2_{\rm baryons}`
 * Angular power spectra :math:`C_\ell` and correlation functions :math:`\xi` for arbitrary
   combinations of tracers including number counts, shear and CMB lensing, as well as custom-made tracers.
 * A comprehensive halo model framework able to combine different prescription for the halo mass function :math:`{\rm d}n/{\rm d}M`,
   halo bias :math:`b(M)`, concentration-mass relation :math:`c(M)`, mass definitions, and halo profiles, as well as to provide
   predictions for the halo-model power spectrum of arbitrary quantities.
 * Support for :math:`\Lambda` CDM, and :math:`w_0-w_a` CDM cosmologies with curvature, as well as simple
   modified gravity extensions (:math:`\Delta f(z)` and :math:`\mu-\Sigma`).

The source code is available on github at https://github.com/LSSTDESC/CCL.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   source/installation
   source/quickstart
   source/citation
   source/reporting_bugs

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide
   :name: devguide

   source/terms
   source/api
   source/navigating_the_code
   source/understanding_the_python_c_interface
   source/developer_installation
   source/development_workflow
   source/writing_and_running_unit_tests
   source/writing_and_running_benchmarks

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :name: apiref

   source/notation_and_other_cosmological_conventions.rst
   API Documentation<api/modules>
   source/changelog
