****************************************
Core Cosmology Library
****************************************

The Core Cosmology Library (CCL) is a standardized library of routines to calculate basic observables used in cosmology. It will be the standard analysis package used by the LSST Dark Energy Science Collaboration (DESC).

The core functions of this package include:

 * Matter power spectra :math:`P(k)` from numerous models including CLASS, the Mira-Titan Emulator  and halofit
 * Hubble constant :math:`H(z)` as well as comoving distances :math:`\chi(z)` and distance moduli :math:`\mu(z)`
 * Growth of structure :math:`D(z)` and :math:`f`
 * Angular power spectra :math:`C_\ell` and correlation functions :math:`\xi` for arbitrary
   combinations of tracers including galaxies, shear and number counts
 * Halo mass function :math:`{\rm d}n/{\rm d}M` and halo bias :math:`b(M)`
 * Approximate baryonic modifications to the matter power spectra :math:`\Delta^2_{\rm baryons}`
 * Simple modified gravity extensions :math:`\Delta f(z)` and :math:`\mu-\Sigma`

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
