Components
-----

1. Core Cosmology (CX 4.2)
2. Numerical Libraries (CX 4.2 / CX 4.3)
3. Value-added calculations (CX 4.3)


Core Quantities
---------------

These are standard quantities the true values of which should be unaffected 
by nuisance parameters, though there are different models.

The code in this section will provide numerically validated results for 
basic cosmological functions with documented precision. 

In some cases they will be wrappers around existing code like CLASS 
with predefined settings, in others cases we may need new code.


- We are doing chi(z), H(z), D_X(z), D(z), V(z), dV/dZ, rho_crit
- P_L(k,z), P_NL(k,z), sigma(M)
- z, chi, t, a

Open questions:
- Scope of library - what else should be here and how far should we go?
    - Quantities
    - Models
 - How much do we open up interfaces to underlying codes like CLASS
- Mutable or immutable cosmology / recomputation
- Automatic or explicit computation models
- Units!

Numerical Libraries
-------------------

These are functions/tools that are generally useful across analyses and 
have generic behavior. They implement tricky calculations
which might cause numerical problems.

Examples:

- Limber integrator
- Non-limber integrator
- Hankel transforms

Open questions:
- what other calculations should go in this category?


Extended Theory Calculations
----------------------------

These are components of connecting the outputs of the core cosmology library
to observables. Many things in here will be model specific and have 
multiple versions.

In this part there are more analysis-specfic systematics models:

- Lensing weights with user-supplied photo-z models
- Mass functions
- Galaxy bias models


Open questions:

- What else is on this list?
- How should we structure work done by different DESC members on this?
    - Ownership of code by WG/Group/individual?
    - How can this code interact with the code above.
    - Where does it live? What is centralized?
    - Discussed in CI2 report.
- Code standards and review (also see CI2 report)


Applications
------------

- A supernova mu(z)
- Simple lensing C_ell
- Cluster number counts in a redshift bin

 


Immediate Actions
-----------------

Build an example code for getting
 - C_shear(ell)
 - mu(z)
