v 0.3 API changes:

Summary: the user interface for setting up cosmologies with neutrinos has been altered. Users should from now on pass Neff, the effective number of relativistic neutrino species in the early universe, and mnu, either a sum or neutrino masses or a set of 3 neutrinos masses.

C library:
In ccl_core.c:
In the function, ccl\_parameters\_create, the arguements 'double N\_nu\_rel', and 'double N\_nu\_mass' have been removed. The arguments 'double Neff' and 'ccl\_mnu\_convention mnu\_type' have been added. The argument 'mnu' has changed in type from 'double mnu' to 'double* mnu'.
Similar changes apply in 'ccl\_cosmology\_create\_with\_params' and all 'ccl\_parameters\_create...nu' convenience functions.
In ccl_neutrinos.c:
The function ccl\_Omeganuh2\_to\_Mnu has been renamed ccl\_nu\_masses. The arguments 'double a' and 'gsl\_interp\_accel* accel' have been removed. The argument 'ccl\_neutrino\_mass\_splits mass\_split' has been added.

Python wrapper:
In core.py:
In the Parameters class, the arguments 'N\_nu\_rel', and 'N\_nu\_mass' have been removed. The optional argument 'mnu\_type' has been added. A similar change occurs in the Cosmology class. 
In neutrinos.py
In the function Omeganuh2, the argument 'Neff' has been removed. It is now fixed to the length of the argument 'mnu'.
The function 'Omeganuh2\_to\_Mnu' has been renamed 'nu\_masses'. The arguments 'a' and 'Neff' have been removed. The argument 'mass\_split' has been added.


