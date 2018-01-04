%module ccl_neutrinos

%{
#define SWIG_FILE_WITH_INIT
#include "../include/ccl_neutrinos.h"
%}

// Strip the ccl_ prefix from function names
%rename("%(strip:[ccl_])s") "";

// Automatically document arguments and output types of all functions
%feature("autodoc", "1");

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* a, int na)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* output, int nout)};

// Flag status variable as input/output variable
/*%apply (int* INOUT) {(int * status)};*/

%include "../include/ccl_neutrinos.h"

%inline %{

void Omeganuh2_vec(double Neff, double mnu, double TCMB,
                   double* a, int na,
                   double* output, int nout,
                   int* status)
{
    assert(nout == na);
    
    for(int i=0; i < na; i++){
      output[i] = ccl_Omeganuh2(a[i], Neff, mnu, TCMB, NULL, status);
    }   
}

void Omeganuh2_to_Mnu_vec(double Neff, double OmNuh2, double TCMB,
                          double* a, int na,
                          double* output, int nout,
                          int* status)
{
    assert(nout == na);
    
    for(int i=0; i < na; i++){
      output[i] = ccl_Omeganuh2_to_Mnu(a[i], Neff, OmNuh2, TCMB, NULL, status);
    }   
}

%}
