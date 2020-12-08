%module ccl_neutrinos

%{
/* put additional #include here */
%}

%include "../include/ccl_neutrinos.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* a, int na), (double* mnu, int nm)};
%apply (int DIM1, double* ARGOUT_ARRAY1){(int nout, double* output)};

%feature("pythonprepend") Omeganuh2_vec %{
    if numpy.shape(a) != (nout,):
        raise CCLError("Input shape for `a` must match `(nout,)`!")

    if numpy.shape(mnu) != (N_nu_mass,):
        raise CCLError("Input shape for `mnu` must match `(N_nu_mass,)`!")
%}

%inline %{

void Omeganuh2_vec(int N_nu_mass, double T_CMB, double* a, int na,
                   double* mnu, int nm, int nout, double* output, int* status) {
    for(int i=0; i < na; i++){
        output[i] = ccl_Omeganuh2(a[i], N_nu_mass, mnu, T_CMB, status);
    }
}

void nu_masses_vec(double OmNuh2, int label, double T_CMB,
                   int nout, double* output, int* status) {
    double* mnu;
    mnu = ccl_nu_masses(OmNuh2, label, T_CMB, status);
    for(int i=0; i < nout; i++){
        output[i] = *(mnu+i);
    }
}

%}
