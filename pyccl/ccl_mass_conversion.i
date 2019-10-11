%module ccl_mass_conversion

%{
/* put additional #includes here */
%}

%include "../include/ccl_mass_conversion.h"

// Enable vectorised arguments for arrays
%apply (double* IN_ARRAY1, int DIM1) {(double* c_in, int nc)};
%apply (int DIM1, double* ARGOUT_ARRAY1) {(int nout, double* output)};

%feature("pythonprepend") %{
    if numpy.shape(c_in) != (nout,):
        raise CCLError("Input shape for `c` must match `(nout,)`!")
%}

%inline %{

void get_new_concentration_vec(double delta_old,
			       double* c_in, int nc,
			       double delta_new,
			       int nout, double *output,
			       int *status) {
  *status=0;
  ccl_get_new_concentration(delta_old, nc, c_in,
			    delta_new, output,status);
}
%}

/* The directive gets carried between files, so we reset it at the end. */
%feature("pythonprepend") %{ %}
