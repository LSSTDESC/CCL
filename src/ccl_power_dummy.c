#include "ccl_core.h"
#include "ccl_utils.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_interp.h"
#include "gsl/gsl_spline.h"
//#include "gsl/gsl_interp2d.h"
//#include "gsl/gsl_spline2d.h"
#include "ccl_placeholder.h"
#include "ccl_background.h"
#include "ccl_power.h"
#include "ccl_error.h"
//#include "../class/include/class.h"

double ccl_linear_matter_power(ccl_cosmology * cosmo, double a, double k){
    return 0.;
}

double ccl_nonlin_matter_power(ccl_cosmology * cosmo, double a, double k){
    return 0.;
}

void ccl_linear_matter_powers(ccl_cosmology * cosmo, int n, double a[n], double k[n], double output[n]){
    printf("ccl_linear_matter_powers\n");
}

double ccl_sigmaR(ccl_cosmology *cosmo, double R){
    return 0.;
}

double ccl_sigma8(ccl_cosmology *cosmo){
    return 0.;
}

void ccl_cosmology_compute_power(ccl_cosmology * cosmo){
    printf("ccl_cosmology_compute_power\n");
}
