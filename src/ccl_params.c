#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ccl.h"
#include "ccl_params.h"

int ccl_get_pk_spline_na(void)
{
  if(ccl_splines==NULL || ccl_gsl==NULL) {
    ccl_cosmology_read_config();
  }
  return ccl_splines->A_SPLINE_NA_PK + ccl_splines->A_SPLINE_NLOG_PK - 1;
}

void ccl_get_pk_spline_a_array(int ndout,double* doutput,int *status)
{
  double *d;
  if(ccl_splines==NULL || ccl_gsl==NULL) {
    ccl_cosmology_read_config();
  }
  if(ndout!=ccl_get_pk_spline_na())
    *status=CCL_ERROR_INCONSISTENT;
  if(*status==0) {
    d=ccl_linlog_spacing(ccl_splines->A_SPLINE_MINLOG_PK,
			 ccl_splines->A_SPLINE_MIN_PK,
			 ccl_splines->A_SPLINE_MAX,
			 ccl_splines->A_SPLINE_NLOG_PK,
			 ccl_splines->A_SPLINE_NA_PK);
    if(d==NULL)
      *status=CCL_ERROR_MEMORY;
  }
  if(*status==0)
    memcpy(doutput,d,ndout*sizeof(double));
  free(d);
}

int ccl_get_pk_spline_nk(void)
{
  if(ccl_splines==NULL || ccl_gsl==NULL) {
    ccl_cosmology_read_config();
  }
  double ndecades = log10(ccl_splines->K_MAX) - log10(ccl_splines->K_MIN);
  return (int)ceil(ndecades*ccl_splines->N_K);
}

void ccl_get_pk_spline_lk_array(int ndout,double* doutput,int *status)
{
  double *d;
  if(ccl_splines==NULL || ccl_gsl==NULL) {
    ccl_cosmology_read_config();
  }
  if(ndout!=ccl_get_pk_spline_nk())
    *status=CCL_ERROR_INCONSISTENT;
  if(*status==0) {
    d=ccl_log_spacing(ccl_splines->K_MIN,ccl_splines->K_MAX,ndout);
    if(d==NULL)
      *status=CCL_ERROR_MEMORY;
  }
  if(*status==0) {
    for(int ii=0;ii<ndout;ii++)
      doutput[ii]=log(d[ii]);
  }
  free(d);
}
