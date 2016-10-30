#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

//This test code compares the modified growth function computed by CCL
//against the exact result for a particular modification of the growth rate.
int main(int argc, char * argv[])
{
  int status=0;
  int ii,nz_mg=128;
  double *z_mg,*df_mg;
  ccl_parameters params1,params2;
  ccl_cosmology *cosmo1,*cosmo2;

  z_mg=malloc(nz_mg*sizeof(double));
  df_mg=malloc(nz_mg*sizeof(double));
  for(ii=0;ii<nz_mg;ii++) {
    z_mg[ii]=4*(ii+0.0)/(nz_mg-1.);
    df_mg[ii]=0.1/(1+z_mg[ii]);
  }
  params1=ccl_parameters_create(0.25,0.05,0,0,-1,0,0.7,2.1E-9,0.96,-1,NULL,NULL);
  params2=ccl_parameters_create(0.25,0.05,0,0,-1,0,0.7,2.1E-9,0.96,nz_mg,z_mg,df_mg);
  cosmo1=ccl_cosmology_create(params1,default_config);
  cosmo2=ccl_cosmology_create(params2,default_config);

  //We have included a growth modification \delta f = K*a (with K==0.1 arbitrarily)
  //This case has an analytic solution, given by D(a) = D_0(a)*exp(K*(a-1))
  //Here we check the growth computed by the library with the analytic solution.
  for(ii=0;ii<nz_mg;ii++) {
    double a=1./(1+z_mg[ii]);
    double d1=ccl_growth_factor(cosmo1,a);
    double d2=ccl_growth_factor(cosmo2,a);
    double f1=ccl_growth_rate(cosmo1,a);
    double f2=ccl_growth_rate(cosmo2,a);
    double f2r=f1+0.1*a;
    double d2r=d1*exp(0.1*(a-1));
    if((fabs(d2r/d2-1)>1E-4) || (fabs(f2r/f2-1)>1E-4)) {
      fprintf(stderr,"Bad accuracy at z=%lE, D(f) = %lE, D(D) = %lE\n",z_mg[ii],f2r-f2,d2r-d2);
      exit(1);
    }
  }
  printf("Success!\n");

  free(z_mg);
  free(df_mg);
  ccl_cosmology_free(cosmo1);
  ccl_cosmology_free(cosmo2);

  printf("%d\n",status);
  return status;
}
/*
dlog(D)/dloga = dlog(D0)/dloga + Df
dlog(D) = dlog(D0) + Df * dloga

-log(D(a)) = -dlog(D0(a))+Int[ Df da/a,a,1]

D(a) = D0(a) * exp(-Int[Df da/a,a,1]) 

Df=0.1*a -> D(a) = D0(a)*exp(0.1*(a-1))
*/
