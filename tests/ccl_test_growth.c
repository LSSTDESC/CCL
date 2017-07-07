#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// The tolerance in D(z) for all the
#define GROWTH_TOLERANCE 1.0e-4

CTEST_DATA(growth) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double N_nu_rel;
  double N_nu_mass;
  double mnu;
  double Omega_v[5];
  double Omega_k[5];
  double w_0[5];
  double w_a[5];
  
  double z[6];
  double gf[5][6];
};

// Read the fixed format file containing all the growth factor
// benchmarks
static void read_growth_test_file(double z[6], double gf[5][6])
{
  //Growth is normalized to ~a at early times
  FILE * f = fopen("./tests/benchmark/growth_model1-5.txt", "r");
  ASSERT_NOT_NULL(f);
  
  // Ignore header line
  char str[1024];
  char* rtn;
  rtn = fgets(str, 1024, f);
  
    // File is fixed format - five rows and six columns
  for (int i=0; i<6; i++) {
    int count = fscanf(f, "%le %le %le %le %le %le\n", &z[i],
		       &gf[0][i], &gf[1][i], &gf[2][i], &gf[3][i], &gf[4][i]);
    	// Check that all the stuff in the benchmark is there
    ASSERT_EQUAL(6, count);
  }
  fclose(f);
}

// Set up the cosmological parameters to be used in each of the
// models
CTEST_SETUP(growth) {
  // Values that are the same for all 5 models
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->N_nu_rel=0;
  data->N_nu_mass=0;
  data->mnu=0;
  
  
  // Values that are different for the different models
  double Omega_v[5] = {  0.7,  0.7,  0.7,  0.65, 0.75 };
  double w_0[5]     = { -1.0, -0.9, -0.9, -0.9, -0.9  };
  double w_a[5]     = {  0.0,  0.0,  0.1,  0.1,  0.1  };
  
  // Fill in the values from these constant arrays.
  for (int i=0; i<5; i++) {
    data->Omega_v[i] = Omega_v[i];
    data->w_0[i]     = w_0[i];
    data->w_a[i]     = w_a[i];
    data->Omega_k[i] = 1.0 - data->Omega_c - data->Omega_b - data->Omega_v[i];
  }

  // The file of benchmark data.
  read_growth_test_file(data->z, data->gf);
}

static void compare_growth(int model, struct growth_data * data)
{
  int status=0; 	
  // Make the parameter set from the input data
  // Values of some parameters depend on the model index
  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k[model], data->N_nu_rel, data->N_nu_mass, data->mnu, data->w_0[model], data->w_a[model], data->h, data->A_s, data->n_s,-1,NULL,NULL, &status);
  params.Omega_g=0;
  // Make a cosmology object from the parameters with the default configuration
  ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);
  ASSERT_NOT_NULL(cosmo);
  
  // Compare to benchmark data
  for (int j=0; j<6; j++) {
    double a = 1/(1.+data->z[j]);
    double gf_ij=ccl_growth_factor_unnorm(cosmo,a, &status);
    if (status) printf("%s\n",cosmo->status_message);
    double absolute_tolerance = GROWTH_TOLERANCE*data->gf[model][j];
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(data->gf[model][j], gf_ij, absolute_tolerance);
  }

  ccl_cosmology_free(cosmo);
}

//This test code compares the modified growth function computed by CCL
//against the exact result for a particular modification of the growth rate.
static void check_mgrowth(void)
{
  int ii,nz_mg=128;
  double *z_mg,*df_mg;
  ccl_parameters params1,params2;
  ccl_cosmology *cosmo1,*cosmo2;
  int status=0;
  z_mg=malloc(nz_mg*sizeof(double));
  df_mg=malloc(nz_mg*sizeof(double));
  for(ii=0;ii<nz_mg;ii++) {
    z_mg[ii]=4*(ii+0.0)/(nz_mg-1.);
    df_mg[ii]=0.1/(1+z_mg[ii]);
  }
  params1=ccl_parameters_create(0.25,0.05,0,0,0,0,-1,0,0.7,2.1E-9,0.96,-1,NULL,NULL, &status);
  params2=ccl_parameters_create(0.25,0.05,0,0,0,0,-1,0,0.7,2.1E-9,0.96,nz_mg,z_mg,df_mg, &status);
  cosmo1=ccl_cosmology_create(params1,default_config);
  cosmo2=ccl_cosmology_create(params2,default_config);

  //We have included a growth modification \delta f = K*a (with K==0.1 arbitrarily)
  //This case has an analytic solution, given by D(a) = D_0(a)*exp(K*(a-1))
  //Here we check the growth computed by the library with the analytic solution.
  for(ii=0;ii<nz_mg;ii++) {
    double a=1./(1+z_mg[ii]);
    double d1=ccl_growth_factor(cosmo1,a,&status);
    double d2=ccl_growth_factor(cosmo2,a,&status);
    double f1=ccl_growth_rate(cosmo1,a,&status);
    double f2=ccl_growth_rate(cosmo2,a,&status);
    double f2r=f1+0.1*a;
    double d2r=d1*exp(0.1*(a-1));
    ASSERT_DBL_NEAR_TOL(d2r/d2,1.,GROWTH_TOLERANCE);
    ASSERT_DBL_NEAR_TOL(f2r/f2,1.,GROWTH_TOLERANCE);
  }

  free(z_mg);
  free(df_mg);
  ccl_cosmology_free(cosmo1);
  ccl_cosmology_free(cosmo2);
}

CTEST2(growth, model_1) {
  int model = 0;
  compare_growth(model, data);
}

CTEST2(growth, model_2) {
  int model = 1;
  compare_growth(model, data);
}

CTEST2(growth, model_3) {
  int model = 2;
  compare_growth(model, data);
}

CTEST2(growth, model_4) {
  int model = 3;
  compare_growth(model, data);
}

CTEST2(growth, model_5) {
  int model = 4;
  compare_growth(model, data);
}

CTEST2(growth,mgrowth) {
  check_mgrowth();
}
