#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// The tolerance in D(z) for all the
#define GROWTH_TOLERANCE 1.0e-4

#define N_Z 10
#define N_MODEL 5

CTEST_DATA(growth_allz) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double Neff;
  double* mnu;
  double Omega_v[N_MODEL];
  double Omega_k[N_MODEL];
  double w_0[N_MODEL];
  double w_a[N_MODEL];
  ccl_mnu_convention mnu_type;
  double mu_0;
  double sigma_0;
  
  double z[N_Z];
  double gf[N_Z][N_MODEL];
};

// Read the fixed format file containing all the growth factor
// benchmarks
static void read_growth_test_file(const char* filename, double z[N_Z], double gf[N_Z][N_MODEL])
{
  //Growth is normalized to ~a at early times
  FILE * f = fopen(filename, "r");
  ASSERT_NOT_NULL(f);
  
  // Ignore header line
  fscanf(f, "%*[^\n]\n", NULL);

  double val;
  // Read the file
  for(int i_z=0; i_z<N_Z; i_z++) {
    for(int i_model=0; i_model<N_MODEL+1; i_model++) {
      int count = fscanf(f, "%le", &val);
      // Check that all the value was read successfully
      ASSERT_EQUAL(1, count);
      if(i_model == 0) {
        // The first column holds the redshift values
        z[i_z] = val;
      }
      else {
        gf[i_z][i_model-1] = val;
      }
    }
  }
  fclose(f);
}

// Set up the cosmological parameters to be used in each of the
// models
CTEST_SETUP(growth_allz) {
  // Values that are the same for all 5 models
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->Neff=0;
  double mnuval = 0.;
  data->mnu= &mnuval;
  data-> mnu_type = ccl_mnu_sum;
  data->mu_0 =0.;
  data->sigma_0 = 0.;
  
  
  // Values that are different for the different models
  double Omega_v[N_MODEL] = {  0.7,  0.7,  0.7,  0.65, 0.75 };
  double w_0[N_MODEL]     = { -1.0, -0.9, -0.9, -0.9, -0.9  };
  double w_a[N_MODEL]     = {  0.0,  0.0,  0.1,  0.1,  0.1  };
  
  // Fill in the values from these constant arrays.
  for (int i=0; i<N_MODEL; i++) {
    data->Omega_v[i] = Omega_v[i];
    data->w_0[i]     = w_0[i];
    data->w_a[i]     = w_a[i];
    data->Omega_k[i] = 1.0 - data->Omega_c - data->Omega_b - data->Omega_v[i];
  }

  // The file of benchmark data.
  read_growth_test_file("./tests/benchmark/growth_cosmomad_allz.txt", data->z, data->gf);
}

static void compare_growth(int model, struct growth_allz_data * data)
{
  int status=0; 	
  // Make the parameter set from the input data
  // Values of some parameters depend on the model index
  ccl_parameters params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k[model], 
                                                data->Neff, data->mnu, data->mnu_type, 
                                                data->w_0[model], data->w_a[model], 
                                                data->h, data->A_s, data->n_s,
                                                -1,-1,-1,data->mu_0, data->sigma_0, -1,NULL,NULL, &status);
  params.Omega_g=0;
  // Make a cosmology object from the parameters with the default configuration
  ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);
  ASSERT_NOT_NULL(cosmo);
  
  // Compare to benchmark data
  for (int i=0; i<N_Z; i++) {
    double a = 1/(1.+data->z[i]);
    double gf_ccl=ccl_growth_factor_unnorm(cosmo, a, &status);
    if (status) printf("%s\n",cosmo->status_message);

    double absolute_tolerance = GROWTH_TOLERANCE*data->gf[i][model];
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(data->gf[i][model], gf_ccl, absolute_tolerance);
  }

  ccl_cosmology_free(cosmo);
}

CTEST2(growth_allz, model_1) {
  int model = 0;
  compare_growth(model, data);
}

CTEST2(growth_allz, model_2) {
  int model = 1;
  compare_growth(model, data);
}

CTEST2(growth_allz, model_3) {
  int model = 2;
  compare_growth(model, data);
}

CTEST2(growth_allz, model_4) {
  int model = 3;
  compare_growth(model, data);
}

CTEST2(growth_allz, model_5) {
  int model = 4;
  compare_growth(model, data);
}
