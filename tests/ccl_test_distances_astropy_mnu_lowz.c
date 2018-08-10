#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// The tolerance in chi 
// We use 1.0e-3 here because we compare with benchmarks produced by 
// astropy, which uses a fitting formula for the neutrino
// phasespace integral, which itself differs from the exact expression
// at greater than a 1.0e-4 level.
#define DISTANCES_TOLERANCE 1.0e-3

CTEST_DATA(distances_mnu) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double Neff[5];
  double mnu0[3], mnu1[3], mnu2[3], mnu3[3], mnu4[3];
  ccl_mnu_convention mnu_type;
  double Omega_v[5];
  double Omega_k[5];
  double w_0[5];
  double w_a[5];
  
  double z[5];
  double chi[5][5];
  double dm[5][5];
};

// Read the fixed format file containing all the radial comoving
// distance benchmarks
static void read_chi_test_file(double z[5], double chi[5][5])
{
  //Distances are in Mpc
  FILE * f = fopen("./tests/benchmark/chi_mnu_model1-5.txt", "r");
  ASSERT_NOT_NULL(f);
  
  // Ignore header line
  char str[1024];
  fgets(str, 1024, f);
  
  // File is fixed format - five rows and six columns
  for (int i=0; i<5; i++) {
    int count = fscanf(f, "%le %le %le %le %le %le\n", &z[i],
               &chi[0][i], &chi[1][i], &chi[2][i], &chi[3][i], &chi[4][i]);
    // Check that all the stuff in the benchmark is there
    ASSERT_EQUAL(6, count);
  }
  fclose(f);
}

static void read_dm_test_file(double z[5], double dm[5][5])
{
  //Distances are in Mpc
  FILE * f = fopen("./tests/benchmark/dm_mnu_model1-5.txt", "r");
  ASSERT_NOT_NULL(f);

  // Ignore header line
  char str[1024];
  fgets(str, 1024, f);

  // File is fixed format - five rows and six columns
  for (int i=0; i<5; i++) {
    int count = fscanf(f, "%le %le %le %le %le %le\n", &z[i],
                       &dm[0][i], &dm[1][i], &dm[2][i], &dm[3][i], &dm[4][i]);
    // Check that all the stuff in the benchmark is there
    ASSERT_EQUAL(6, count);
  }
  fclose(f);
}

// Set up the cosmological parameters to be used in each of the
// models
CTEST_SETUP(distances_mnu) {
  // Values that are the same for all 5 models
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->mnu_type = ccl_mnu_list;

  // Values that are different for the different models
  double Omega_v[5] = {  0.7,  0.7,  0.7,  0.65, 0.75 };
  double w_0[5]     = { -1.0, -0.9, -0.9, -0.9, -0.9  };
  double w_a[5]     = {  0.0,  0.0,  0.1,  0.1,  0.1  };
  
  // We use a total of 3 neutrinos instead of 3.046 
  // This is to compare with benchmarks from astropy
  // which splits equally total N between all species
  double Neff[5] 	= {3, 3, 3, 3, 3};
  
  double mnu0[3]	= 	{0.04, 0., 0.};
  double mnu1[3]	= 	{0.05, 0.01, 0.};
  double mnu2[3]	= 	{0.03, 0.02, 0.04};
  double mnu3[3]	= 	{0.05, 0., 0.};
  double mnu4[3]	=	{0.03, 0.02, 0.};
  
  data->mnu0[0] = mnu0[0];
  data->mnu1[0] = mnu1[0];
  data->mnu2[0] = mnu2[0];
  data->mnu3[0] = mnu3[0];
  data->mnu4[0] = mnu4[0];
  
  data->mnu0[1] = mnu0[1];
  data->mnu1[1] = mnu1[1];
  data->mnu2[1] = mnu2[1];
  data->mnu3[1] = mnu3[1];
  data->mnu4[1] = mnu4[1];
  
  data->mnu0[2] = mnu0[2];
  data->mnu1[2] = mnu1[2];
  data->mnu2[2] = mnu2[2];
  data->mnu3[2] = mnu3[2];
  data->mnu4[2] = mnu4[2];

  // Fill in the values from these constant arrays.
  for (int i=0; i<5; i++) {
    data->Omega_v[i] = Omega_v[i];
    data->w_0[i]     = w_0[i];
    data->w_a[i]     = w_a[i];
    data->Omega_k[i] = 1.0 - data->Omega_c - data->Omega_b - data->Omega_v[i];
    data->Neff[i] = Neff[i];
  }

  // The file of benchmark data.
  read_dm_test_file(data->z, data->dm);
  read_chi_test_file(data->z, data->chi);
}

static void compare_distances_mnu(int model, struct distances_mnu_data * data)
{
  int status=0;
  // Make the parameter set from the input data
  // Values of some parameters depend on the model index
  
  // The arrays of massive neutrions are different lengths, so we need to have an if-statement here to deal with that.
  
  ccl_parameters params;
  
  
  if (model==0){
  
  params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k[model],
						data->Neff[model], data->mnu0, data-> mnu_type, 
						data->w_0[model], data->w_a[model],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  } else if (model==1){
	  params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k[model],
						data->Neff[model], data->mnu1, data->mnu_type,
						data->w_0[model], data->w_a[model],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  } else if (model==2){
	 params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k[model],
						data->Neff[model], data->mnu2, data->mnu_type,
						data->w_0[model], data->w_a[model],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  } else if (model ==3){
	params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k[model],
						data->Neff[model], data->mnu3, data->mnu_type,
						data->w_0[model], data->w_a[model],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  }else if (model ==4){
	  params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k[model],
						data->Neff[model], data->mnu4, data->mnu_type,
						data->w_0[model], data->w_a[model],
						data->h, data->A_s, data->n_s,-1,-1,-1,-1,NULL,NULL, &status);
  }
  
  // Make a cosmology object from the parameters with the default configuration
  ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);
  ASSERT_NOT_NULL(cosmo);
  
  // Compare to benchmark data
  for (int j=0; j<5; j++) {
    double a = 1/(1.+data->z[j]);
    double chi_ij=ccl_comoving_radial_distance(cosmo,a, &status);
    if (status) printf("%s\n",cosmo->status_message);
    double absolute_tolerance = DISTANCES_TOLERANCE*data->chi[model][j];
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(data->chi[model][j], chi_ij, absolute_tolerance);

    if(a!=1) {  //skip this test for a=1 since it will raise an error
        double dm_ij=ccl_distance_modulus(cosmo,a, &status);
        if (status) printf("%s\n",cosmo->status_message);
        //NOTE tolerances are different!
        absolute_tolerance = 10*DISTANCES_TOLERANCE*data->dm[model][j];
        if (fabs(absolute_tolerance)<1e-4) absolute_tolerance = 1e-4;
        ASSERT_DBL_NEAR_TOL(data->dm[model][j], dm_ij, absolute_tolerance);
    }
  }
  
  ccl_cosmology_free(cosmo);
}

CTEST2(distances_mnu, model_1) {
  int model = 0;
  compare_distances_mnu(model, data);
}

CTEST2(distances_mnu, model_2) {
  int model = 1;
  compare_distances_mnu(model, data);
}

CTEST2(distances_mnu, model_3) {
  int model = 2;
  compare_distances_mnu(model, data);
}

CTEST2(distances_mnu, model_4) {
  int model = 3;
  compare_distances_mnu(model, data);
}

CTEST2(distances_mnu, model_5) {
  int model = 4;
  compare_distances_mnu(model, data);
}
