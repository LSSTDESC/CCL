#include "ccl.h"
#include "ctest.h"
#include <stdio.h>
#include <math.h>

// The tolerance in chi 
#define DISTANCES_TOLERANCE 1.0e-6

// We test the models CCL1-5 and CCL7-11 from the paper.
#define N_MODEL 10
// The test are done at 10 logarithmically spaced redshifts between 1e-2 and 1000.
#define N_Z 10

CTEST_DATA(distances_class) {
  double Omega_c;
  double Omega_b;
  double h;
  double A_s;
  double n_s;
  double Neff[N_MODEL];
  double mnu[N_MODEL][3];
  ccl_mnu_convention mnu_type;
  double Omega_k[N_MODEL];
  double w_0[N_MODEL];
  double w_a[N_MODEL];
  
  double z_chi[N_Z];
  double z_dm[N_Z];

  double chi_benchmark[N_Z][N_MODEL];
  double dm_benchmark[N_Z][N_MODEL];
};

// Read the fixed format file containing all the radial comoving
// distance benchmarks
static void read_benchmark_file(const char* filename, double z[N_Z], double benchmark[N_Z][N_MODEL])
{
  //Distances are in Mpc
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
        benchmark[i_z][i_model-1] = val;
      }
    }
  }
  fclose(f);
}

// Set up the cosmological parameters to be used in each of the
// models
CTEST_SETUP(distances_class) {
  // Values that are the same for all 5 models
  data->Omega_c = 0.25;
  data->Omega_b = 0.05;
  data->h = 0.7;
  data->A_s = 2.1e-9;
  data->n_s = 0.96;
  data->mnu_type = ccl_mnu_list;

  // Values that are different for the different models
  double Omega_k[N_MODEL] = {  0.0, 0.0, 0.0, 0.05, -0.05,
                               0.0, 0.0, 0.0, 0.05, -0.05 };
  double w_0[N_MODEL]     = { -1.0, -0.9, -0.9, -0.9, -0.9,
                              -1.0, -0.9, -0.9, -0.9, -0.9  };
  double w_a[N_MODEL]     = {  0.0,  0.0,  0.1,  0.1,  0.1,
                               0.0,  0.0,  0.1,  0.1,  0.1  };
  
  double mnu[N_MODEL][3]	= { {0.0,  0.0,  0.0},
                              {0.0,  0.0,  0.0},
                              {0.0,  0.0,  0.0},
                              {0.0,  0.0,  0.0},
                              {0.0,  0.0,  0.0},
                              {0.04, 0.0,  0.0},
                              {0.05, 0.01, 0.0},
                              {0.03, 0.02, 0.04},
                              {0.05, 0.0,  0.0},
                              {0.03, 0.02, 0.0} };

  double Neff[N_MODEL]   = {3.046, 3.046, 3.046, 3.046, 3.046,
                            3.013, 3.026, 3.040, 3.013, 3.026};
  

  // Fill in the values from these constant arrays.
  for(int i=0; i<N_MODEL; i++) {
    data->Omega_k[i] = Omega_k[i];
    data->w_0[i]     = w_0[i];
    data->w_a[i]     = w_a[i];
    data->Neff[i]    = Neff[i];
    for(int j=0; j<3; j++) {
      data->mnu[i][j]     = mnu[i][j];
    }
  }

  // The file of benchmark data.
  read_benchmark_file("./tests/benchmark/chi_class_allz.txt", data->z_chi, data->chi_benchmark);
  read_benchmark_file("./tests/benchmark/dm_class_allz.txt", data->z_dm, data->dm_benchmark);
}

static void compare_distances(int model, struct distances_class_data * data)
{
  int status=0;
  // Make the parameter set from the input data
  // Values of some parameters depend on the model index
  
  // The arrays of massive neutrions are different lengths, so we need to have an if-statement here to deal with that.
  
  ccl_parameters params;
  
  params = ccl_parameters_create(data->Omega_c, data->Omega_b, data->Omega_k[model],
						                     data->Neff[model], data->mnu[model], data->mnu_type, 
						                     data->w_0[model], data->w_a[model],
						                     data->h, data->A_s, data->n_s,
                                 -1,-1,-1,-1,NULL,NULL, &status);
  ASSERT_EQUAL(0, status);
  
  // Make a cosmology object from the parameters with the default configuration
  ccl_cosmology * cosmo = ccl_cosmology_create(params, default_config);
  ASSERT_NOT_NULL(cosmo);
  
  // Compare to benchmark data
  for(int i=0; i<N_Z; i++) {
    // Check comoving radial distance
    double a = 1/(1.+data->z_chi[i]);
    double chi_ccl = ccl_comoving_radial_distance(cosmo, a, &status);
    if(status) printf("%s\n",cosmo->status_message);

    double absolute_tolerance = DISTANCES_TOLERANCE*data->chi_benchmark[i][model];
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(data->chi_benchmark[i][model], chi_ccl, absolute_tolerance);

    // Check distance modulus
    a = 1/(1.+data->z_dm[i]);
    double dm_ccl = ccl_distance_modulus(cosmo, a, &status);
    if(status) printf("%s\n",cosmo->status_message);

    absolute_tolerance = DISTANCES_TOLERANCE*data->dm_benchmark[i][model];
    if (fabs(absolute_tolerance)<1e-12) absolute_tolerance = 1e-12;
    ASSERT_DBL_NEAR_TOL(data->dm_benchmark[i][model], dm_ccl, absolute_tolerance);
  }
  
  ccl_cosmology_free(cosmo);
}

CTEST2(distances_class, model_1) {
  int model = 0;
  compare_distances(model, data);
}

CTEST2(distances_class, model_2) {
  int model = 1;
  compare_distances(model, data);
}

CTEST2(distances_class, model_3) {
  int model = 2;
  compare_distances(model, data);
}

CTEST2(distances_class, model_4) {
  int model = 3;
  compare_distances(model, data);
}

CTEST2(distances_class, model_5) {
  int model = 4;
  compare_distances(model, data);
}

CTEST2(distances_class, model_7) {
  int model = 5;
  compare_distances(model, data);
}

CTEST2(distances_class, model_8) {
  int model = 6;
  compare_distances(model, data);
}

CTEST2(distances_class, model_9) {
  int model = 7;
  compare_distances(model, data);
}

CTEST2(distances_class, model_10) {
  int model = 8;
  compare_distances(model, data);
}

CTEST2(distances_class, model_11) {
  int model = 9;
  compare_distances(model, data);
}
