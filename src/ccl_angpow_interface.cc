#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// CCL includes
#include "ccl.h"

// Angpow includes
#include "angpow_cosmo_base.h"
#include "angpow_radial_base.h"
#include "angpow_powspec_base.h"
#include "angpow_clbase.h"


// Structure : some fonction that call for CCL quantities and output C_ell ?


// CCL inputs
// comoving radial distance
ccl_comoving_radial_distance
// P(k)

// growth ? RSD ? P(k,z) ?
ccl_growth_factor
...
// redshift windows

// Angpow classes to feed
CosmoCoordBase * coscoord;
RadSelecBase * Z1win;
RadSelecBase * Z2win;
PowerSpecBase * pk;
Clbase * clout;

CREER DES CLASSES PkCCL, CosmoCCL... qui heritent de PkBase, CosmoBase

  Pk2Cl()
  Compute()
