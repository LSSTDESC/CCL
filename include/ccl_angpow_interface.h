#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// CCL includes
#include "ccl_core.h"

// Angpow includes
/* #include <iostream> */
/* #include <fstream>  */
/* #include <string>  */
/* #include <vector> */
/* #include <numeric>  */
/* #include <math.h> */


/* #include "angpow_numbers.h" */
/* #include "angpow_func.h" */
#include "angpow_tools.h" 
/* #include "angpow_cosmo.h" */
#include "angpow_powspec_base.h"
#include "angpow_cosmo_base.h"
#include "angpow_radial_base.h"
#include "angpow_clbase.h"


// CCL inputs :
// comoving radial distance
// ccl_comoving_radial_distance
// P(k)

// growth ? RSD ? P(k,z) ?
// ccl_growth_factor
// ...
// redshift windows

// Angpow classes to feed
//CosmoCoordBase * coscoord;
//RadSelecBase * Z1win;
//RadSelecBase * Z2win;
//PowerSpecBase * pk;
//Clbase * clout;

// Write new classes PkCCL, CosmoCCL... that inherit from PkBase, CosmoBase

namespace Angpow {

class PowerSpecCCL : public PowerSpecBase {
 public:
  //! Constructor
  PowerSpecCCL(ccl_cosmology * cosmo, double a, double kmin, double kmax);
  //! Destructor
  virtual ~PowerSpecCCL() {}
  
  //! Used to delete explicitly the local pointers
  virtual void ExplicitDestroy() { 
    //if(mypGE_) delete mypGE_;
    if(Pk_) delete Pk_;
  }

  /*! Explicit to get a clone of the primary object via shallow copy
    using the Copy Ctor
   */
  virtual PowerSpecBase* clone() const {
    return new PowerSpecCCL(static_cast<const PowerSpecCCL&>(*this));
  }
  
  /*! called by angpow_kinteg.cc to fix the value of some function
    at fixed z value (and l too if necessayr)
   */
  void Init(int, r_8 z) {}

  //Main operator
  virtual r_8 operator()(int, r_8 k, r_8 z) {
     return (Pk_->operator()(k));
  }


 private:

  SLinInterp1D* Pk_;          //!< access to  Pk(k)
  //GrowthEisenstein* mypGE_;   //!< access tp  D(z)
  //r_8 growth2_;               //!< D(zi)^2

  //forbid for the time beeing the assignment operator
  PowerSpecCCL& operator=(const PowerSpecCCL& copy);
  
  //Minimal copy to allow Main operator(int, r_8, r_8) to work
  PowerSpecCCL(const PowerSpecCCL& copy) :
    Pk_(copy.Pk_) {} //, mypGE_(copy.mypGE_), growth2_(copy.growth2_)
};

}//end namespace


// Do the computation with :
//  Pk2Cl()
//  Compute()
