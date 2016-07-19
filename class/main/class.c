/** @file class.c
 * Julien Lesgourgues, 17.04.2011
 */

#include "class.h"

int main(int argc, char **argv) {

  struct precision pr;        /* for precision parameters */
  struct background ba;       /* for cosmological background */
  struct thermo th;           /* for thermodynamics */
  struct perturbs pt;         /* for source functions */
  struct transfers tr;        /* for transfer functions */
  struct primordial pm;       /* for primordial spectra */
  struct spectra sp;          /* for output spectra */
  struct nonlinear nl;        /* for non-linear spectra */
  struct lensing le;          /* for lensed spectra */
  struct output op;           /* for output files */
  ErrorMsg errmsg;            /* for error messages */
  struct file_content fc;	

  if (parser_init(&fc,15,"none",errmsg) == _FAILURE_){
    printf("\n\nparser_init\n=>%s\n",errmsg);
    return _FAILURE_;  	
  }
  strcpy(fc.name[0],"h");
  strcpy(fc.value[0],"0.7");

  strcpy(fc.name[1],"Omega_cdm");
  strcpy(fc.value[1],"0.25");

  strcpy(fc.name[2],"Omega_b");
  strcpy(fc.value[2],"0.04");

  strcpy(fc.name[3],"Omega_k");
  strcpy(fc.value[3],"0.0");

  // set Omega_Lambda = 0.0 if w !=-1
  strcpy(fc.name[4],"Omega_Lambda");
  strcpy(fc.value[4],"0.0");

  strcpy(fc.name[5],"w0_fld");
  strcpy(fc.value[5],"-.99");

  strcpy(fc.name[6],"wa_fld");
  strcpy(fc.value[6],"0.0");

  strcpy(fc.name[7],"n_s");
  strcpy(fc.value[7],"0.0");

  strcpy(fc.name[8],"A_s");
  strcpy(fc.value[8],"2.215e-9");

  strcpy(fc.name[9],"output");
  strcpy(fc.value[9],"mPk");

  strcpy(fc.name[10],"non linear");
  strcpy(fc.value[10],"Halofit");

  strcpy(fc.name[11],"P_k_max_1/Mpc");
  strcpy(fc.value[11],"100.");

  strcpy(fc.name[12],"z_max_pk");
  strcpy(fc.value[12],"10.");

  strcpy(fc.name[13],"modes");
  strcpy(fc.value[13],"s");

  strcpy(fc.name[14],"lensing");
  strcpy(fc.value[14],"no");


  if (input_init(&fc,&pr,&ba,&th,&pt,&tr,&pm,&sp,&nl,&le,&op,errmsg) == _FAILURE_) {
    printf("\n\nError running input_init\n=>%s\n",errmsg);
    return _FAILURE_;
  }

  if (background_init(&pr,&ba) == _FAILURE_) {
    printf("\n\nError running background_init \n=>%s\n",ba.error_message);
    return _FAILURE_;
  }

  if (thermodynamics_init(&pr,&ba,&th) == _FAILURE_) {
    printf("\n\nError in thermodynamics_init \n=>%s\n",th.error_message);
    return _FAILURE_;
  }

  if (perturb_init(&pr,&ba,&th,&pt) == _FAILURE_) {
    printf("\n\nError in perturb_init \n=>%s\n",pt.error_message);
    return _FAILURE_;
  }

  if (primordial_init(&pr,&pt,&pm) == _FAILURE_) {
    printf("\n\nError in primordial_init \n=>%s\n",pm.error_message);
    return _FAILURE_;
  }

  if (nonlinear_init(&pr,&ba,&th,&pt,&pm,&nl) == _FAILURE_) {
    printf("\n\nError in nonlinear_init \n=>%s\n",nl.error_message);
    return _FAILURE_;
  }

   if (transfer_init(&pr,&ba,&th,&pt,&nl,&tr) == _FAILURE_) {
     printf("\n\nError in transfer_init \n=>%s\n",tr.error_message);
     return _FAILURE_;
   }

   if (spectra_init(&pr,&ba,&pt,&pm,&nl,&tr,&sp) == _FAILURE_) {
     printf("\n\nError in spectra_init \n=>%s\n",sp.error_message);
     return _FAILURE_;
   }
   double Z, ic;
  int i = spectra_pk_at_k_and_z(&ba, &pm, &sp,0.001,0.0, &Z,&ic);
  printf("%e\n",Z);
  i = spectra_pk_nl_at_k_and_z(&ba, &pm, &sp,0.001,0.0, &Z);
  printf("%e\n",Z);

  // if (lensing_init(&pr,&pt,&sp,&nl,&le) == _FAILURE_) {
  //   printf("\n\nError in lensing_init \n=>%s\n",le.error_message);
  //   return _FAILURE_;
  // }

  // if (output_init(&ba,&th,&pt,&pm,&tr,&sp,&nl,&le,&op) == _FAILURE_) {
  //   printf("\n\nError in output_init \n=>%s\n",op.error_message);
  //   return _FAILURE_;
  // }

  // /****** all calculations done, now free the structures ******/

  // if (lensing_free(&le) == _FAILURE_) {
  //   printf("\n\nError in lensing_free \n=>%s\n",le.error_message);
  //   return _FAILURE_;
  // }

   if (spectra_free(&sp) == _FAILURE_) {
     printf("\n\nError in spectra_free \n=>%s\n",sp.error_message);
     return _FAILURE_;
   }

   if (transfer_free(&tr) == _FAILURE_) {
     printf("\n\nError in transfer_free \n=>%s\n",tr.error_message);
     return _FAILURE_;
   }

  if (nonlinear_free(&nl) == _FAILURE_) {
    printf("\n\nError in nonlinear_free \n=>%s\n",nl.error_message);
    return _FAILURE_;
  }

  if (primordial_free(&pm) == _FAILURE_) {
    printf("\n\nError in primordial_free \n=>%s\n",pm.error_message);
    return _FAILURE_;
  }

  if (perturb_free(&pt) == _FAILURE_) {
    printf("\n\nError in perturb_free \n=>%s\n",pt.error_message);
    return _FAILURE_;
  }

  if (thermodynamics_free(&th) == _FAILURE_) {
    printf("\n\nError in thermodynamics_free \n=>%s\n",th.error_message);
    return _FAILURE_;
  }

  if (background_free(&ba) == _FAILURE_) {
    printf("\n\nError in background_free \n=>%s\n",ba.error_message);
    return _FAILURE_;
  }

  return _SUCCESS_;

}
