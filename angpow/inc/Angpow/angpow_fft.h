#ifndef ANGPOW_FFT_SEEN
#define ANGPOW_FFT_SEEN
/*
 *  This file is part of Angpow.
 *
 *  Angpow is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  Angpow is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Angpow; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  Angpow is being developed at the Linear Accelerateur Laboratory (LAL)
 *  91898 ORSAY CEDEX - FRANCE
 *  main author: J.E Campagne
 *    co-authors: J. Neveu, S. Plaszczynski
 */

#include <iostream>
#include <vector>

#include <omp.h>          //OpenMP for function sampling
#include <fftw3.h>        //Discrete Cosine Function

#include "angpow_exceptions.h" //exceptions
#include "angpow_numbers.h"    //r_8

namespace Angpow {

/*!
  Class to handle the Discrete Cosine Transform plan (type I) : here use of FFTW
*/

class FFTPlanning {
public:

  enum FFT_OPTION { FFT_FASTCTOR, FFT_OPTIM }; //JEC 17/10/16
  

  //JEC 28/10/16 Use only 1 thread (seems the best for the time beeing)

  static void Finalize() {
    fftw_cleanup();
  }


  FFTPlanning(int n, std::vector<r_8>& vec, FFT_OPTION opt= FFT_FASTCTOR) {
    CreatePlan(n,vec,opt);
  }
  ~FFTPlanning() { DestroyPlan(); }
    
  

  void Execute() const { fftw_execute(plan_); }
    
  void DestroyPlan() {
    if(plan_) {
      fftw_destroy_plan(plan_); 
      plan_=NULL;
    }
  }

  //JEC 28/10/16 make it private
private:  
  void CreatePlan(int n, std::vector<r_8>& vec,  FFT_OPTION opt =  FFT_FASTCTOR) {
    
    /* JEC 17/10/16 FFTW_PATIENT, 
       is like FFTW_MEASURE (default planning), but considers a wider range of algorithms and 
       often produces a more optimal plan (especially for large transforms), but at the expense 
       of several times longer planning time (especially for large transforms). 
       it will automatically disable threads for sizes that don't benefit from parallelization.
           
    */

    //------------- MAJOR 
    //JEC 28/10/16 : restore switch AND set to "n" the dimension 
    //------------- MAJOR 

    switch (opt) {
    case  FFT_FASTCTOR: 
      plan_ = fftw_plan_r2r_1d(n,&vec.data()[0], &vec.data()[0], FFTW_REDFT00, FFTW_ESTIMATE);
      break;
    case FFT_OPTIM:
      plan_ = fftw_plan_r2r_1d(n,&vec.data()[0], &vec.data()[0], FFTW_REDFT00, FFTW_PATIENT);
      break;
    }//sw
    if(plan_ == NULL)
      throw AngpowError("FATAL: CreatePlan failed!");
  }
private:
  fftw_plan plan_;

};

}//end namespace
#endif //ANGPOW_FFT_SEEN
