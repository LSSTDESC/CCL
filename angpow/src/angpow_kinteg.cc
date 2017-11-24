#include <algorithm>
#include "Angpow/angpow_kinteg.h"
#include "Angpow/angpow_parameters.h"
#include "Angpow/angpow_powspec_base.h" 
#include "Angpow/angpow_integrand_base.h" 
#include "Angpow/angpow_clbase.h"

namespace Angpow {
  


KIntegrator::KIntegrator(const std::vector<r_8>& radiusI, const std::vector<r_8>& radiusJ,
	      const std::vector<r_8>& zI, const std::vector<r_8>& zJ,
	      const std::vector<r_8>& winW,
	      int NRvalI, int NRvalJ, int Lmax, 
	      int iOrd1, int iOrd2, int nRootPerInt, 
	      r_8 kMax): 
  nRootPerInt_(nRootPerInt),
  kMax_(kMax),
  NRvalI_(NRvalI),
  NRvalJ_(NRvalJ),    
  Lmax_(Lmax),
  Ri_(radiusI),
  Rj_(radiusJ),
  zi_(zI),
  zj_(zJ),
  winW_(winW)
{

  same_sampling_=std::equal(zi_.begin(),zi_.end(),zj_.begin());
  //  if(same_sampling_)std::cout<<"KIntegrator: Same redshift sampling detected" << std::endl;
  

#ifdef PROFILING
  tstack_push("ChebyshevInt Ctor...");  
#endif

#ifdef _OPENMP
  size_t MAXTHREADS=omp_get_max_threads();
#else
  size_t MAXTHREADS=1;
#endif
  for (size_t i=0;i<MAXTHREADS;i++) 
    cheInts_.push_back(new ChebyshevInt(iOrd1, iOrd2));

#ifdef PROFILING
  tstack_pop("ChebyshevInt Ctor...");  
#endif

 
  NRvals_ = NRvalI_*NRvalJ_;


}//Ctor



void KIntegrator::Compute(PowerSpecBase& pws, Clbase& clout){

  //Get information from control parameters
  Parameters para = Param::Instance().GetParam();
  r_8 total_weight_cut = para.total_weight_cut;
  bool has_deltaR_cut  = para.has_deltaR_cut;
  r_8 deltaR_cut       = para.deltaR_cut;

  //Bessel roots 

  //JEC 20/4/17 use the mean of all R_i and R_j 
  //r_8 RcurMax = std::max(Ri_.back(), Rj_.back());   // max(max(Ri),max(Rj))
  r_8 tmp1 = std::accumulate(Ri_.begin(), Ri_.end(), 0.);
  r_8 tmp2 = std::accumulate(Rj_.begin(), Rj_.end(), tmp1);
  r_8 RcurMax = tmp2/((r_8)(Ri_.size()+Rj_.size())); //Nb: RcurMax is the mean not the max but we do not change yet the variable naming

  //  std::cout << ">>>>> RcurMax: " << RcurMax << ", old = " <<  std::max(Ri_.back(), Rj_.back()) << std::endl;

  int Pmaxmax = (kMax_*RcurMax)/M_PI;
    
  BesselRoot broots(Lmax_,Pmaxmax,nRootPerInt_);
  int NbessRoots = broots.NRootsL(); //last bessel root is NbessRoots*nRootPerInt
    
    
  r_8 Rcur = RcurMax;
  r_8 kscale = 1./Rcur;
  
  int tid;

#pragma omp parallel shared(clout)
  {
#pragma omp for schedule(dynamic,1) private(tid) 
  for(int index_l=0; index_l<clout.Size(); index_l++){
    
    int l=clout[index_l].first;

#ifdef _OPENMP
    tid=omp_get_thread_num();
#else
    tid=0;
#endif
    
    //printf("l=%d\tthread=%d : running \n",l,tid);

    PowerSpecBase* pwsPTR = pws.clone();
    
    int maxRoots = NbessRoots;

    
    std::vector<r_8> qlp; broots.GetVecRoots(qlp, l);	

    //get the last value = kMax
    
    r_8 kLast = qlp.back()*kscale;
    while(kLast>kMax_ && !qlp.empty()){
      qlp.pop_back();
      kLast = qlp.back()/Rcur;
    }
    if(kLast<kMax_){
      qlp.push_back(Rcur*kMax_);
    }
    kLast = qlp.back()*kscale;
    
    //final k-integral bounds
    std::vector<r_8> klp;
    klp.push_back(BesselJImp::Xmin(l));
    klp.insert(klp.end(),qlp.begin(),qlp.end());
    std::transform(klp.begin(),klp.end(),klp.begin(),std::bind1st(std::multiplies<r_8>(),kscale));

    maxRoots = klp.size();

    r_8 cl_value= 0;

    for(int p = 1; p<maxRoots; p++){ //init at p=1

      r_8 lowBound = klp[p-1];
      r_8 uppBound = klp[p];
      
      if(lowBound > uppBound)
	throw AngpowError("KIntegrator::Compute uppBound < lowBound Fatal");

      std::vector< std::vector<r_8> > ChebTrans1(NRvalI_);
      std::vector<bool> done1(NRvalI_,false); //vector of sampling demand
      std::vector< std::vector<r_8> > ChebTrans2(NRvalJ_);
      std::vector<bool> done2(NRvalJ_,false); //vector of sampling demand

#ifdef PROFILING
      tstack_push("Integral Top....");
#endif

      if(same_sampling_) { //symetric sampling (Auto correlation)
	if(NRvalI_ != NRvalJ_)
	  throw AngpowError("angpow_kinteg.c: FATAL same_sampling but NRvalI != NRvalJ");
	int ij=0;
	r_8 tmp=0;
	for(int i=0; i<NRvalI_; i++){
	  //diagonal term
	  ij=i*(NRvalI_+1);

	  if(fabs(winW_[ij])<total_weight_cut) continue; //TEST 21/10/16
	  
#ifdef PROFILING
	  tstack_push("Sampling 1....");
#endif
	  //Test STart
	  r_8 Rcuri = Ri_[i];
	  if(!done1[i]){
	    JBess1 jfunc(l,Rcuri);
	    pwsPTR->Init(zi_[i]);
	    PowSqrtJBess* f1 = new PowSqrtJBess(pwsPTR, &jfunc, l, zi_[i]);
	    cheInts_[tid]->ChebyshevTransform(f1, NULL, ChebTrans1[i], lowBound, uppBound);
	    delete f1;	    
	    done1[i]= true;
	    ChebTrans2[i] = ChebTrans1[i]; //To be optimzed later (use a single vector)
	    done2[i]= true;
	  }
	  //Test End
#ifdef PROFILING
	  tstack_pop("Sampling 1....");
#endif

#ifdef PROFILING
	  tstack_push("Integral 1....");
#endif
	  tmp = cheInts_[tid]->ComputeIntegral(ChebTrans1[i], ChebTrans2[i], lowBound, uppBound);	  
	  cl_value += tmp * winW_[ij];
	  //if(l<10)
	  //  std::cout<<l<<" "<<zi_[i]<<" "<<winW_[ij]<<" "<<lowBound<<" "<<uppBound<<" "<<tmp<<" "<<cl_value<<std::endl;
#ifdef PROFILING
	  tstack_pop("Integral 1....");
#endif

	  //upper triangle part with weight 2
	  for(int j=i+1; j<NRvalI_; j++){    
	    ij = i*NRvalI_+j;

	    //Cutting
	    if(fabs(winW_[ij])<total_weight_cut) continue; //TEST 21/10/16
	    if(has_deltaR_cut && (fabs(Ri_[j]-Ri_[i])>deltaR_cut) ) continue;

	    //Test STart
#ifdef PROFILING
	    tstack_push("Sampling 2....");
#endif


	    r_8 Rcurj = Rj_[j];
	    if(!done2[j]){
	      JBess1 jfunc(l,Rcurj);
	      pwsPTR->Init(zj_[j]);
	      PowSqrtJBess* f2 = new PowSqrtJBess(pwsPTR, &jfunc, l, zj_[j]);
	      cheInts_[tid]->ChebyshevTransform(NULL, f2, ChebTrans2[j], lowBound, uppBound);
	      delete f2;	      
	      done2[j]= true;

	      ChebTrans1[j] = ChebTrans2[j]; //To be optimzed later (use a single vector)
	      done1[j]= true;
	    }
#ifdef PROFILING
	    tstack_pop("Sampling 2....");
#endif
	  //Test End


#ifdef PROFILING
	    tstack_push("Integral 2....");
#endif
	    tmp = cheInts_[tid]->ComputeIntegral(ChebTrans1[i], ChebTrans2[j], lowBound, uppBound);
	    cl_value += 2.* tmp * winW_[ij];
#ifdef PROFILING
	    tstack_pop("Integral 2....");
#endif
	  }//loop-j
	}//loop-i
	
      } else { //non-symetric sampling (Cross correlation)
	int ij=0;
	for(int i=0; i<NRvalI_; i++){
	  r_8 Rcuri = Ri_[i];
	  for(int j=0; j<NRvalJ_; j++){    
	    r_8 Rcurj = Rj_[j];
	    
	    r_8 wij = winW_[ij]; ij++;

	    // Cutting 
 	    if(fabs(wij)<total_weight_cut) continue;
	    if(has_deltaR_cut && (fabs(Rcurj-Rcuri)>deltaR_cut) ) continue;

#ifdef PROFILING
	    tstack_push("Sampling....");
#endif
	    if(!done1[i]){
	      JBess1 jfunc(l,Rcuri);
	      pwsPTR->Init(zi_[i]);
	      PowSqrtJBess* f1 = new PowSqrtJBess(pwsPTR, &jfunc, l, zi_[i]);
	      cheInts_[tid]->ChebyshevTransform(f1, NULL, ChebTrans1[i], lowBound, uppBound);
	      delete f1;
	      done1[i]= true;
	    }


	    if(!done2[j]){
	      JBess1 jfunc(l,Rcurj);
	      pwsPTR->Init(zj_[j]);
	      PowSqrtJBess* f2 = new PowSqrtJBess(pwsPTR, &jfunc, l, zj_[j]);
	      cheInts_[tid]->ChebyshevTransform(NULL, f2, ChebTrans2[j], lowBound, uppBound);
	      delete f2;
	      done2[j]= true;
	    }

#ifdef PROFILING
	    tstack_pop("Sampling....");    
	    tstack_push("Integral....");
#endif

	    r_8 tmp = cheInts_[tid]->ComputeIntegral(ChebTrans1[i], ChebTrans2[j], lowBound, uppBound);

	    cl_value += tmp * wij;

#ifdef PROFILING
	    tstack_pop("Integral....");
#endif	    
	  }//Rj-loop
	}//Ri-loop
      }

#ifdef PROFILING
      tstack_pop("Integral Top....");
#endif
    }//p-loop        

    clout[index_l].second = cl_value;

    //JEC 24/11/16 delete clone ptr
    delete pwsPTR;


  }//ell-loop
  } //omp
}//compute









  


void KIntegrator::Compute(IntegrandBase& int1, IntegrandBase& int2, Clbase& clout){

  //Get information from control parameters
  Parameters para = Param::Instance().GetParam();
  r_8 total_weight_cut = para.total_weight_cut;
  bool has_deltaR_cut  = para.has_deltaR_cut;
  r_8 deltaR_cut       = para.deltaR_cut;

  //Bessel roots 

  //JEC 20/4/17 use the mean of all R_i and R_j 
  //r_8 RcurMax = std::max(Ri_.back(), Rj_.back());   // max(max(Ri),max(Rj))
  r_8 tmp1 = std::accumulate(Ri_.begin(), Ri_.end(), 0.);
  r_8 tmp2 = std::accumulate(Rj_.begin(), Rj_.end(), tmp1);
  r_8 RcurMax = tmp2/((r_8)(Ri_.size()+Rj_.size())); //Nb: RcurMax is the mean not the max but we do not change yet the variable naming

  //  std::cout << ">>>>> RcurMax: " << RcurMax << ", old = " <<  std::max(Ri_.back(), Rj_.back()) << std::endl;

  int Pmaxmax = (kMax_*RcurMax)/M_PI;
    
  BesselRoot broots(Lmax_,Pmaxmax,nRootPerInt_);
  int NbessRoots = broots.NRootsL(); //last bessel root is NbessRoots*nRootPerInt
    
    
  r_8 Rcur = RcurMax;
  r_8 kscale = 1./Rcur;
  
  int tid;
  
  //JEC 22/4/17 test to define ptr before OMP loop
  std::vector<IntegrandBase*> int1PTR;
  std::vector<IntegrandBase*> int2PTR;
#ifdef _OPENMP
  size_t MAXTHREADS=omp_get_max_threads();
#else
  size_t MAXTHREADS=1;
#endif
  for (size_t i=0;i<MAXTHREADS;i++) {
    int1PTR.push_back(int1.clone());
    int2PTR.push_back(int2.clone());
  }



#pragma omp parallel shared(clout)
  {
#pragma omp for schedule(dynamic,1) private(tid) 
  for(int index_l=0; index_l<clout.Size(); index_l++){
    int l=clout[index_l].first;

#ifdef _OPENMP
    tid=omp_get_thread_num();
    size_t MAXTHREADS=omp_get_max_threads();
#else
    tid=0;
    size_t MAXTHREADS=1;
#endif
    
        
    int maxRoots = NbessRoots;

    
    std::vector<r_8> qlp; broots.GetVecRoots(qlp, l);	

    //get the last value = kMax
    
    r_8 kLast = qlp.back()*kscale;
    while(kLast>kMax_ && !qlp.empty()){
      qlp.pop_back();
      kLast = qlp.back()/Rcur;
    }
    if(kLast<kMax_){
      qlp.push_back(Rcur*kMax_);
    }
    kLast = qlp.back()*kscale;
    
    //final k-integral bounds
    std::vector<r_8> klp;
    klp.push_back(BesselJImp::Xmin(l));
    klp.insert(klp.end(),qlp.begin(),qlp.end());
    std::transform(klp.begin(),klp.end(),klp.begin(),std::bind1st(std::multiplies<r_8>(),kscale));

	
    maxRoots = klp.size();

    r_8 cl_value= 0;

    for(int p = 1; p<maxRoots; p++){ //init at p=1

      r_8 lowBound = klp[p-1];
      r_8 uppBound = klp[p];

      //      printf("l=%d\tthread=%d : %12.6f %12.6f\n",l,tid,lowBound,uppBound);

      if(lowBound > uppBound)
	throw AngpowError("KIntegrator::Compute uppBound < lowBound Fatal");

      std::vector< std::vector<r_8> > ChebTrans1(NRvalI_);
      std::vector<bool> done1(NRvalI_,false); //vector of sampling demand
      std::vector< std::vector<r_8> > ChebTrans2(NRvalJ_);
      std::vector<bool> done2(NRvalJ_,false); //vector of sampling demand

#ifdef PROFILING
      tstack_push("Integral Top....");
#endif

      if(same_sampling_) { //symetric sampling (Auto correlation)

	if(NRvalI_ != NRvalJ_)
	  throw AngpowError("angpow_kinteg.c: FATAL same_sampling but NRvalI != NRvalJ");
	int ij=0;
	r_8 tmp=0;
	for(int i=0; i<NRvalI_; i++){
	  //diagonal term
	  ij=i*(NRvalI_+1);

	  if(fabs(winW_[ij])<total_weight_cut) continue;
	  
#ifdef PROFILING
	  tstack_push("Sampling 1....");
#endif
	  if(!done1[i]){

	    int1PTR[tid]->Init(l, zi_[i]); //JEC tid 22/4/17 // JN 20/04/2017 réinialise l'integrand a z et ell
	    cheInts_[tid]->ChebyshevTransform(int1PTR[tid], NULL, ChebTrans1[i], lowBound, uppBound);
	    int1PTR[tid]->ExplicitDestroy(); //JEC tid 22/4/17

	    done1[i]= true;
	    ChebTrans2[i] = ChebTrans1[i]; //To be optimzed later (use a single vector)
	    done2[i]= true;

	  }
#ifdef PROFILING
	  tstack_pop("Sampling 1....");
#endif

#ifdef PROFILING
	  tstack_push("Integral 1....");
#endif
	  tmp = cheInts_[tid]->ComputeIntegral(ChebTrans1[i], ChebTrans2[i], lowBound, uppBound);	  
	  cl_value += tmp * winW_[ij];
	  //if(l<10)
	  //  std::cout<<l<<" "<<zi_[i]<<" "<<winW_[ij]<<" "<<lowBound<<" "<<uppBound<<" "<<tmp<<" "<<cl_value<<std::endl;
#ifdef PROFILING
	  tstack_pop("Integral 1....");
#endif

	  //upper triangle part with weight 2
	  for(int j=i+1; j<NRvalI_; j++){    
	    ij = i*NRvalI_+j;

	    //Cutting
	    if(fabs(winW_[ij])<total_weight_cut) continue; //TEST 21/10/16
	    if(has_deltaR_cut && (fabs(Ri_[j]-Ri_[i])>deltaR_cut) ) continue;

	    //Test STart
#ifdef PROFILING
	    tstack_push("Sampling 2....");
#endif


	    //r_8 Rcurj = Rj_[j];
	    if(!done2[j]){
	      int2PTR[tid]->Init(l,zj_[j]); //JEC 22/4/17 tid  // JN 20/04/2017 réinialise l'integrand a z et ell
	      cheInts_[tid]->ChebyshevTransform(NULL, int2PTR[tid], ChebTrans2[j], lowBound, uppBound);
	      int2PTR[tid]->ExplicitDestroy(); //JEC 22/4/17
	      done2[j]= true;
	      ChebTrans1[j] = ChebTrans2[j]; //To be optimzed later (use a single vector)
	      done1[j]= true;
	    }
#ifdef PROFILING
	    tstack_pop("Sampling 2....");
#endif
	  //Test End

#ifdef PROFILING
	    tstack_push("Integral 2....");
#endif
	    tmp = cheInts_[tid]->ComputeIntegral(ChebTrans1[i], ChebTrans2[j], lowBound, uppBound);
	    cl_value += 2.* tmp * winW_[ij];
#ifdef PROFILING
	    tstack_pop("Integral 2....");
#endif
	  }//loop-j
	}//loop-i
	
      } else { //non-symetric sampling (Cross correlation)

	int ij=0;
	for(int i=0; i<NRvalI_; i++){
	  r_8 Rcuri = Ri_[i];
	  for(int j=0; j<NRvalJ_; j++){    
	    r_8 Rcurj = Rj_[j];
	    
	    r_8 wij = winW_[ij]; ij++;

	    // Cutting 
 	    if(fabs(wij)<total_weight_cut) continue;
	    if(has_deltaR_cut && (fabs(Rcurj-Rcuri)>deltaR_cut) ) continue;

#ifdef PROFILING
	    tstack_push("Sampling....");
#endif
	    if(!done1[i]){

	      int1PTR[tid]->Init(l,zi_[i]); //JEC 22/4/17 tid  // JN 20/04/2017 réinialise l'integrand a z et ell
	      cheInts_[tid]->ChebyshevTransform(int1PTR[tid], NULL, ChebTrans1[i], lowBound, uppBound);
	      int1PTR[tid]->ExplicitDestroy(); //JEC 22/4/17 tid

	      done1[i]= true;
	    }


	    if(!done2[j]){
	      
	      int2PTR[tid]->Init(l,zj_[j]);//JEC 22/4/17 tid  // JN 20/04/2017 réinialise l'integrand a z et ell
	      cheInts_[tid]->ChebyshevTransform(NULL, int2PTR[tid], ChebTrans2[j], lowBound, uppBound);
 	      int2PTR[tid]->ExplicitDestroy(); //JEC 22/4/17 tid

	      done2[j]= true;
	    }

#ifdef PROFILING
	    tstack_pop("Sampling....");    
	    tstack_push("Integral....");
#endif

	    r_8 tmp = cheInts_[tid]->ComputeIntegral(ChebTrans1[i], ChebTrans2[j], lowBound, uppBound);

	    cl_value += tmp * wij;

#ifdef PROFILING
	    tstack_pop("Integral....");
#endif	    
	  }//Rj-loop
	}//Ri-loop
      }

#ifdef PROFILING
      tstack_pop("Integral Top....");
#endif
    }//p-loop        

    clout[index_l].second = cl_value;


  }//ell-loop
  } //omp

  //JEC 22/4/17
  for (size_t i=0;i<int1PTR.size();i++) delete int1PTR[i]; 
  for (size_t i=0;i<int2PTR.size();i++) delete int2PTR[i];

}//compute


  
}//namespace
