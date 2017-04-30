#include <string>
#include <vector>
#include "Angpow/angpow_radint.h"    //radial integration
#include "Angpow/angpow_utils.h"
#include "Angpow/angpow_quadinteg.h" //intgegration by quadrature

namespace Angpow {


void Quadrature1D::SetQuadrature(RadSelectBase* win) {
  
  
  DiracSelect* diracW = dynamic_cast<DiracSelect*>(win);

  if(diracW != 0) {
    std::cout << "Dirac Select with nodes at " << diracW->GetZval() << std::endl;

    NRval_ = 1;
    DeltaZ_ = 0;   //<- Used ???
    xQuad_.resize(1); xQuad_[0] = diracW->GetZval();
    wQuad_.resize(1); wQuad_[0] = 1.0;

  } else {

    if(quadFile_ == "")throw AngpowError("SetQuadrature: quadFile not set");    
    ifstream ifs(quadFile_);
    bool opt=true;
    if(ifs.is_open()){
      ifs.close(); //close it before as it will be reopenned by the Quadrature Ctor
      opt=false;
    }

    ClenshawCurtisQuadrature<r_8> theQuad(norder_,quadFile_,opt); //opt=false=do not recompute the weights and nodes of the quadrature

    xQuad_ = theQuad.GetAbscissa();
    wQuad_ = theQuad.GetAbscissaW();
    
    if( (NRval_ != -1) &&  ((size_t)NRval_ != xQuad_.size()) )
      throw AngpowError("SetQuadrature: missmatch quadrature size");
    
    r_8 zmin = win->GetZMin();
    r_8 zmax = win->GetZMax();
    
    
    NRval_ = xQuad_.size();
    DeltaZ_ = zmax - zmin;
    
    for(int ixQ=0; ixQ<NRval_; ixQ++){
      xQuad_[ixQ] = zmin+xQuad_[ixQ]*DeltaZ_;
      wQuad_[ixQ] *= DeltaZ_; 
    }
  }//window type

}//SetQuadrature



void Quadrature2D::SetQuadrature(RadSelectBase* winI, RadSelectBase* winJ){
  
  Quadrature1D quadI(norderI_); quadI.SetQuadrature(winI);
  Quadrature1D quadJ(norderJ_); quadJ.SetQuadrature(winJ);

  DeltaZI_ = quadI.GetDeltaZ();
  DeltaZJ_ = quadJ.GetDeltaZ();

  NRvalI_ = quadI.GetNRval();
  NRvalJ_ = quadJ.GetNRval();

  xI_ = quadI.GetNodes();
  xJ_ = quadJ.GetNodes();

  std::vector<r_8> wI = quadI.GetWeights();
  std::vector<r_8> wJ = quadI.GetWeights();

  int NRvals = NRvalI_ * NRvalJ_;

  wQuad_.resize(NRvals);
  int ij=0;
  for(int i=0; i<NRvalI_; i++){
    for(int j=0; j<NRvalJ_; j++){
      wQuad_[ij] = wI[i] * wJ[j];
      ij++;
    }
  }
  
}//SetQuadrature

  


void Radial2DIntegrator::SetQuadrature(RadSelectBase* winI, RadSelectBase* winJ) {
  winI_ = winI;
  winJ_ = winJ;

  //delagate to Quadrature2D the computation of nodes & weights
  Quadrature2D quad(norderI_, norderJ_);
  quad.SetQuadrature(winI, winJ);

  //joined quad
  wQuad_ = quad.GetWeights();
  //indiv nodes
  xI_    = quad.GetInodes();
  xJ_    = quad.GetJnodes();
  
  NRvalI_ = quad.GetNRvalI();
  NRvalJ_ = quad.GetNRvalJ();

  DeltaZI_ = quad.GetDeltaZI();
  DeltaZJ_ = quad.GetDeltaZJ();
    
}//SetQuadrature


void Radial2DIntegrator::ComputeSuperW(std::vector<r_8>& winW){
  int NRvals =  NRvalI_ * NRvalJ_;

  winW.resize(NRvals);

  std::vector<r_8> winI(NRvalI_);
  for(int i=0; i<NRvalI_; i++) winI[i]=(*winI_)(xI_[i]);
  std::vector<r_8> winJ(NRvalJ_);
  for(int j=0; j<NRvalJ_; j++) winJ[j]=(*winJ_)(xJ_[j]);


  int ij=0;
  r_8 norm=0;
  for(int i=0; i<NRvalI_; i++){
    for(int j=0; j<NRvalJ_; j++){
      winW[ij] = winI[i]  * winJ[j] * wQuad_[ij]; 
      norm += winW[ij];
      ij++;
    }
  }

  //normalization
  r_8 invnorm = 1./norm;
  for(int i=0;i<NRvals; i++)winW[i] *= invnorm;


}//ComputeSuperW



}//namespace
