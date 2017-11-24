#include "Angpow/angpow_tools.h"

#include <iostream>
#include <algorithm>
#include <iterator>
#include <assert.h>
#include <math.h> //pow

#include "Angpow/angpow_exceptions.h"


namespace Angpow {

void getLlist(int l_min, int l_max, int l_linstep, double l_logstep, 
	      std::vector<int>&l){

  int index_l,increment,current_l, l_size;

  /** - start from l = l_min and increase with logarithmic step */

  index_l = 0;
  current_l = l_min;
  
  increment = std::max((int)(current_l * (l_logstep-1.)),1);
    
  while ((current_l < l_max) && (increment < l_linstep)) {
      
    index_l ++;
    current_l += increment;
    increment = std::max((int)(current_l * (l_logstep-1.)),1);

  }

  /** - when the logarithmic step becomes larger than some linear step, 
      stick to this linear step till l_max */

  increment = l_linstep;

  while (current_l < l_max) {

    index_l ++;
    current_l += increment;

  }

  /** - last value set to exactly l_max */

  l_size = index_l+1;

  /** - so far we just counted the number of values. Now repeat the
      whole thing but fill array with values. */

  l.resize(l_size,0);

  index_l = 0;
  current_l = l_min;

  increment = std::max((int)(current_l * (l_logstep-1.)),1);

  while ((current_l < l_max) && (increment < l_linstep)) {

    l[index_l]=current_l;
    index_l ++;
    current_l+=increment;
    increment = std::max((int)(current_l * (l_logstep-1.)),1);
     
  }

  increment = l_linstep;

  while (current_l < l_max) {

    l[index_l]=current_l;
    index_l ++;
    current_l+=increment;
 
  }

  l[index_l]=current_l;

}//getLlist


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*!
  Createur pour spline 3 sur "x[0->n],y[0->n]" avec "yp1,ypn" derivees
  au premier et dernier points et "natural" indiquant les types de
  contraintes sur les derivees 2sd au premier et dernier point.
  "order" doit etre mis a "true" si le tableau de "x[]" n'est pas ordonne
  dans l'ordre des "x" croissants ("x[i]<x[i+1]"): cette option
  realloue la place pour les tableaux "x,y" autrement seule une
  connection aux tableaux "x,y" externes est realisee.
*/
CSpline::CSpline(int n,double* x,double* y,double yp1,double ypn
                ,int natural,bool order)
  : Nel(0), corrupt_Y2(true), XY_Created(false), Natural(natural)
  , YP1(yp1), YPn(ypn), X(NULL), Y(NULL), Y2(NULL), tmp(NULL)
{
SetNewTab(n,x,y,order,true);
if( x != NULL && y != NULL) ComputeCSpline();

}


//!	Createur par defaut.
CSpline::CSpline(double yp1,double ypn,int natural)
  : Nel(0), corrupt_Y2(true), XY_Created(false), Natural(natural)
  , YP1(yp1), YPn(ypn), X(NULL), Y(NULL), Y2(NULL), tmp(NULL)
{
}



//////////////////////////////////////////////////////////////////////////////
//! destructeur
CSpline::~CSpline()
{
DelTab();
}

/*!
  Pour changer les tableaux sans recreer la classe,
  memes arguments que dans le createur.
  Pour connecter les tableaux "x[n],y[n]" aux pointeurs internes "X,Y"
  Si "order=true", on considere que x n'est pas range par ordre
  des "x" croissants. La methode alloue de la place pour des tableaux
  internes "X,Y" qu'elle re-ordonne par "x" croissant.
  "force=true" impose la reallocation des divers buffers, sinon
  la reallocation n'a lieu que si le nombre de points augmente.
*/
void CSpline::SetNewTab(int n,double* x,double* y,bool order,bool force)
{
assert( n>3 );

// allocation des buffers Y2 et tmp
if( n>Nel || force ) {
  if( Y2  != NULL ) {delete [] Y2; Y2=NULL;}
  if( tmp != NULL ) {delete [] tmp; tmp=NULL;}
  Y2   = new double[n];
  tmp  = new double[n];
}
// des-allocation eventuelle de X,Y
if( XY_Created ) {
  if( !order || n>Nel || force ) {
    if( X != NULL ) delete [] X;  X = NULL;
    if( Y != NULL ) delete [] Y;  Y = NULL;
    XY_Created = false;
  }
}
// allocation eventuelle de X,Y
if( order ) {
  if( !XY_Created || n>Nel || force ) {
    X = new double[n];
    Y = new double[n];
    XY_Created = true;
  }
}
Nel = n;
corrupt_Y2 = true;

if( x==NULL || y==NULL ) return;

// Classement eventuel par ordre des x croissants
if( order ) {
  if( tmp == NULL ){
   tmp = new double[n];
  }
  
  std::vector<std::pair<double,double> > vxy(Nel); 
  for(int i=0;i<Nel;i++) vxy[i]=std::make_pair(x[i],y[i]);
  std::sort(vxy.begin(),vxy.end(),sort_pair_first<double,double>());

  for(int i=0;i<Nel;i++) {
    X[i] = vxy[i].first;
    Y[i] = vxy[i].second;
    if( i>0 ) if( X[i-1]>= X[i] ) {
      printf("CSpline::SetNewTab_Erreur: X[%d]>=X[%d] (%g>=%g)\n"
            ,i-1,i,X[i-1],X[i]);
      throw AngpowError("CSpline::SetNewTab_Erreur");
    }
  }
} else { X = x; Y = y; }

}

//////////////////////////////////////////////////////////////////////////////
//! destruction des divers tableaux en tenant compte des allocations/connections
void CSpline::DelTab()
{
  if( X   != NULL && XY_Created ){ delete [] X;    X   = NULL;}
  if( Y   != NULL && XY_Created ){ delete [] Y;    Y   = NULL;}
  if( Y2  != NULL ){ delete [] Y2;   Y2  = NULL;}
  if( tmp != NULL ){ delete [] tmp;  tmp = NULL;}

}

//////////////////////////////////////////////////////////////////////////////
/*!
  Pour changer les valeurs des derivees 1ere au 1er et dernier points
  Valeurs imposees des derivees 1ere au points "X[0]" et "X[Nel-1]".
*/
void CSpline::SetBound1er(double yp1,double ypn)
{
if( yp1 == YP1 && ypn == YPn ) return;

YP1 = yp1;
YPn = ypn;

corrupt_Y2 = true;
}

//! Pour calculer les tableaux de coeff permettant le calcul des interpolations spline.
void CSpline::ComputeCSpline()
{
// on ne fait rien si les tableaux ne sont pas connectes
if( X == NULL || Y == NULL ) {
  throw AngpowError("CSpline:::ComputeCSpline()_Erreur: tableaux non connectes");
//   printf("CSpline::ComputeCSpline()_Erreur: tableaux non connectes X=%p Y=%p\n"
//         ,X,Y);
  return;
}
// On ne fait rien si rien n'a change!
if( ! corrupt_Y2 ) return;
// protection si tmp a ete desalloue pour gain de place (ex: CSpline2)
 if( tmp == NULL ) {
   tmp = new double[Nel];
 }

double p,qn,sig,un;

 if (Natural & Natural1){
  Y2[0] = tmp[0] = 0.0;
 }
else {
  if (Natural == AutoDeriv){//JEC 25/9/16 calcul de Y'[0] a l'aide du tableau lui-meme.
    YP1 = ((X[2]-X[0])*(X[2]-X[0])*(Y[1]-Y[0])-
	   (X[1]-X[0])*(X[1]-X[0])*(Y[2]-Y[0]))
      /((X[2]-X[0])*(X[1]-X[0])*(X[2]-X[1]));
  }
  Y2[0] = -0.5;
  tmp[0] = (3.0/(X[1]-X[0]))*((Y[1]-Y[0])/(X[1]-X[0])-YP1);
}

for (int i=1;i<Nel-1;i++) {
  sig = (X[i]-X[i-1])/(X[i+1]-X[i-1]);
  p = sig * Y2[i-1] + 2.0;
  Y2[i] = (sig-1.0)/p;
  tmp[i]= (Y[i+1]-Y[i])/(X[i+1]-X[i]) - (Y[i]-Y[i-1])/(X[i]-X[i-1]);
  tmp[i]= (6.0*tmp[i]/(X[i+1]-X[i-1])-sig*tmp[i-1])/p;
}

if (Natural & NaturalN)
   qn = un = 0.0;
else {
  if (Natural == AutoDeriv){//JEC 25/9/16 calcul de Y'[N-1] a l'aide du tableau lui-meme.
    YPn = ((X[Nel-3]-X[Nel-1])*(X[Nel-3]-X[Nel-1])*(Y[Nel-2]-Y[Nel-1])-
	   (X[Nel-2]-X[Nel-1])*(X[Nel-2]-X[Nel-1])*(Y[Nel-3]-Y[Nel-1]))
      /((X[Nel-3]-X[Nel-1])*(X[Nel-2]-X[Nel-1])*(X[Nel-3]-X[Nel-2]));
  }
   qn = 0.5;
   un = (3.0/(X[Nel-1]-X[Nel-2]))
       *(YPn-(Y[Nel-1]-Y[Nel-2])/(X[Nel-1]-X[Nel-2]));
}
Y2[Nel-1] = (un-qn*tmp[Nel-2])/(qn*Y2[Nel-2]+1.0);
for (int k=Nel-2;k>=0;k--) Y2[k] = Y2[k]*Y2[k+1] + tmp[k];

corrupt_Y2 = false;
}

//!	Interpolation spline en \b x
double CSpline::CSplineInt(double x) const
{
int klo,khi,k;
double h,b,a,y = 0.;

if( corrupt_Y2 ) {
  throw AngpowError("CSpline::CSplineInt: calcul des coef du spline corrupted");
}

klo = 0;
khi = Nel-1;
while (khi-klo > 1) {
  k = (khi+klo) >> 1;
  if (X[k] > x) khi=k;
    else klo=k;
}
h=X[khi]-X[klo];

if (h == 0.0) {
  std::cout<<"CSpline::CSplineInt: pour khi="<<khi<<" klo="<<klo
	   <<" memes valeurs de X[]: "<<X[khi]<<std::endl;
  throw AngpowError("CSpline::CSplineInt error");
}

a = (X[khi]-x)/h;
b = (x-X[klo])/h;
y = a*Y[klo]+b*Y[khi]+((a*a*a-a)*Y2[klo]+(b*b*b-b)*Y2[khi])*(h*h)/6.0;

return y;
}


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
/*!
  Contructeur - Meme commentaire que pour CSpline avec:
  \verbatim
  x1[n1]: liste des coordonnees selon l axe 1
  x2[n2]: liste des coordonnees selon l axe 2
  y[n1*n2]: liste des valeurs avec le rangement suivant
  x1[0]......x1[n1-1]  x1[0]......x1[n1-1]  ... x1[0]......x1[n1-1]
  |    0<=i<n1      |  |    0<=i<n1      |  ... |    0<=i<n1      |
  |      j=0 X2[0]            j=1 X2[1]              j=n2-1 X2[n2-1]
  \endverbatim
*/
CSpline2::CSpline2(int n1,double* x1,int n2,double* x2,double* y
		   ,int natural,bool order)
  : Nel1(0), Nel2(0), corrupt_Y2(true), XY_Created(false), Natural(natural)
  , X1(NULL), X2(NULL), Y(NULL), Y2(NULL)
  , Nel_S(0), S(NULL), Sint(NULL), tmp(NULL)
{
  SetNewTab(n1,x1,n2,x2,y,order,true);
  if( x1 != NULL && x2 != NULL && y != NULL) ComputeCSpline();

}

//!	Createur par defaut.
CSpline2::CSpline2(int natural)
  : Nel1(0), Nel2(0), corrupt_Y2(true), XY_Created(false), Natural(natural)
  , X1(NULL), X2(NULL), Y(NULL), Y2(NULL)
  , Nel_S(0), S(NULL), Sint(NULL), tmp(NULL)
{
}

CSpline2::~CSpline2()
{
  DelTab();
}

//!	Voir commentaire meme methode de CSpline
void CSpline2::SetNewTab(int n1,double* x1,int n2,double* x2,double* y
			 ,bool order,bool force)
{
  assert( n1>3 && n2>3 );
  
  int n = ( n1 < n2 ) ? n2 : n1;
  
  //allocation des buffers Y2 et tmp et des CSpline 1D
  if( n1>Nel1 || n2>Nel2 || force ) {
    if( Y2  != NULL ){ delete [] Y2; Y2=NULL;}
    if( tmp != NULL ){ delete [] tmp; tmp=NULL;}
    Y2   = new double[n1*n2];
    tmp  = new double[n];
    
    // et les CSpline[n1] pour memoriser les interpolations sur x1(0->n1)
    if( S != NULL ) {
     for(int i=0;i<Nel_S;i++) if(S[i] != NULL) { delete S[i]; S[i]=NULL;}
     delete S; S = NULL;
    }
    S = new CSpline * [n2];
    for(int j=0;j<n2;j++) {
      S[j] = new CSpline(n1,NULL,NULL,0.,0.,Natural);
      S[j]->Free_Tmp();
    }
    Nel_S = n2;

    if( Sint != NULL ) { delete Sint; Sint = NULL;} //JEC was S != NULL
    Sint = new CSpline(n2,NULL,NULL,0.,0.,Natural);
  }

  //des-allocation eventuelle de X1,X2,Y
  if( XY_Created ) {
    if( !order || n1>Nel1 || n2>Nel2 || force ) {
      if( X1 != NULL ) delete [] X1; X1 = NULL;
      if( X2 != NULL ) delete [] X2; X2 = NULL;
      if( Y != NULL )  delete [] Y;  Y  = NULL;
      XY_Created = false;
    }
  }
  //allocation eventuelle de X1,X2,Y
  if( order ) {
    if( !XY_Created || n1>Nel1 || n2>Nel1 || force ) {
      X1 = new double[n1];
      X2 = new double[n2];
      Y  = new double[n1*n2];
      XY_Created = true;
    }
  }
  Nel1 = n1;  Nel2 = n2;
  corrupt_Y2 = true;

  if( x1==NULL || x2==NULL || y==NULL ) return;

  //Classement eventuel par ordre des x1 et x2 croissants (JEC 26/9/16 use sort)
  if( order ) {
    
    int N12=n1*n2;

    std::vector<std::pair<std::pair<double,double>,double> >vx1x2y(N12);
    int ij=0;
    for(int j=0;j<n2;j++){
      for(int i=0;i<n1;i++){
	vx1x2y[ij++]=std::make_pair(std::make_pair(x1[i],x2[j]),y[j*n1+i]);
      }
    }
    
    std::sort(vx1x2y.begin(),vx1x2y.end(), sort_pairpair_second<double,double>());
      
      for(int i=0;i<Nel1;i++){//j=0
	X1[i] = vx1x2y[i].first.first;
      }
      
      for(int j=0;j<Nel2;j++){// ij=j*Nel1+i; avec i=0
	X2[j] = vx1x2y[j*Nel1].first.second;
      }
      
      for(int i=0;i<Nel1;i++){
	for(int j=0;j<Nel2;j++){
	  Y[j*Nel1+i] = vx1x2y[i+j*Nel1].second;
	}
      }

    
  } else { //pas de tri necessaire
    X1 = x1;
    X2 = x2;
    Y  = y; 
  }

}//SetNewTab

void CSpline2::DelTab()
{

  if( X1  != NULL && XY_Created ) { delete [] X1;  X1   = NULL;}
  if( X2  != NULL && XY_Created ) { delete [] X2;  X2   = NULL;}
  if( Y   != NULL && XY_Created ) { delete [] Y;   Y   = NULL;}
  if( Y2  != NULL ) { delete [] Y2;   Y2  = NULL;}
  if( tmp != NULL ) { delete [] tmp;  tmp = NULL;}
  if( S != NULL ) {
    for(int i=0;i<Nel_S;i++) if(S[i] != NULL) { 
      S[i]->tmp = tmp; //JEC
      delete S[i]; S[i]=NULL;}
    delete S; S = NULL;
  }

  if( Sint != NULL ) { delete Sint; Sint=NULL;}

}


//!	Voir commentaire meme methode de CSpline
void CSpline2::ComputeCSpline()
{
  //on ne fait rien si X1 ou X2 ou Y non connectes
  if( X1 == NULL || X2 == NULL || Y == NULL ) return;
  //  On ne fait rien si rien n'a change
  if( ! corrupt_Y2 ) return;

  for(int j=0; j<Nel2; j++) {
    // on n'alloue pas de place nouvelle, on utilise CSpline2::tmp
    S[j]->tmp = tmp;
    // connection de X1,Y au spline 1D sans ordre demande
    S[j]->SetNewTab(Nel1,X1,&Y[j*Nel1],false,false);
    // calcul des coeff splien pour l'interpolation future
    S[j]->ComputeCSpline();
  }
  
  corrupt_Y2 = false;
}

//! Calcule la valeur interpole (spline) pour le point \b (x1,x2)
double CSpline2::CSplineInt(double x1,double x2)
{
  //calcul de la valeur Y pour x=x1 et remplissage du tampon tmp
  for(int j=0;j<Nel2;j++) tmp[j] = S[j]->CSplineInt(x1);

  //connection X2,tmp pour interpolation selon x=x2
  Sint->SetNewTab(Nel2,X2,tmp,false,false);
  //calcul des coeff pour interpolation selon X2
  Sint->ComputeCSpline();
  //Interpolation finale
  return Sint->CSplineInt(x2);
}



//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*! 
  \class SLinInterp1D
  \ingroup NTools
  \brief Simple linear interpolation class 

  \code
  ...
  vector<double> ys;
  double xmin = 0.5;
  double xmax = 0.;
  for(int i=0; i<=12; i++) {
    xmax = xmin+i*0.1;
    yreg.push_back(sin(xmax)*cos(2.2*xmax));
  }
  SLinInterp1D interpYR(xmin, xmax, yreg);
  cout << interpYR
  for(int i=0; i<=12; i++) {
    double x = drand01()*2.;
    cout << " Interpol result for X=" << x << " -> " << interpYR(x) << " ?= " << sin(x)*cos(2.2*x) << endl;
  }
  \endcode 
*/

/* --Methode-- */
SLinInterp1D::SLinInterp1D()
  : xmin_(0.), xmax_(1.), dx_(1.), ksmx_(1), npoints_(0)
{
  xs_.push_back(0.);
  xs_.push_back(1.); 
  ys_.push_back(0.);
  ys_.push_back(1.); 
}

/*!
  \brief Constructor from regularly spaced points in X with Y values defined by yreg 

  \b xmin and \b xmax are the two extreme points in X corresponding to yreg.
  Example: xmin=1, xmax=10, with yreg.size()=10 , 
  yreg[0]=Y(1) , yreg[1]=Y(2)  ...  yreg[9]=Y(10) 
*/
SLinInterp1D::SLinInterp1D(double xmin, double xmax, std::vector<double>& yreg)
  : xmin_(0.), xmax_(1.), dx_(1.), ksmx_(1), npoints_(0)
{
  DefinePoints(xmin, xmax, yreg);
}


/*!
  \brief Constructor from a set of \b (xs[i],ys[i]) pairs.

  if (npt > 0), interpolates to a finer regularly spaced grid, from \b xmin to \b xmax 
  with npt points.  use \b (xs[0],xs[xs.size()-1]) as limits if \b xmax<xmin 
*/
SLinInterp1D::SLinInterp1D(std::vector<double>& xs, std::vector<double>& ys, double xmin, double xmax, size_t npt)
  : xmin_(0.), xmax_(1.), dx_(1.), ksmx_(1), npoints_(0)
{
  DefinePoints(xs, ys, xmin, xmax, npt);
}

/* --Methode-- */
double SLinInterp1D::YInterp(double x)  const
{
  if (npoints_>0) {  // on utilise les points regulierement espace 
    long i = (long)((x-xmin_)/dx_);  
    if (i<0) return ( yreg_[0]+(x-xmin_)*(yreg_[1]-yreg_[0])/dx_ );
    if (i>=(long)npoints_) return ( yreg_[npoints_]+(x-xmax_)*(yreg_[npoints_]-yreg_[npoints_-1])/dx_ );
    return (yreg_[i]+(x-X(i))/dx_*(yreg_[i+1]-yreg_[i]));
  }
  else {  // On utilise les points xs_,ys_ directement 
    if (x<=xs_[0]) return ( ys_[0]+(x-xs_[0])*(ys_[1]-ys_[0])/(xs_[1]-xs_[0]) );
    if (x>=xs_[ksmx_]) return ( ys_[ksmx_]+(x-xs_[ksmx_])*(ys_[ksmx_]-ys_[ksmx_-1])/(xs_[ksmx_]-xs_[ksmx_-1]) );

    size_t k=1;
    while(x>xs_[k]) k++; 
    if (k>=xs_.size()) {  // ne devrait pas arriver ...
      std::string emsg = " SLinInterp1D::YInterp()  out of range k -> BUG in code ";
      throw AngpowError(emsg);
    }

    double rv=ys_[k-1]+(x-xs_[k-1])*(ys_[k]-ys_[k-1])/(xs_[k]-xs_[k-1]);
    //    cout << " DBG- x=" << x << " k=" << k << " xs[k]=" << xs_[k] << " ys[k]" << ys_[k] 
    //	 << " rv=" << rv << endl;
    return rv;
  }
}

/*!
  \brief Defines the interpolation points from regularly spaced points in X with Y values defined by yreg 

  \b xmin and \b xmax are the two extreme points in X corresponding to yreg.
  Example: xmin=1, xmax=10, with yreg.size()=10 , 
  yreg[0]=Y(1) , yreg[1]=Y(2)  ...  yreg[9]=Y(10) 
*/
void SLinInterp1D::DefinePoints(double xmin, double xmax, std::vector<double>& yreg)
{
  if (yreg.size()<2)  {
    std::string emsg = "SLinInterp1D::DefinePoints(xmin,xmax,yreg) Bad parameters yreg.size()<2 ";
    throw AngpowError(emsg);
  }
  xmin_ = xmin; 
  xmax_ = xmax;
  npoints_ = yreg.size()-1;
  dx_ = (xmax_-xmin_)/(double)npoints_;
  yreg_ = yreg;
}


/*!
  \brief Define the interpolation points from a set of \b (xs[i],ys[i]) pairs.

  if (npt > 0), interpolates to a finer regularly spaced grid, from \b xmin to \b xmax 
  with npt points.  use \b (xs[0],xs[xs.size()-1]) as limits if \b xmax<xmin 
*/
void SLinInterp1D::DefinePoints(std::vector<double>& xs, std::vector<double>& ys, double xmin, double xmax, size_t npt)
{
  if ((xs.size() != ys.size())||(xs.size()<2))  {
    std::string emsg = "SLinInterp1D::DefinePoints() Bad parameters (xs.size() != ys.size())||(xs.size()<2) ";
    throw AngpowError(emsg);
  }
  for(size_t k=1; k<xs.size(); k++) {
    if (xs[k-1]>=xs[k])  { 
       std::string emsg =  "SLinInterp1D::DefinePoints()  unsorted xs ";
      throw AngpowError(emsg);
    }
  }
  xs_=xs;
  ys_=ys;
  ksmx_=xs_.size()-1;
  npoints_ = npt;
  if (xmin>=xmax)  {
    xmin_ = xs_[0]; 
    xmax_ = xs_[ksmx_];
  }
  else {
    xmin_ = xmin; 
    xmax_ = xmax;
  }
  if (npoints_<1) {
    dx_=(xmax_-xmin_)/(double)(xs_.size()-1);
    return;
  }
  dx_ = (xmax_-xmin_)/(double)npoints_;
  yreg_.resize(npoints_+1);
  
  // Compute the the y values for regularly spaced x xmin <= x <= xmax 
  // and keep values in the yreg std::vector
  yreg_[0] = ys_[0];
  yreg_[npoints_] = ys_[ksmx_];
  size_t k=1;
  for(size_t i=0; i<npoints_; i++)  {
    double x = X(i);
    while(x>xs_[k]) k++;
    if (k>=xs_.size())  k=xs_.size()-1;   
    yreg_[i] = ys_[k-1]+(x-xs_[k-1])*(ys_[k]-ys_[k-1])/(xs_[k]-xs_[k-1]);
    //DBG cout << " DBG* i=" << i << " X(i)=" << X(i) << " yreg_[i]= " << yreg_[i] << " X^2= " << X(i)*X(i) 
    //DBG << " k=" << k << " xs[k]=" << xs_[k] << endl;
  }
  return;
}


/*!
  \brief Read  Y values from the file \b filename

  Read Y values ( one/line) for regularly spaced X's from file \b filename and call 
  DefinePoints(xmin, xmax, yreg). Return the number of Y values read.
  \param filename : input file name 
  \param xmin,xamx : X range limits 
  \param clm : comment character, lines starting with \b clm are ignored 
*/
size_t SLinInterp1D::ReadYFromFile(std::string const& filename, double xmin, double xmax, char clm)
{
  std::ifstream inputFile;
  inputFile.open(filename.c_str(), std::ifstream::in);  
#ifndef __DECCXX
// ifstream.is_open() ne passe pas avec OSF-cxx
  if(! inputFile.is_open()) {
    std::string emsg = "  SLinInterp1D::ReadYFromFile() problem opening file ";
    emsg += filename;
    throw AngpowError(emsg);
  }
#endif

  std::vector<double> xsv, ysv;
  size_t cnt=0;
  double cola;
  std::string eline;
  while(!inputFile.eof())  { 
    inputFile.clear(); 
    if (inputFile.peek() == (int)clm)  {  // skip comment lines 
      getline(inputFile, eline);
      continue;
    }
    inputFile >> cola; 
    if ( (!inputFile.good()) || inputFile.eof())   break;
    inputFile.ignore(1024,'\n');  // make sure we go to the next line
    //cout << cola<< "    "<<colb<<endl;
    ysv.push_back(cola);
    cnt++;
  }
  inputFile.close();
  std::cout << " SLinInterp1D::ReadYFromFile()/Info: " << cnt << " Y-values read from file " << filename << std::endl;  
  DefinePoints(xmin, xmax, ysv);
  return cnt;
}

/*!
  \brief Read pairs of  ( X Y ) values from file \b filename 

  Read pairs of (X Y) values, one pair / line from the specified file and call DefinePoints(xs, ys ...).
  One pair of space or tab separated numbers on each line. Return the number of Y values read.
  \param filename : input file name 
  \param xmin,xamx : X range limits. use the X limits from the file if xmax<xmin
  \param npt : number of points for regularly spaced interpolation points
  \param clm : comment character, lines starting with \b clm are ignored 
*/
size_t SLinInterp1D::ReadXYFromFile(std::string const& filename, double xmin, double xmax, size_t npt, char clm)
{
  std::ifstream inputFile;
  inputFile.open(filename.c_str(), std::ifstream::in); 
#ifndef __DECCXX 
// ifstream.is_open() ne passe pas avec OSF-cxx
  if(! inputFile.is_open()) {
    std::string emsg = "  SLinInterp1D::ReadXYFromFile() problem opening file ";
    emsg += filename;
    throw AngpowError(emsg);
  }
#endif

  std::vector<double> xsv, ysv;
  size_t cnt=0;
  double cola, colb;
  std::string eline;
  while(!inputFile.eof())  {  
    inputFile.clear();
    if (inputFile.peek() == (int)clm)  {  // skip comment lines 
      getline(inputFile, eline);
      continue;
    }
    inputFile >> cola >> colb; 
    if ( (!inputFile.good()) || inputFile.eof())   break;
    inputFile.ignore(1024,'\n');  // make sure we go to the next line
    //    cout << " DEBUG-GCount " << inputFile.gcount() << endl;
    //    cout << " DEBUG - cnt=" << cnt << " x=" << cola << "  y= "<<colb<<endl;
    xsv.push_back(cola);
    ysv.push_back(colb);
    cnt++;
  }
  inputFile.close();
  std::cout << " SLinInterp1D::ReadXYFromFile()/Info: " << cnt << " (x,y) pairs read from file " << filename << std::endl;  
  DefinePoints(xsv, ysv, xmin, xmax, npt);
  return cnt;
}


/* --Methode-- */
std::ostream& SLinInterp1D::Print(std::ostream& os, int lev)  const
{
  os << " ---- SLinInterp1D::Print() XMin=" << XMin() << " XMax=" << XMax() << " NPoints=" << npoints_ << std::endl;
  os << "  xs_.size()= " << xs_.size() << "  ys_.size()= " << ys_.size() << "  yreg_.size()= " << yreg_.size() << std::endl;
  if ((lev>0)&&(xs_.size()>0)) {
    for(size_t i=0; i<xs_.size(); i++) 
      os << " xs[" << i << " ]=" << xs_[i] << " -> ys[" << i << "]=" << ys_[i] << std::endl;
  } 
  if ((lev>0)&&(yreg_.size()>0)) {
    for(size_t i=0; i<yreg_.size(); i++) 
      os << " Regularly Spaced X(" << i << " )=" << X(i) << " -> yreg_[" << i << "]=" << yreg_[i] << std::endl;
  } 
  os << " -----------------------------------------------------------------------" << std::endl;
  return os;
}


}//namespace
