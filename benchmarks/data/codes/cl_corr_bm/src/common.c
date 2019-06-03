#include "common.h"

void *dam_malloc(size_t size)
{
  void *outptr=malloc(size);
  if(outptr==NULL) dam_report_error(1,"Out of memory\n");

  return outptr;
}

void *dam_calloc(size_t nmemb,size_t size)
{
  void *outptr=calloc(nmemb,size);
  if(outptr==NULL) 
    dam_report_error(1,"Out of memory\n");

  return outptr;
}

FILE *dam_fopen(const char *path,const char *mode)
{
  FILE *fout=fopen(path,mode);
  if(fout==NULL) 
    dam_report_error(1,"Couldn't open file %s\n",path);

  return fout;
}

int dam_linecount(FILE *f)
{
  int i0=0;
  char ch[1024];
  while((fgets(ch,sizeof(ch),f))!=NULL) {
    i0++;
  }
  return i0;
}

void dam_report_error(int level,char *fmt,...)
{
  va_list args;
  char msg[256];

  va_start(args,fmt);
  vsprintf(msg,fmt,args);
  va_end(args);
  
  if(level) {
    fprintf(stderr,"Fatal: %s",msg);
    exit(level);
  }
  else {
    fprintf(stderr,"Warning: %s",msg);
  }
}

SplPar *spline_init(int n,double *x,double *y,double y0,double yf)
{
  SplPar *spl=(SplPar *)dam_malloc(sizeof(SplPar));
  spl->intacc=gsl_interp_accel_alloc();
  spl->spline=gsl_spline_alloc(gsl_interp_cspline,n);
  gsl_spline_init(spl->spline,x,y,n);
  spl->x0=x[0];
  spl->xf=x[n-1];
  spl->y0=y0;
  spl->yf=yf;

  return spl;
}

double spline_eval(double x,SplPar *spl)
{
  if(x<=spl->x0)
    return spl->y0;
  else if(x>=spl->xf) 
    return spl->yf;
  else
    return gsl_spline_eval(spl->spline,x,spl->intacc);
}

void spline_free(SplPar *spl)
{
  gsl_spline_free(spl->spline);
  gsl_interp_accel_free(spl->intacc);
  free(spl);
}

RunParams *param_new(void)
{
  RunParams *par=(RunParams *)dam_malloc(sizeof(RunParams));
  par->om=0.3;
  par->ol=0.7;
  par->ob=0.05;
  par->w0=-1.;
  par->wa=0.;
  par->ns=0.96;
  par->s8=0.8;
  par->fname_window=dam_malloc(2*sizeof(char *));
  par->fname_window[0]=dam_malloc(256*sizeof(char));
  par->fname_window[1]=dam_malloc(256*sizeof(char));
  sprintf(par->fname_window[0],"default");
  sprintf(par->fname_window[1],"default");
  sprintf(par->fname_bias,"default");
  sprintf(par->fname_sbias,"default");
  sprintf(par->fname_pk,"default");
  sprintf(par->prefix_out,"default");
  par->lmax=100;
  par->cpar=NULL;
  par->chi_horizon=-1.;
  par->chi_LSS=-1.;
  par->prefac_lensing=-1.;
  par->dchi=-1.;
  par->aofchi=NULL;
  par->zofchi=NULL;
  par->hofchi=NULL;
  par->gfofchi=NULL;
  par->fgofchi=NULL;
  par->wind_0=NULL;
  par->wind_M=NULL;
  par->wind_L=NULL;
  par->bias=NULL;
  par->sbias=NULL;
  par->do_nc=0;
  par->do_shear=0;
  par->do_cmblens=0;
  par->do_isw=0;
  par->has_bg=0;
  par->has_dens=0;
  par->has_rsd=0;
  par->has_lensing=0;
  par->cl_dd=NULL;
  par->cl_d1l2=NULL;
  par->cl_d2l1=NULL;
  par->cl_dc=NULL;
  par->cl_di=NULL;
  par->cl_ll=NULL;
  par->cl_lc=NULL;
  par->cl_li=NULL;
  par->cl_cc=NULL;
  par->cl_ci=NULL;
  par->cl_ii=NULL;
  par->do_w_theta=0;
  par->th_min=0;
  par->th_max=10.;
  par->n_th=15;
  par->n_th_logint=5;
  par->wt_dd=NULL;
  par->wt_d1l2=NULL;
  par->wt_d2l1=NULL;
  par->wt_dc=NULL;
  par->wt_di=NULL;
  par->wt_ll_pp=NULL;
  par->wt_ll_mm=NULL;
  par->wt_lc=NULL;
  par->wt_li=NULL;
  par->wt_cc=NULL;
  par->wt_ci=NULL;
  par->wt_ii=NULL;
  return par;
}

void param_free(RunParams *par)
{
  csm_params_free(par->cpar);
  if(par->has_bg) {
    spline_free(par->aofchi);
    spline_free(par->zofchi);
    spline_free(par->hofchi);
    spline_free(par->gfofchi);
    spline_free(par->fgofchi);
  }
  if(par->do_nc || par->do_shear) {
    spline_free(par->wind_0[0]);
    spline_free(par->wind_0[1]);
    free(par->wind_0);
  }
  if(par->do_nc) {
    free(par->cl_dd);
    if(par->do_w_theta)
      free(par->wt_dd);
    if(par->do_shear) {
      free(par->cl_d1l2);
      free(par->cl_d2l1);
      if(par->do_w_theta) {
	free(par->wt_d1l2);
	free(par->wt_d2l1);
      }
    }
    if(par->do_cmblens) {
      free(par->cl_dc);
      if(par->do_w_theta)
	free(par->wt_dc);
    }
    if(par->do_isw) {
      free(par->cl_di);
      if(par->do_w_theta)
	free(par->wt_di);
    }
    if(par->has_dens)
      spline_free(par->bias);
    if(par->has_lensing) {
      spline_free(par->sbias);
      spline_free(par->wind_M[0]);
      spline_free(par->wind_M[1]);
      free(par->wind_M);
    }
  }
  if(par->do_shear) {
    spline_free(par->wind_L[0]);
    spline_free(par->wind_L[1]);
    free(par->wind_L);
    free(par->cl_ll);
    if(par->do_w_theta) {
      free(par->wt_ll_pp);
      free(par->wt_ll_mm);
    }
    if(par->do_cmblens) {
      free(par->cl_lc);
      if(par->do_w_theta)
	free(par->wt_lc);
    }
    if(par->do_isw) {
      free(par->cl_li);
      if(par->do_w_theta)
	free(par->wt_li);
    }
  }
  if(par->do_cmblens) {
    free(par->cl_cc);
    if(par->do_isw) {
      free(par->cl_ci);
      if(par->do_w_theta)
	free(par->wt_ci);
    }
  }
  if(par->do_isw)
    free(par->cl_ii);

  free(par);
}
