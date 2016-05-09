#include "ccl.h"
#include <assert.h>
#include <math.h>


int test_ccl_linear_spacing(){
    
    int n1;
    double * m1 = ccl_linear_spacing(0.0, 1.0, 1.0, &n1);
    assert ((n1==2));
    assert (m1!=NULL);
    assert (m1[0]==0.0);
    assert (m1[1]==1.0);
    free(m1);

    int n2;
    double * m2 = ccl_linear_spacing(0.0, 1.0, 0.75, &n2);
    assert (n2==0);
    assert (m2==NULL);
    free(m2);

    int n3;
    double * m3 = ccl_linear_spacing(0.0, 1.0, 2.0, &n3);
    assert (n3==0);
    assert (m3==NULL);
    free(m3);

    int n4;
    double * m4 = ccl_linear_spacing(A_SPLINE_MIN, A_SPLINE_MAX, A_SPLINE_DELTA, &n4);
    assert (n4!=0);
    assert (m4!=NULL);
    assert (fabs(m4[n4-1]-A_SPLINE_MAX)<1e-5);
    assert (fabs(m4[n4-1])<=1.0);
    assert (fabs(m4[0]-A_SPLINE_MIN)<1e-5);
    free(m4);

    return 0;


}



int main(int argc, char * argv[]){
    test_ccl_linear_spacing();
    return 0;
}
