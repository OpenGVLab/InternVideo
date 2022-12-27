/*  \index
 * weibull.cpp provides the core functionality for computing weibull fittings, as well as CDF and INF given parms 
 *  
 * 
 * @Author Brian Heflin <bheflin@securics.com>
 * @Author Walter Scheirer <walter@securics.com>
 * @Author Terry Boult tboult@securics.com
 *
 * Copyright 2010, 2011, Securics Inc.
 *
 * @section LICENSE
 *  See accompanying LICENSE agrement for full details on rights.
 *
 * Parts of this technology are subject to SBIR data rights and as described in DFARS 252.227-7018 (June 1995) SBIR Data Rights which apply to Contract Number: N00014-11-C-0243 and STTR N00014-07-M-0421 to Securics Inc, 1870 Austin Bluffs Parkway, Colorado Springs, CO 80918
 *
 *The Government's rights to use, modify, reproduce, release, perform, display, or disclose technical data or computer software marked with this legend are restricted during the period shown as provided in paragraph (b)(4) of the Rights in Noncommercial Technical Data and Computer Software-Small Business Innovative Research (SBIR) Program clause contained in the above identified contract.  Expiration of SBIR Data Rights: Expires four years after completion of the above cited project work for this or any other follow-on SBIR contract, whichever is later.
 *
 * No restrictions on government use apply after the expiration date shown above.  Any reproduction of technical data, computer software, or portions thereof marked with this legend must also reproduce the markings.
 *
 *
 * See overall comments in weibull.h
 *
 */


#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "malloc.h"
#include <memory.h>
#include <float.h>

#include "weibull.h"

#ifdef __cplusplus
extern "C" {
#endif

  /* if WEIBULL_USE_ASSERTS is defined, the code will use asserts to ensure its requirements are true, otherwise it returns error codes. Default is not defined */ 
  /* if WEIBULL_IGNORE_ERRORS is defined, the code will just presume things will work out and not waste time on testing for error. Default is not defined */ 

  /*#define WEIBULL_USE_ASSERTS */
  /* #define WEIBULL_IGNORE_ERRORS  */

#ifdef WEIBULL_USE_ASSERTS
#include <assert.h>
#endif




#ifdef WEIBULL_IGNORE_ERRORS
  int weibull_fit_verbose_debug=0;
#define WEIBULL_ERROR_HANDLER(x,msg) 
#else 
  int weibull_fit_verbose_debug=1;
  static int  tthrow(int x, const char* msg){if(weibull_fit_verbose_debug) fprintf(stderr,"%s\n",msg); return x;}
#define WEIBULL_ERROR_HANDLER(x,msg) return tthrow(x,msg)
#endif

  
  



  /*  weibull_cdf computes the probability (given our assumptions) that the value x is an outlier ABOVE the fit distribution.  if the distribution was non-match data, then it provides this probability that x is a match score.   If data was match-data then it would be the probability of it being a larger non-match. 
  computes @f[ 1-e^{{x/scale}^{shape}} @f]

  @param x  the location at which to compute the probability of being an outlier
  @param scale the scale parameter of the Weibull.  This is the first element in weibullparms (as computed by our weibull_fit) 
  @param shape the scale parameter of the Weibull.  This is the first second in weibullparms (as computed by our weibull_fit) 
  @return if in the range [0-1] it is the probability of X being an outlier.  Any value < 0 is an error code.  returns -1 for invalid scale <=0 ,  -2 for invalid shape <=0 
  *
  */ 
double weibull_cdf(double x, double scale, double shape)
{
    double cdf;
    double tempVal, tempVal1;

    if(x<0) return 0; /* optimize for the simple case that can be common in playing with SVMs.  (a valid value, and can ignore other possible errors) */

#ifndef WEIBULL_IGNORE_ERRORS
#ifdef WEIBULL_USE_ASSERTS
    assert(scale>0);
    assert(shape>0);
#else
    if(scale<=0) WEIBULL_ERROR_HANDLER( -1, "Bad scale in weibull_cdf");
    if(shape <=0) WEIBULL_ERROR_HANDLER(-2, "Bad shape in weibull_cdf"); 
#endif
#endif



    tempVal =  x/scale;
    tempVal1 = pow(tempVal,shape);
    cdf = 1-exp(-1*tempVal1);

    return cdf;

}


  /*  weibull_inv computes the inverse weibull, i.e. returns the score S (given our assumptions) such that x=weibull_cdf(s,scale,shape). Note it estimates from above, so if x=1.0 expect an answer of inf (infinity). 

  @param x  the location at which you compute the inverse (must be between [0,1]
  @param scale the scale parameter of the weibull.  This is the first element in weibullparms (as computed by our weibull_fit) 
  @param shape the scale parameter of the weibull.  This is the first second in weibullparms (as computed by our weibull_fit) 
  @return if X in the range [0-1], return S such that x=weibull_cdf(s,scale,shape).  The return value is in the range [0,inf].  Any return value < 0 is an error code.  returns -1 for invalid scale <=0 ,  -2 for invalid shape <=0  -3 for  X<0, -4 for x >1
  *
  */ 
double weibull_inv(double x, double scale, double shape)
{
    double inv;
    double tempVal, tempVal1, tempVal2;
#ifndef WEIBULL_IGNORE_ERRORS
#ifdef WEIBULL_USE_ASSERTS
    assert(scale>0);
    assert(shape>0);
    assert(x>=0);
    assert(x<=1);
#else
    if(scale<=0) WEIBULL_ERROR_HANDLER( -1, "Bad scale in weibull_cdf");
    if(shape <=0) WEIBULL_ERROR_HANDLER(-2, "Bad shape in weibull_cdf"); 
    if(x<0) WEIBULL_ERROR_HANDLER(-3,"Invalid X<0 in weibull_ing");
     if(x>1) WEIBULL_ERROR_HANDLER(-4,"Invalid X>1 in weibull_ing");
#endif 
#endif 

    tempVal = log(1-x);
    tempVal *= -1;

    tempVal1 = 1/shape;
    tempVal2 = pow(tempVal, tempVal1);

    inv = scale * tempVal2;

    return inv;
}

  /*
     printWeibullBuildInfo prints, to the file provied as an argument some information about the current package  build information 
  */

void printWeibullBuildInfo(FILE *fh)
{
    if(fh == NULL)
        fh = stdout;
#ifdef HAVE_CONFIG_H
    fprintf(fh, "Project name:    %s\n", PACKAGE_STRING);
#endif
    fprintf(fh, "Git SHA Hash  $Id: e6733c27c2f37c37aa58fe5f2b7d30aa084cd1e5 $\n");
}

/* abandon all hope ye who pass this point.  We be mixing stuff in ugly ugly way  including some conversion from fortran (erf approximations etc) and some non-obvious MLE estimations. */


#ifdef __cplusplus
}
#endif


#ifdef _WIN32
static __inline int fix(double n)
#else
static inline int fix(double n)
#endif
{
    return (int)n;
}




/* erf function, based on Fortran calcerf  which is turn is basd on 
  "Rational Chebyshev approximations for the error function" C   by W. J. Cody, Math. Comp., 1969, PP. 631-638. 
  This  code uses rational functions that theoretically approximate  erf(x)  and  erfc(x)  to at least 18 significant decimal digits, and on IEEE hardware is generally to near machine precision.
Note there are accelerated versions for GPUs and in the Intel Math Kernel library, so if you do this a lot it may be worh using those libraries. 
 */ 

static double wcalcerfc(double x)
{
    double PI =  3.141592653589793238462;
    double thresh = 0.46875;

    double a [] = {3.16112374387056560e00, 1.13864154151050156e02, 3.77485237685302021e02, 3.20937758913846947e03, 1.85777706184603153e-1};
    double b [] = {2.36012909523441209e01, 2.44024637934444173e02, 1.28261652607737228e03, 2.84423683343917062e03};
    double c [] = {5.64188496988670089e-1, 8.88314979438837594e00, 6.61191906371416295e01, 2.98635138197400131e02, 8.81952221241769090e02, 1.71204761263407058e03, 2.05107837782607147e03, 1.23033935479799725e03, 2.15311535474403846e-8};
    double d [] = {1.57449261107098347e01, 1.17693950891312499e02, 5.37181101862009858e02, 1.62138957456669019e03, 3.29079923573345963e03, 4.36261909014324716e03, 3.43936767414372164e03, 1.23033935480374942e03};
    double p [] = {3.05326634961232344e-1, 3.60344899949804439e-1, 1.25781726111229246e-1, 1.60837851487422766e-2, 6.58749161529837803e-4, 1.63153871373020978e-2};
    double q [] = {2.56852019228982242e00, 1.87295284992346047e00, 5.27905102951428412e-1, 6.05183413124413191e-2, 2.33520497626869185e-3};

    double result=0;
    double xk;
    double absxk;
    double y,z;
    double xnum,xden;
    double tempVal, tempVal1;
    double del;
    int i;

    xk = x;
    absxk = fabs(xk);

    if (absxk <= thresh) /* evaluate  erf  for  |x| <= 0.46875 */
    {
        y = absxk;
        z = y * y;
        xnum = a[4]*z;
        xden = z;

        for (i=0; i<3; i++)
        {
            xnum = (xnum + a[i]) * z;
            xden = (xden + b[i]) * z;
        }

        tempVal=xk*(xnum + a[3]);
        tempVal1=xden + b[3];

        result = tempVal/tempVal1;
        result = 1 - result;
    }
    else if (absxk <= 4.0)/* evaluate  erfc  for 0.46875 <= |x| <= 4.0 */
    {
        y = absxk;
        xnum = c[8]*y;
        xden = y;

        for (i = 0; i< 7; i++)
        {
            xnum = (xnum + c[i]) * y;
            xden = (xden + d[i]) * y;
        }

        tempVal=xnum + c[7];
        tempVal1=xden + d[7];
        result = tempVal/tempVal1;

        tempVal=fix(y*16);
        tempVal1=16;
        z=tempVal/tempVal1;

        del = (y-z)*(y+z);
        result = exp((-1*z)*z) * exp((-1*del)) * result;

    }
    else /*% evaluate  erfc  for |x| > 4.0 */
    {   
        y = absxk;
        z = 1/(y*y); 
        xnum = p[5]*z;
        xden = z;
        for (i = 0; i<4; i++)
        {
            xnum = (xnum + p[i]) * z;
            xden = (xden + q[i]) * z;
        }

        tempVal=z*(xnum + p[4]);
        tempVal1=xden + q[4];
        result=tempVal/tempVal1;

        tempVal=1/sqrt(PI);        
        tempVal -= result;
        tempVal1 = y;
        result=tempVal/tempVal1;

        tempVal=fix(y*16);
        z=tempVal/16;
        del = (y-z) * (y+z);
        result = exp((-1*z)*z) * exp((-1*del)) * result;

        /*check to see if result is finite */
        {
          int test;
#ifdef _WIN32
          test = _finite(result);
#else
          test = isfinite(result);
#endif
          if (test == 0) result = 0;
        }
    }

    /*fix up for negative argument, erf, etc. */
    if (xk < -thresh)
    {
        result = 2 - result;
    }
    else if (xk < -thresh)/* jint must = 2 */
    {
      if (xk < -26.628)  /* if less than XNEG (the largest negative argument acceptable to ERFCX) by IEEE standard */ 
        {
          result = 1000000; /*%%ERROR (INF) */
            WEIBULL_ERROR_HANDLER(-8,"wcalcerfc helper function failed to converge.." );
        }
        else
        {
            tempVal=fix(xk*16);
            tempVal1=16;
            z=tempVal/tempVal1;
            del = (xk-z)*(xk+z);
            y = exp(z*z) * exp(del);
            result = (y+y) - result;
        }
    }

    return result;
}

/* 

Calculate the inverse complementary error function of the input argument y, for y in the interval [0, 2]. The inverse complementary error function find the value x that satisfies the equation y = erfc(x).  based on fortran code based on  "Rational Chebyshev approximations for the error function" C   by W. J. Cody, Math. Comp., 1969, PP. 631-638. 
Note there are accelerated versions for GPUs and in the Intel Math Kernel library, so if you do this a lot it may be worh using those libraries. 
 */ 


static double derfcinv(double x)
{

    double a [] = {1.370600482778535e-02, -3.051415712357203e-01, 1.524304069216834, -3.057303267970988, 2.710410832036097, -8.862269264526915e-01};
    double b [] = {-5.319931523264068e-02, 6.311946752267222e-01, -2.432796560310728, 4.175081992982483, -3.320170388221430};
    double c [] = {5.504751339936943e-03, 2.279687217114118e-01, 1.697592457770869, 1.802933168781950, -3.093354679843504, -2.077595676404383};
    double d [] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996, 3.754408661907416};
    double q,r,u;

    double result = 0;
    double xlow = 0.0485000000;
    double xhigh = 1.9515000000;
    double tosp = 1.1283791670955126645;
    double xk;
    double tempVal, tempVal1, tempVal2;

    xk = x;

    /*Rational approximation for central region */
    if ((xlow <= xk) && (xk <= xhigh))
    { 
        q = xk - 1;
        r = q*q;

        tempVal1=(((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q;
        tempVal2=((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1;

        result = tempVal1/tempVal2;
    }

    /*Rational approximation for lower region */
    else if ((0 < xk) && (xk < xlow))
    {
        tempVal=xk/2;
        q = sqrt(-2*log(tempVal));

        tempVal1=((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5];
        tempVal2=(((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1;

        result=tempVal1/tempVal2;
    }

    /*Rational approximation for upper region */
    else if ((xhigh < xk) && (xk < 2))
    {
        tempVal=xk/2;
        q = sqrt(-2*log(1-tempVal));
        tempVal1= -1*(((((c[0]*q+c[1])*q+c[2]))*q+c[3])*q+c[4])*q+c[5];
        tempVal2= (((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1;

        result = tempVal1/tempVal2;

    }

    /* Root finding with Halley's method */
    /* For f = erfc(x) - y  so then  f' = -2/sqrt(pi)*exp(-x^2), and  f" = -2*x*f' */


    {
      double erfcx = wcalcerfc(result);
      if(erfcx <0) WEIBULL_ERROR_HANDLER(-9999,"derfcinv fails because wcalcerfc failed");

      tempVal= (-1*tosp);
      tempVal1=(-1*result)*(result);
      tempVal1=exp(tempVal1);
      tempVal2=tempVal*tempVal1;

      u = (erfcx - xk)/tempVal2;
    }

    tempVal=(1+result*u);

    tempVal1= u / tempVal;

    result = result - tempVal1;

    return result;

}


static int  weibull_neg_log_likelihood(double* nlogL, double* acov, double* weibulparms, double* data,
                     double* censoring, double* frequency, int size)
{
    int i;
    double mu = weibulparms[0]; /* scale */ 
    double sigma = weibulparms[1]; /* shape */ 

    double* z = (double*)malloc(sizeof(double)*size);
    double* expz = (double*)malloc(sizeof(double)*size);
    double* L = (double*)malloc(sizeof(double)*size);
    double logSigma;

    if (sigma <= 0) WEIBULL_ERROR_HANDLER(-1,"Bad sigma (shape) in weibull_neg_log_likelihood..");

    logSigma = log(sigma);

    for (i = 0; i < size; i++)
    {
        z[i] = (data[i]-mu)/sigma;
        expz[i] = exp(z[i]);
        L[i] = (z[i]-logSigma)*(1-censoring[i]-expz[i]);
    }

    /* Sum up the individual contributions, and return the negative log-likelihood. */
    for (i=0; i<size; i++)
    {
        *nlogL += (frequency[i]*L[i]);
    }

    *nlogL = *nlogL * -1;

    /*Compute the negative hessian at the parameter values, and invert to get */
    /*the observed information matrix. */
    {
      double* unc=(double*)malloc(sizeof(double)*size);
      double nH11=0;
      double nH12=0;
      double nH22=0;

      for (i=0; i<size; i++)
        {
          unc[i]=(1-censoring[i]);
          nH11=nH11+(frequency[i]*expz[i]);
        }
      
      for (i=0; i<size; i++)
        {
          nH12=nH12+(frequency[i] * ((z[i] + 1) * expz[i] - unc[i]));
          nH22=nH22+(frequency[i] * (z[i] *(z[i] + 2) * expz[i] - ((2 * z[i] + 1) *unc[i])));
        }
      
      {
        double sigmaSq = sigma * sigma;
        double avarDenom = (nH11*nH22 - nH12*nH12);

        acov[0]=sigmaSq*(nH22/avarDenom);
        acov[1]=sigmaSq*((-1*nH12)/avarDenom);
        acov[2]=sigmaSq*((-1*nH12)/avarDenom);
        acov[3]=sigmaSq*(nH11/avarDenom);
      }
      free(unc);
    }

    free(z);
    free(expz);
    free(L);
    return 0;
}

static double weibull_scale_likelihood(double sigma, double* x, double* w, double xbar, int size)
{
    double v;
    double* wLocal;
    int i;
    double sumxw;
    double sumw;

    wLocal=(double*)malloc(sizeof(double)*size);

    for (i=0; i<size; i++)
    {
        wLocal[i]=w[i]*exp(x[i]/sigma);
    }

    sumxw=0;
    sumw=0;

    for (i=0; i<size; i++)
    {
        sumxw+=(wLocal[i]*x[i]);
        sumw+=wLocal[i];
    }

    v = (sigma + xbar - sumxw / sumw);


    free(wLocal);
    return v;
}

/* based on dfzero from fortan, it finxs the zero in the given search bands, and stops if it is within tolerance. */
static int wdfzero(double* sigmahat, double* likelihood_value, double* err, double* search_bands, double tol,
                   double* x0, double* frequency, double meanUncensored, int size)
{
    double exitflag;
    double a,b,c=0.0,d=0.0,e=0.0,m,p,q,r,s;
    double fa,fb,fc;
    double fval;
    double tolerance;

    exitflag=1;
    *err = exitflag;

    a = search_bands[0];
    b = search_bands[1];

    fa = weibull_scale_likelihood(a,x0,frequency,meanUncensored,size);
    fb = weibull_scale_likelihood(b,x0,frequency,meanUncensored,size);

    if (fa == 0)
    {
        b=a;
        *sigmahat=b;
        fval = fa;
        *likelihood_value = fval;
        return 1;
    }
    else if (fb == 0)
    {   
        fval=fb;
        *likelihood_value = fval;
        *sigmahat=b;
        return 1;
    }
    else if ((fa > 0) == (fb > 0))
    {   
      WEIBULL_ERROR_HANDLER(-4,"ERROR: wdfzero says function values at the interval endpoints must differ in sign\n");
    }

    fc = fb;

    /*Main loop, exit from middle of the loop */
    while (fb != 0)
    {
      /* Insure that b is the best result so far, a is the previous */
      /* value of b, and that c is  on the opposite size of the zero from b. */
        if ((fb > 0) == (fc > 0))
        {
            c = a;  
            fc = fa;
            d = b - a;  
            e = d;
        }

        {
          double absFC;
          double absFB;
          
          absFC=fabs(fc);
          absFB=fabs(fb);
          
          if (absFC < absFB)
            {
              a = b;    
              b = c;    
              c = a;
              fa = fb;  
              fb = fc;  
              fc = fa;
            }
        }
          
        /*set up for test of Convergence, is the interval small enough? */
        m = 0.5*(c - b);

        {
          double absB,  absM,  absFA,absFB, absE;
          absB=fabs(b);
          absM=fabs(m);
          absFA=fabs(fa);
          absFB=fabs(fb);
          absE=fabs(e);
          
          {
            tolerance = 2.0*tol *((absB > 1.0) ? absB : 1.0);

          if ((absM <= tolerance) | (fb == 0.0))
            break;

          /*Choose bisection or interpolation */
          if ((absE < tolerance) | (absFA <= absFB))
            {
              /*Bisection */
              d = m; 
              e = m;
            }
          else
            {
              /*Interpolation */
              s = fb/fa;
              
              if (a == c)
                {
                  /*Linear interpolation */
                  p = 2.0*m*s;
                  q = 1.0 - s;
                }
              else
                {
                  /*Inverse quadratic interpolation */
                  q = fa/fc;
                  r = fb/fc;
                  p = s*(2.0*m*q*(q - r) - (b - a)*(r - 1.0));
                  q = (q - 1.0)*(r - 1.0)*(s - 1.0);
                }
              
              if (p > 0) 
                q = -1.0*q; 
              else
                p = -1.0*p;
            }
          }

              
          {
            double tempTolerance = tolerance*q;
            double absToleranceQ;
            double absEQ;
            double tempEQ = (0.5 * e * q);
            absToleranceQ=fabs(tempTolerance);
            absEQ=fabs(tempEQ);
            
            /*Is interpolated point acceptable */
            if ((2.0*p < 3.0*m*q - absToleranceQ) & (p < absEQ))
              {
                e = d;  
                d = p/q;
              }
            else
              {
                d = m;  
                e = m;
              }
          }
            
        } /*Interpolation */
          
        /*Next point */
        a = b;
        fa = fb;

        if (fabs(d) > tolerance) 
            b = b + d;
        else if (b > c) 
            b = b - tolerance;
        else
            b = b + tolerance;

        fb = weibull_scale_likelihood(b,x0,frequency,meanUncensored,size);

    }/*Main loop (While) */

    fval=weibull_scale_likelihood(b,x0,frequency,meanUncensored,size);
    *likelihood_value = fval;
    *sigmahat=b;

    return 1;
}

static int wnorminv(double* x, double* p,double *mu, double* sigma, int size)
{
    double* tempP = (double *)malloc(sizeof(double)*4);
    double* tempMU = (double *)malloc(sizeof(double)*4);
    double* tempSigma = (double *)malloc(sizeof(double)*4);

    tempP[0]=p[0];
    tempP[1]=p[0];
    tempP[2]=p[1];
    tempP[3]=p[1];

    tempMU[0]=mu[0];
    tempMU[1]=mu[0];
    tempMU[2]=mu[1];
    tempMU[3]=mu[1];

    tempSigma[0]=sigma[0];
    tempSigma[1]=sigma[0];
    tempSigma[2]=sigma[1];
    tempSigma[3]=sigma[1];

    {
      double myTemp;
      double tempVal1, tempVal2;
      double* x0 = (double*)malloc(sizeof(double)*4);

      myTemp=tempP[0];
      {
        double terfc=derfcinv(2*myTemp);
        if(terfc==-9999) WEIBULL_ERROR_HANDLER(-7,"wnorminv fails since derfcinv");
        tempVal1=(-1*sqrt((double)2))* terfc;
       
        myTemp=tempP[2];
        terfc=derfcinv(2*myTemp);
        if(terfc==-9999) WEIBULL_ERROR_HANDLER(-7,"wnorminv fails since derfcinv");
        tempVal2=(-1*sqrt((double)2))* terfc;
      }

      x0[0]=tempVal1;
      x0[2]=tempVal1;
      x0[1]=tempVal2;
      x0[3]=tempVal2;
      {
        int i;
        for (i=0; i<size; i++)
          {
            x[i]=tempSigma[i]*(x0[i])+tempMU[i];   
          }
      }
      free(x0);
    }

    free(tempP);
    free(tempMU);
    free(tempSigma);
    return 0;
}



  /* weibul  fitting is based on  methods developed for S/R and described
     NIST/SEMATECH e-Handbook of Statistical Methods, http://www.itl.nist.gov/div898/handbook/
     Lawless, J.F. (1982) Statistical Models and Methods for Lifetime Data, Wiley,
     New York.  and  Meeker, W.Q. and L.A. Escobar (1998) Statistical Methods for Reliability Data,         Wiley, New York. 
     with some checking and validation with various tools incldsdfasding R, S, MTLAB and
     http://www.engineeredsoftware.com/nasa/pe_weibull_mle.htm   (last accessed June 4 2012)

*/



#ifdef __cplusplus
extern "C" {
#endif

  /*
     weibull_fit does a maximum likelihood fitting to estimate the shape and scale parameters of a weibull probability distributon  @f[ \frac{shape}{scale} (\frac{x}{scale}e^-{{x/scale}^{shape}} @f]     
     
  @param weibullparms is an array of 2 doubles, which must be preallocated.  On successful completeion it will have shape and scale respectively.
  @param wparm_confidenceintervals is an array of 4 doubles, which must be preallocated.  On successful completeion it will have confidence interval for shape in the first two item and the CI for scale in the second two items
  @param inputData is a pointer the data to use for fitting the distribution. It must have at least size elements
  @param size is the size of the data to be used for fitting.
  @param alpha is parameter for Confidence interval size estimation. 
  @return return should be  1 if all went well. Values < 0 imply errors in fitting or data.  -1 means some data was negative, -2 means bad data range (e.g. all the same)  -3 or lower means MLE did not converge.

   */

int weibull_fit(double* weibullparms, double* wparm_confidenceintervals, double* inputData, double alpha, int size)
{


    double PI =  3.141592653589793238462;
    double FULL_PRECISION_MIN = 2.225073858507201e-308; /* smalled full precision positive number anything smaller is unnormalized, for testing for underflow */ 
    double FULL_PRECISION_MAX = 1.797693134862315e+308; /* largest full precision positive number, for testing for overflow */ 
    double  tol = 1.000000000000000e-006;/* this impacts the non-linear estimation..  if your problem is highly unstable (small scale) this might be made larger but we never recommend anything greater than 10e-5.  Also if larger it will converge faster, so if yo can live with lower accuracy, you can change it */
    double n;
    double nuncensored=0;
    double ncensored=0;
    int i;
    int code;

    double *censoring= (double *)malloc(sizeof(double)*size);
    double *frequency  = (double *)malloc(sizeof(double)*size);
    double * var = (double *)malloc(sizeof(double)*size);
    double* x0 =    (double *)malloc(sizeof(double)*size);

#ifndef WEIBULL_IGNORE_ERRORS
#ifdef WEIBULL_USE_ASSERTS
    assert(x0 != NULL);
#else
    if(x0== NULL)
      WEIBULL_ERROR_HANDLER( -1,"malloc failed in weibull_fit\n");
#endif
#endif




    /*set frequency to all 1.0's */
    /*and censoring to 0.0's */
    for (i=0; i< size; i++)
    {
        frequency[i]=1.0;
        censoring[i]=0.0;
    }
    /*  ********************************************** */
    for (i=0; i<size; i++)
    {
#ifndef WEIBULL_IGNORE_ERRORS
#ifdef WEIBULL_USE_ASSERTS
    assert(inputData[i]>0);
#else
    if(inputData[i]<=0) 
      WEIBULL_ERROR_HANDLER( -1,"cannot have data <=0  in call to weibull_fit\n");
#endif
#endif

        inputData[i]=log(inputData[i]);
    }
    /*  ********************************************** */
    {
      double mySum;
      
      mySum=0;
      for (i=0; i<size; i++)
        {
          mySum+=frequency[i];
        }
      
      n=mySum;
      if(n<=1) WEIBULL_ERROR_HANDLER(-2,"Insufficient distinct data in weibull_fit\n");
      /*  ********************************************** */
      {
        mySum=0;
        
        for (i=0; i<size; i++)
          {
            mySum+=(frequency[i]*censoring[i]);
          }
        
        ncensored=mySum;
        nuncensored = n - ncensored;
#ifndef WEIBULL_IGNORE_ERRORS
#ifdef WEIBULL_USE_ASSERTS
        assert(nuncensored>0);
        assert(n>1);
#else
    /* too much uncensored data means a plateau with no max */ 
        if(nuncensored<=0) WEIBULL_ERROR_HANDLER(-2,"Insufficient distinct data, hit a plateau in weibull_fit\n");
#endif
#endif

      }
    }

    /* declar local for max/range computation  ********************************************** */
    {
      double maxVal, minVal;
      double range, maxx;
      double tempVal;
      
      maxVal=-1000000000;
      minVal=1000000000;
      
      for (i=0; i<size; i++)
        {
          tempVal=inputData[i];
          
          if (tempVal < minVal)
            minVal=tempVal;
          
          if (tempVal > maxVal)
            maxVal=tempVal;
        }
      
      range = maxVal - minVal;
      maxx = maxVal;
#ifndef WEIBULL_IGNORE_ERRORS
#ifdef WEIBULL_USE_ASSERTS
      assert(range>0);
#else
      if(range<=0) WEIBULL_ERROR_HANDLER(-2,"Insufficient distinct data range in weibull_fit\n");
#endif
#endif
      /*Shift x to max(x) == 0, min(x) = -1 to make likelihood eqn more stable. */
      /*  ********************************************** */
      {
        double mean, myStd;
      double sigmahat;
      double meanUncensored;
      double upper, lower;
      double search_band[2];


      
      for (i=0; i<size; i++)
        {
          x0[i]=(inputData[i]-maxx)/range;
        }
      
      mean=0;
      myStd=0;
      
      for (i=0; i<size; i++)
        {
          mean+=x0[i];
        }
      
      mean/=n;

      for (i=0; i<size; i++)
        {
          var[i] = x0[i] - mean;
        }
      
      for (i=0; i<size; i++)
        {
          myStd+=var[i]*var[i];
        }
      
      myStd/=(n-1);
      myStd=sqrt(myStd);
      
      sigmahat = (sqrt((double)(6.0))*myStd)/PI;

      
      meanUncensored=0;

      for (i=0; i<size; i++)
        {
          meanUncensored+=(frequency[i]*x0[i])/n;
        }

      if ((tempVal=weibull_scale_likelihood(sigmahat,x0,frequency,meanUncensored,size)) > 0)
        {
          upper=sigmahat;
          lower=0.5*upper;
          
          while((tempVal=weibull_scale_likelihood(lower,x0,frequency,meanUncensored,size)) > 0)
            {
              upper = lower;
              lower = 0.5 * upper;
              
              if (lower < FULL_PRECISION_MIN)
                {
                  WEIBULL_ERROR_HANDLER(-3,"MLE in wbfit Failed to converge leading for underflow in root finding\n");
                }
            }
        }
      else
        {
          lower = sigmahat;
          upper = 2.0 * lower;
          
          while ((tempVal=weibull_scale_likelihood(upper,x0,frequency,meanUncensored,size)) < 0)
            {
              lower=upper;
              upper = 2 * lower;
              /* check for overflow, no finite root */
#ifndef WEIBULL_IGNORE_ERRORS
#ifdef WEIBULL_USE_ASSERTS
              assert(upper <= FULL_PRECISION_MAX);
#else
              if(upper > FULL_PRECISION_MAX) WEIBULL_ERROR_HANDLER(-3,"MLE in wbfit Failed to converge leading for overflow in root finding\n");
#endif
#endif
            }
        }
      /* ****************************************** */
      search_band[0]=lower;
      search_band[1]=upper;
      
      /* ... Next we  go find the root (zero) of the likelihood eqn which  wil be the MLE for sigma. */
      /* then  the MLE for mu has an explicit formula from that.  */
      
      {
        double err;
        double likelihood_value;
        
        
        code = wdfzero(&sigmahat,&likelihood_value,&err,search_band,tol,x0,frequency,meanUncensored,size);
        
#ifndef WEIBULL_IGNORE_ERRORS
#ifdef WEIBULL_USE_ASSERTS
        assert(code == 1);
#else
        if(code != 1) WEIBULL_ERROR_HANDLER(-4, "weibull_fit failed, could not find solution in MLE. Probably has insufficnt data distribution (e.g. all same value)...\n");
#endif
#endif
      }

  /* ****************************************** */
        {
          double muHat;
          double sumfrequency;
          
          muHat=0;
          sumfrequency=0;
          
          for (i=0; i<size; i++)
            {
              tempVal=exp(x0[i]/sigmahat);
              sumfrequency +=(frequency[i]*tempVal);
            }
          
          sumfrequency = sumfrequency / nuncensored;
          muHat = sigmahat * log(sumfrequency);

  /* ****************************************** */

          /*Those were parameter estimates for the shifted, scaled data, now */
          /*transform the parameters back to the original location and scale. */
          weibullparms[0]=(range*muHat)+maxx;
          weibullparms[1]=(range*sigmahat);
        }
    }
    }

    {
          int rval;
          double nlogL=0, tempVal;
          double transfhat[2], se[2], probs[2],acov[4]; 

          probs[0]=alpha/2;
          probs[1]=1-alpha/2;
          /* ****************************************** */
          
          
          rval=weibull_neg_log_likelihood(&nlogL,acov,weibullparms,inputData,censoring,frequency,size);
          if(rval<0) WEIBULL_ERROR_HANDLER(-5,"Failed to fine final parameters settings MLE failed. Memory leaked");

          /* ****************************************** */
          /*Compute the Confidence Interval (CI)  for mu using a normal approximation for muhat.  Compute */
          /*the CI for sigma using a normal approximation for log(sigmahat), and */
          /*transform back to the original scale. */
          
          transfhat[0]=weibullparms[0];
          transfhat[1]=log(weibullparms[1]);
          
          se[0]=sqrt(acov[0]);
          se[1]=sqrt(acov[3]);
          se[1]=se[1]/weibullparms[1];
          
          rval=wnorminv(wparm_confidenceintervals,probs,transfhat,se,4);
          if(rval<0) WEIBULL_ERROR_HANDLER(-7,"Cannot compute confidence interval since wnorminv fails. Memory leaked");

          wparm_confidenceintervals[2]=exp(wparm_confidenceintervals[2]);
          wparm_confidenceintervals[3]=exp(wparm_confidenceintervals[3]);

          tempVal=wparm_confidenceintervals[2];
          wparm_confidenceintervals[2]=1/wparm_confidenceintervals[3];
          wparm_confidenceintervals[3]=1/tempVal;
          
          wparm_confidenceintervals[0]=exp(wparm_confidenceintervals[0]);
          wparm_confidenceintervals[1]=exp(wparm_confidenceintervals[1]);

          weibullparms[0]=exp(weibullparms[0]);
          weibullparms[1]=1/weibullparms[1];
    }

          /*free all memory */
    free(x0);
    free(var);
    free(censoring);
    free(frequency);
    return 1;
}

#ifdef __cplusplus
}
#endif
