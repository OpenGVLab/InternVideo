/*! \file
 * weibull.h provides the headers for the  core functionality for the internal computing weibull fittings, as well as CDF and INF given parameters 
 * this file is not intended for direc usage...
 *  
 * 
 * Author Brian Heflin bheflin  at securics com
 * Author Walter Scheirer walter at securics com
 * Author Terry Boult tboult  at securics com
 *
 * Copyright 2010, 2011, Securics Inc.
 *
 * @section LICENSE
 *  See accompanying LICENSE agreement for full details on rights.
 *
 * Parts of this technology are subject to SBIR data rights and as described in DFARS 252.227-7018 (June 1995) SBIR Data Rights which apply to Contract Number: N00014-11-C-0243 and STTR N00014-07-M-0421 to Securics Inc, 1870 Austin Bluffs Parkway, Colorado Springs, CO 80918
 *
 *The Government's rights to use, modify, reproduce, release, perform, display, or disclose technical data or computer software marked with this legend are restricted during the period shown as provided in paragraph (b)(4) of the Rights in Non-commercial Technical Data and Computer Software-Small Business Innovative Research (SBIR) Program clause contained in the above identified contract.  Expiration of SBIR Data Rights: Expires four years after completion of the above cited project work for this or any other follow-on SBIR contract, whichever is later.
 *
 * No restrictions on government use apply after the expiration date shown above.  Any reproduction of technical data, computer software, or portions thereof marked with this legend must also reproduce the markings.
 *
 * @section Summary Description
 * This file contains the "C" interface functions for very basic Weibull usage for Meta-Recognition.   The weibull_fit and weibull_cdf are the primary functions to use. 
 *
 * The code herein has a number of STRONG assumptions you must follow as we cannot test for all of them which is why we don't recommend use it directly
 *   1) All fitting and testing are presuming  "larger is better",  If you are fitting something where smaller is better you need to transform it.
 *   2) All data is positive (okay we can and do test for that, but better to know up front what you are doing) 
 *   3) There must be sufficient range in your data to actually fit the Weibull.  If all the data is the same, or nearly the same, it may fail to converge and will report errors.
 *   4) For efficient fitting, we must satisfy a regularity condition (see N. M. Kiefer, Maximum likelihood estimation (MLE),  http://instruct1.cit.cornell.edu/courses/econ620/reviewm5.pdf, 2007),  and to do that the lower bound in Weibull data/fitting cannot be too small  so we recommend you translated the data to be well away from zero (part of why we only fit on upper side and, in the MetaRecognition class we translate more than just the min..)
 * 
 *
 */

#pragma once
#ifndef WEIBULL_H
#define WEIBULL_H

#ifdef _WIN32
#ifdef __cplusplus
extern "C" {
#endif
_declspec(dllexport) double weibull_inv(double x, double scale, double shape);
_declspec(dllexport) double weibull_cdf(double x, double scale, double shape);
_declspec(dllexport) int weibull_fit(double* weibull_parms, double* wparm_confidenceintervals, double* inputData, double alpha, int size);
_declspec(dllexport) void printWeibullBuildInfo(FILE *fh);
#ifdef __cplusplus
}
#endif
#else
#ifdef __cplusplus
extern "C" {
#endif

  /** if WEIBULL_USE_ASSERTS is defined, the code will use asserts to ensure its requirements are true, otherwise it returns error codes. Default is not defined */ 
  /** if WEIBULL_IGNORE_ERRORS is defined, the code will just presume things will work out and not waste time on testing for error. Default is not defined */ 


  /*#define WEIBULL_USE_ASSERTS  //!< \def define this to force asserts rather than error codes. */
  /*#define WEIBULL_IGNORE_ERRORS //!< \def defien this to skip printing/return code for errors */


  /**  weibull_cdf computes the probability (given our assumptions) that the value x is an outlier ABOVE the fit distribution.  if the distribution was non-match data, then it provides this probability that x is a match score.   If data was match-data then it would be the probability of it being a larger non-match. 
  computes @f[ 1-e^{{\frac{x}{scale}}^{shape}} @f]

  @param x  the location at which to compute the probability of being an outlier
  @param scale the scale parmaeter of the weibull.  This is the first element in weibull_parms (as computed by our wlbfit) 
  @param shape the scale parmaeter of the weibull.  This is the first second in weibull_parms (as computed by our wlbfit) 
  @return if in the range [0-1] it is the probability of X being an outlier.  Any value < 0 is an error code.  returns -1 for invalid scale <=0 ,  -2 for invalid shape <=0 
  *
  */ 
double weibull_cdf(double x, double scale, double shape);


  /**  weibull_inv computes the inverse weibull, i.e. returns the score S (given our assumptions) such that x=wlbcdf(s,scale,shape). Note it estimates from above, so if x=1.0 expect an answer of Inf (infinity). 

  @param x  the location at which you compute the inverse (must be between [0,1]
  @param scale the scale parmaeter of the weibull.  This is the first element in weibull_parms (as computed by our wlbfit) 
  @param shape the scale parmaeter of the weibull.  This is the first second in weibull_parms (as computed by our wlbfit) 
  @return if X in the range [0-1], return S such that x=wlbcdf(s,scale,shape).  The return value is in the range [0,Inf].  Any return value < 0 is an error code.  returns -1 for invalid scale <=0 ,  -2 for invalid shape <=0  -3 for  X<0, -4 for x >1
  *
  */ 
double weibull_inv(double x, double scale, double shape);

  /**
     weibull_fit does a maximum likelihood fitting to estimate the shape and scale parameters of a weibull probability distributon  @f[ \frac{shape}{scale} \left(\frac{x}{scale} \cdot e^{-{\left(\frac{x}{scale}\right)}^{shape}}\right)@f]     
     
  @param weibull_parms is an array of 2 doubles, which must be preallocated.  On successful completeion it will have shape and scale respectively.
  @param wparm_confidenceintervals is an array of 4 doubles, which must be preallocated.  On successful completeion it will have confidence interval for shape in the first two item and the CI for scale in the second two items
  @param inputData is a pointer the data to use for fitting the distribution. It must have at least size elements
  @param size is the size of the data to be used for fitting.
  @param alpha is parameter for Confidence interval size estimation. 
  @return return should be  1 if all went well. Values < 0 imply errors in fitting or data.  -1 means some data was negative, -2 means bad data range (e.g. all the same)  -3 or lower means MLE did not converge.

   */
int weibull_fit(double* weibullparms, double* wparm_confidenceintervals, double* inputData, double alpha, int size);


  /**
     Print information about this build to a file descriptor.  Used for checking what is loaded for supporting people
  */
void printWeibullBuildInfo(FILE *fh); 
#ifdef __cplusplus
}
#endif

#endif
#endif
