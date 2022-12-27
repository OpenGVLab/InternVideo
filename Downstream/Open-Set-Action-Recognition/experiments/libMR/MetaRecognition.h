/**
 * MetaRecognition.h: 

 * @Author Terry Boult tboult at securics com
 * @Author Vijay Iyer viyer at securics com

 *
 * Copyright 2010, 2011, Securics Inc.

 * Copyright 2011, Securics Inc.
   See accompanying LICENSE agrement for details on rights.

Parts of this technology are subject to SBIR data rights and as described in DFARS 252.227-7018 (June 1995) SBIR Data Rights which apply to Contract Number: N00014-11-C-0243 and STTR N00014-07-M-0421 to Securics Inc, 1870 Austin Bluffs Parkway, Colorado Springs, CO 80918

The Government's rights to use, modify, reproduce, release, perform, display, or disclose technical data or computer software marked with this legend are restricted during the period shown as provided in paragraph (b)(4) of the Rights in Noncommercial Technical Data and Computer Software-Small Business Innovative Research (SBIR) Program clause contained in the above identified contract.  Expiration of SBIR Data Rights: Expires four years after completion of the above cited project work for this or any other follow-on SBIR contract, whichever is later.

No restrictions on government use apply after the expiration date shown above.  Any reproduction of technical data, computer software, or portions thereof marked with this legend must also reproduce the markings.
 *
*/

#pragma once
#ifndef MetaRecognition_H
#define MetaRecognition_H


#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <stdio.h>
#include <istream>
#include <ostream>
#include <iostream>
#include <sstream>
#include <string>


#include "weibull.h"

#ifdef _WIN32
#define DLLEXPORT _declspec(dllexport)
#else 
#define DLLEXPORT
#endif

#define MAX_LINE 256

/// structure for svm data used by libSVM, used to  allow easy MetaRecognition for SVM results (used as an argument for MetaRecogniton::fitSVM)
struct svm_node_libsvm
{
  int index;  //!< class label,  classic is -1 for negative class add +1 for positive class, but can be general its for multi-class
  double value;//!< the SVM decision score
};

/**!
  Class MetaRecognition provides a object-based interface for Meta-Recognition.  The object can be ...
 
  TBD


*/
class DLLEXPORT MetaRecognition //!  Primary object/methods for tranforming and computing needed for any Meta recogntion task 
{
public:

/**  Ctor,  can call with no arguments (uses default arguments for construciton).  
     All space is on the stack. 
     Object will exist but is not valid until some fitting fucntion is called 
*/

  MetaRecognition( int scores_to_drop=0,   //!< is this object for prediction, if so  how many top scores to drop when fitting
                   int fitting_size=9,     //!< tail size for fitting.  With small data the defaults are fine.. if you have millions make it larger for better predictions 
                   bool verbose = false,    //!< is the code chatty on errors during fitting, useful for debugging
                   double alpha=5.0,        //!< band for confidence interfals
                   int translate_amount=10000 //!< shifting data to ensure all is positive.. if data is very broad and you want some probabilities for all points you can make it larger.. 
                   );

	~MetaRecognition();

        bool is_valid(); //!< is this object valid..i.e. has data been properly fit to determine parameters.
        void set_translate(double t); //!< Change translate_amount to x, invalidates object

        void Reset(); //!< reset to "invalid" state

	bool Predict_Match(double x, double threshold = .9999999);     //!< Is X from the "match" distribution (i.e. we reject null hypothesis of non-match), 
	double W_score(double x); //!< This is the commonly used function.. after fitting, it returns the probability of the given score being "correct".  It is the same as CDF
	double CDF(double x);     //!< This is the cummumlative probablity of match being corrrect (or more precisely the probility the score (after transform) being an outlier for the distribution, which given the transforms applied, so bigger is better, this is the probablity the score is correct. 
	double Inv(double p);     //!< This is score for which one would obtain CDF probability p (i.e. x such that p = CDF(x))

	int ReNormalize(double *invec, double *outvec, int length);     //!< W-score Renormalize the vecotor invec[0:length-1] into outvec (in and out can be same) return is 1 for success, <0 for error code


        /// Use FitHight if your data is such that is larger is better.  The code will still transform, and keep parmeters to keep small data away from zero.  
        // If you get scores that are complain about it being negative, make a MR object with different (larger) translate amount
        /// returns 1 for success, <0 for error code
	int FitHigh(double* inputData, int inputDataSize,  int fit_size=-1); 

        ///Use FitLow if your data is such that smaller scores are better.. we'll transform it for you and keep the
        ///transform parameters in the class so later calls to W_score or CDF do the right thing.  
        /// returns 1 for success, <0 for error code
	int FitLow(double* inputData, int inputDataSize,  int fit_size=-1);// 

        /// the types of fitting supported for SVM modeling 
        typedef enum  {complement_reject=1, positive_reject=2, complement_model=3, positive_model=4} MR_fitting_type; 

        /// The function to use if you have SVM data, it separated out the data for the label of interst (or rejecting
        /// the complement of that label, which is the default) and uses that for fitting.  
        /// Returns 1 if it worked, <0 for error codes. 
        int FitSVM(svm_node_libsvm* SVMdata, int inputDataSize, int label_of_interest =1, bool label_has_positive_score=true, int fit_type = 1, int fit_size=9 ); 


        friend std::ostream& operator<<( std::ostream&, const MetaRecognition& );         //!< various I/O functions
        friend std::istream& operator>>( std::istream&, MetaRecognition& );        //!< various I/O functions

	void Save(std::ostream &outputStream) const;         //!< various I/O functions
	void Load(std::istream &inputStream);        //!< various I/O functions
	void Save(FILE *outputFile) const;        //!< various I/O functions
	void Load(FILE *inputFile);        //!< various I/O functions
	void Save(char* filename) const;        //!< various I/O functions
	void Load(char* filename);        //!< various I/O functions
        int get_fitting_size();  //!<  Get get_fitting_size (aka tail size)
        int set_fitting_size(int nsize);  //!<  reset object and define new fitting size
        int get_translate_amount();  //!<  Get get_internal tranlation amount (you probably don't need this, but just in case)
        int set_translate_amount(int ntrans);  //!<  reset object and define new translate amount.. if you get errors because of negative data, increase this
        int get_sign();   //!<  Get get_internal sign variable. (you probably don't need this, but just in case)
        int set_sign(int nsign); //!<  reset object and  set sign  (you probably don't need this, but just in case)
        double get_small_score();   //!<  Get get_internal smaller translation amount (you probably don't need this, but just in case)
        double set_small_score(double nscore); //!<  reset object and  reset internal smaller translation amount (you probably don't need this, but just in case)
        bool verbose;  //!<  do we print internal/debugging stuff.  Default is false. (you probably don't need this, but just in case)
        std::string to_string(); //!< Convert this object to a C++ string
        void from_string(std::string in); //!< Convert this object from a C++ string

protected:
        int EvtGeneric(double* inputData, int inputDataSize, int fit_inward=0, double x=0);
	double parmhat[2];          //!<  parameters of the Weibull,  scale then shape
	double parmci[4];    //!< confidence interval for parms  scale high, scale low, shape high, shape low
	double alpha;  //!< parameter for estimation of size of confidence interval
	int sign;   //!< sign is postive is larger is better,  negative means orginally smaller was better (we transformed for fitting).
        MR_fitting_type ftype;  //!< type of fitting used for SVM.. default is reject complement
	int fitting_size;   //!< tail size for fitting in any of the FitXX functions
	int translate_amount; //!< we transform data so all fittng data data is positive and bigger is better, this predefined constant helps ensure more of the end-user data is non-negative.  
	double small_score;   //!< the smallest score, so all fitting data is consistently postive. part of our transform
	int scores_to_drop; //!< when fitting for recognition prediction, how many top score are hypothesized to be a match, so we can fit on non-match data.  Only used in for fitting, no impact on transform. 
        bool isvalid; //!< is the parameters in the object valid. private:

};

#endif
