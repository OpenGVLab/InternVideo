/*
 * MetaRecognition.cpp
 * Copyright 2011, Securics Inc.
   See accompanying LICENSE agrement for details on rights.

Parts of this technology are subject to SBIR data rights and as described in DFARS 252.227-7018 (June 1995) SBIR Data Rights which apply to Contract Number: N00014-11-C-0243 and STTR N00014-07-M-0421 to Securics Inc, 1870 Austin Bluffs Parkway, Colorado Springs, CO 80918

The Government's rights to use, modify, reproduce, release, perform, display, or disclose technical data or computer software marked with this legend are restricted during the period shown as provided in paragraph (b)(4) of the Rights in Noncommercial Technical Data and Computer Software-Small Business Innovative Research (SBIR) Program clause contained in the above identified contract.  Expiration of SBIR Data Rights: Expires four years after completion of the above cited project work for this or any other follow-on SBIR contract, whichever is later.

No restrictions on government use apply after the expiration date shown above.  Any reproduction of technical data, computer software, or portions thereof marked with this legend must also reproduce the markings.
 *
*/

/** \mainpage

  
    This library provides support for meta-recognition, i.e. recognizing when a recognition system is working well and when it is not and using that self-knowledge to improve the system.    It can be used for prediction of failure,  fusion,  score renormalization, SVM renormalization and converting SVM or recognition scores into statistially well supported probility estimtes.  The analysis is based on an analysis of the recognition system scores. 


The fundamental ideas are described in 

"Meta-Recognition: The Theory and Practice of Recognition Score Analysis,"
Walter J. Scheirer, Anderson Rocha, Ross Micheals, Terrance E. Boult,
IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI),
33(8), pp 1689--1695, Aug, 2011.

and SVM support as described in 

"Multi-Attribute Spaces: Calibration for Attribute Fusion and Similarity Search,"
Walter J. Scheirer, Neeraj Kumar, Peter N. Belhumeur, Terrance E. Boult,
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
June 2012.
 

The underlying extream value theory provide stong theortical basis for the computations, but to make it useful one must transform the data into the proper frame.   The C++ version provides objects that can compute and store information about the transform and then provide for prediction, w-score values (probability estimates), or  renormalizatoin of a vector of data. 
   
  The library also contains a  "C" interface functions for very basic weilbull usage for Meta-Recognition.    
  The C-based library  has a number of STRONG assumptions you must follow as we cannot test for all of them.
    1) All fitting and testing are presuming  "larger is better",  If you are fitting something where smaller is better you need to transform it before fitting. 
    2) All data is positive (okay we can and do test for that, but better to know upfront what you are doing) 
    3) There must be sufficient range in your data to actually fit the weilbull.  If all the data is the same, or nearly the same, it may fal to converge and will report errors.

    While free for non-commercial use this library is subject to the license restrictions, see LICENSE.TXT  for details.  
    
*/

#include "MetaRecognition.h"
#include <string.h> 
//#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif
extern int weibull_fit_verbose_debug;
#ifdef __cplusplus
}
#endif
MetaRecognition::MetaRecognition(int scores_to_dropx,  int fitting_sizex, bool verb, double alphax, int translate_amountx):
  scores_to_drop(scores_to_dropx),verbose(verb),fitting_size(fitting_sizex),alpha(alphax),translate_amount(translate_amountx)
{
  memset(parmhat,0,sizeof(parmhat));
  memset(parmci,0,sizeof(parmci));
  sign = 1;
  ftype = complement_reject;
  small_score=0;
  isvalid=false;
  if(verb) weibull_fit_verbose_debug=1;
  else weibull_fit_verbose_debug=0;
}

MetaRecognition::~MetaRecognition()
{
  //  free(parmhat);
  //  free(parmci);
}

bool MetaRecognition::is_valid(){
  return isvalid;
}

void MetaRecognition::set_translate(double t){
  translate_amount = t;
  isvalid=false;
};


void MetaRecognition::Reset(){
  memset(parmhat,0,sizeof(parmhat));
  memset(parmci,0,sizeof(parmci));
  sign = 1;
  scores_to_drop = 0;
  small_score=0;
  isvalid=false;
}


int compare_sort_decending (const void * a, const void * b)
{
	const double *da = (const double *) a;
	const double *db = (const double *) b;
	return  (*da < *db) - (*da > *db);
}

int compare_sort_assending (const void * a, const void * b)
{
	const double *da = (const double *) a;
	const double *db = (const double *) b;
	return   (*da > *db) - (*da < *db);
}

inline const char * const BoolToString(bool b)
{
	return b ? "true" : "false";
}

inline int  const BoolToInt(bool b)
{
	return b ? 1 : 0;
}

inline const bool IntToBool(const char * s)
{
  int val= atoi(s);
  if(val) return true;
  else return false;
}

//Wraps calls to real weibull_inv and weibull_cdf functions and handles properly translating the data passed
//May eventually be a good idea to move real implementations of the functions here
//IF we do away with the C implementation. For now this allows for backward compantiblity with
//older code
// Inv computes the scores of the inverse CDF, i.e. returns y such that  CDF(y) = 
double MetaRecognition::Inv(double x)
{
  if(!isvalid) return -9999.0;
  double score = weibull_inv(x, parmhat[0], parmhat[1]);
  return (score - translate_amount + small_score)*sign;
}

double MetaRecognition::CDF(double x)
{
  if(!isvalid) return -9999.0;
  double translated_x = x*sign + translate_amount - small_score;
  double wscore=weibull_cdf(translated_x, parmhat[0], parmhat[1]);
  if(ftype==complement_model || ftype==positive_model) return 1-wscore;
  return wscore;
};

double MetaRecognition::W_score(double x){
    return CDF(x);
};

bool MetaRecognition::Predict_Match(double x, double threshold){
  double score = Inv(threshold);
  if(sign <0)   return (x < score);
  return (x > score);
};

int MetaRecognition::ReNormalize(double *invec, double *outvec, int length)
{
  if(!isvalid) return -9997.0;
  int rval=1;
  for(int i=0; i< length; i++){
    outvec[i] = W_score(invec[i]);
  }
  return rval;
}


//used by weibull__evt_low and weibull__evt_high, which sets the desired sign(low -1, high 1)
//before passing to generic
int MetaRecognition::EvtGeneric(double* inputData, int inputDataSize, int inward, double x)
{
  double * inputDataCopy = (double *) malloc(sizeof(double) * inputDataSize);

  double * dataPtr = NULL;
  int icnt=0;
  if(!inward && (sign > 0) ) {
    icnt = inputDataSize;
    memcpy(inputDataCopy,inputData, inputDataSize*sizeof(double));
  }
  if(!inward && (sign < 0) ){
    for(int i=0; i < inputDataSize; i++)       inputDataCopy[i] = (inputData[i]*sign);       //doing extremes just flip sign if needed
    icnt = inputDataSize;
  }
  else if(inward  && (sign < 0)) { /* this is fit above x but  approaching x */ 
    for(int i=0; i < inputDataSize; i++)       {
      if(inputData[i] > x) {
        inputDataCopy[icnt++] = (inputData[i]*sign);       //copy what is above x, and flip signs (so biggest is important)
      } 
    }
  } else if(inward  && (sign > 0)) { /* this is fit below x but  approaching x */ 
      for(int i=0; i < inputDataSize; i++)       {
        if(inputData[i] < x) {
          inputDataCopy[icnt++] = (inputData[i]);       //copy only what is above x. 
        } 
      }
  } 

  //sort data and get smallest score
  qsort(inputDataCopy, icnt , sizeof(double), compare_sort_decending);

  //Want only the top fitting_size scores but als noneed to adap if dropping top score
  if(scores_to_drop>0){
    dataPtr=inputDataCopy+scores_to_drop;
  } else {
    dataPtr=inputDataCopy;
  }

  small_score = dataPtr[fitting_size-1];
  
  for(int i=0; i < fitting_size; i++)
    {	
      //translate and subtract small score
      dataPtr[i] = dataPtr[i] + translate_amount - small_score;
    }
  
 
  int rval =   weibull_fit(parmhat, parmci, dataPtr, alpha, fitting_size);
  isvalid= true;
  if(rval != 1) Reset();
  free(inputDataCopy);
  return rval;
}

//Wrapper fitting functions EvtLow and EvtHigh to make it simpler for new users of the library.
int MetaRecognition::FitLow(double* inputData, int inputDataSize, int fsize)
{
  if(fsize>0)    fitting_size=fsize;
  sign = -1;
  return EvtGeneric(inputData, inputDataSize);
}

int MetaRecognition::FitHigh(double* inputData, int inputDataSize, int fsize)
{
  if(fsize>0)    fitting_size=fsize;
  sign = 1;
  return EvtGeneric(inputData, inputDataSize);
}

int MetaRecognition::FitSVM(svm_node_libsvm* SVMdata, int inputDataSize, int label_of_interest, bool label_has_positive_score, int fit_type, int fit_size )
{

  Reset();
  ftype = (MR_fitting_type)fit_type;
  fitting_size = fit_size;
  double * inputDataCopy = (double *) malloc(sizeof(double) * inputDataSize);
  int sign_of_label_of_interest=0;
  double * dataPtr = NULL;
  int sign_of_expected_score=-1;
  if(label_has_positive_score) sign_of_expected_score=1;

  int icnt=0;
  bool rejection=(ftype==complement_reject || ftype == positive_reject);
  if(rejection) {  // default we fit on the complement class and then do rejection to determine probability
    for(int i=0; i < inputDataSize; i++) {
      if(SVMdata[i].index != label_of_interest) inputDataCopy[icnt++] = (SVMdata[i].value);       //doing extremes just flip sign if needed
      else {
        if(SVMdata[i].value >0) sign_of_label_of_interest++;
        else sign_of_label_of_interest--;
      }
    }
  } else {
    for(int i=0; i < inputDataSize; i++) {
      if(SVMdata[i].index == label_of_interest) inputDataCopy[icnt++] = (SVMdata[i].value);       //doing extremes just flip sign if needed
      else {
        if(SVMdata[i].value >0) sign_of_label_of_interest++;
        else sign_of_label_of_interest--;
      }
    }
  }
  if (verbose && sign_of_label_of_interest * sign_of_expected_score > 0){
    printf("In MetaRecognition::FitSVM,  warning: possible inconsistency average of the non-matching data has score %d, but expected sign is %d\n",
           sign_of_label_of_interest, -sign_of_expected_score);
  }


  /* expected sign combines with reject_complement to determine if we have to flip or not.
     We flip if positives scores, with smaller is better, is the goal, 
     we flip if sign_of_expected_score >0 and !force_rejection
     we flip if sign_of_expected_score <0 and force_rejection */

  if((!label_has_positive_score  && rejection)
     || (label_has_positive_score  && !rejection)) {
    sign = -1;
    for(int i=0; i < icnt; i++) {
      inputDataCopy[i] *= -1;       //doing extremes just flip sign if needed
    }
  } else sign=1;

  //sort data and get smallest score
  qsort(inputDataCopy, icnt , sizeof(double), compare_sort_decending);

  //Want only the top fitting_size scores but als noneed to adap if dropping top score
  if(scores_to_drop){
    dataPtr=inputDataCopy+scores_to_drop;
  } else {
    dataPtr=inputDataCopy;
  }

  small_score = dataPtr[fitting_size - 1];
  
  for(int i=0; i < fitting_size; i++)
    {	
      //translate and subtract small score
      dataPtr[i] = dataPtr[i] + translate_amount - small_score;
    }
  
  int rval = weibull_fit(parmhat, parmci, dataPtr, alpha, fitting_size);

  isvalid= true;
  if(rval != 1) Reset();
  free(inputDataCopy);
  printf("Completed weibull fitting\n");  
  return rval;
};

void MetaRecognition::Save(std::ostream &outputStream) const
{
	if(outputStream.good() && isvalid)
	{
		try {
		outputStream.precision(21);
		outputStream.setf(std::ios::scientific,std::ios::floatfield); 
		outputStream << parmhat[0] << " " << parmhat[1] <<   "  "  
                             << parmci[0] << " " << parmci[1] << " " 
                             << parmci[2] << " " << parmci[3] << "  " 
                             << sign << " " 
                             << alpha << " " 
                             << (int) ftype << " " 
                             << fitting_size << " " 
                             << translate_amount << " " 
                             << small_score<< " "
                             << scores_to_drop
                             << std::endl;
		} catch(std::bad_alloc& e) {
			std::cout << "Could not allocate the required memory, failed with error: '" << e.what() << "'" << std::endl;
		}
	}
}

std::ostream& operator<< ( std::ostream& os, const MetaRecognition& mr )
  {
    mr.Save(os);
    return os;
  }

std::istream& operator>> ( std::istream& is, MetaRecognition& mr )
  {
    mr.Load(is);
    return is;
  }


void MetaRecognition::Load(std::istream &inputStream)
{
  isvalid=false;
  int temp;
  if(inputStream.good())
    {
      int iftype;
      inputStream >> parmhat[0] >> parmhat[1]
                  >> parmci[0] >> parmci[1]
                  >> parmci[2] >> parmci[3]
                  >> sign 
                  >> alpha 
                  >> iftype 
                  >> fitting_size 
                  >> translate_amount 
                  >> small_score
                  >> scores_to_drop;
      isvalid=true;
      ftype =  (MR_fitting_type) iftype;
    }
}

void MetaRecognition::Save(FILE *outputFile) const
{
	if((outputFile != NULL) && !feof(outputFile))
	{
          fprintf(outputFile, 
                  "%21.18g %21.18g  " //parmaht
                  "%21.18g %21.18g " //parmci 
                  "%21.18g %21.18g  "
                  "%d %f %d %d "  //sign, alpha, fitting size
                  "%d %21.18g %d\n", //translate,  small_score, scores_to_drop
                  parmhat[0], parmhat[1],
                  parmci[0],parmci[1],
                  parmci[2],parmci[3],
                  sign, alpha, (int) ftype,fitting_size,
                  translate_amount, small_score, scores_to_drop);
	}
}

void MetaRecognition::Load(FILE *inputFile)
{
  int temp, iftype;
  int retcode=0;
  isvalid=false;
  if((inputFile != NULL) && !feof(inputFile))
    {
      
      retcode = fscanf(inputFile, 
                       "%lf %lf " //parmaht
                       "%lf %lf " //parmci 
                       "%lf %lf "
                       "%d %lf %d %d "  //sign, alpha, fitting size
                       "%d %lf %d ", //translate, small_score, scores_to_drop, 
                       parmhat, parmhat+1,
                       parmci,parmci+1,
                       parmci+2,parmci+3,
                       &sign, &alpha, &iftype, &fitting_size,
                       &translate_amount, &small_score, &scores_to_drop);
      isvalid=true;
      ftype =  (MR_fitting_type) iftype;
    }
}


void MetaRecognition::Save(char* filename) const
{
  FILE*  fp = fopen(filename,"w");
  if(fp) {
    Save(fp);
    fclose(fp);
  } else if(strlen(filename)>0) 
    fprintf(stderr,"SaveWeibull could not open file |%s|\n",filename);
  else     fprintf(stderr,"SaveWeibull called with null filename\n");
}

void MetaRecognition::Load(char* filename){
  FILE*  fp = fopen(filename,"r");
  isvalid=false;
  if(fp) {
    Load(fp);
    isvalid=true;
    fclose(fp);
  } else if(strlen(filename)>0) 
    fprintf(stderr,"LoadWeibull could not open file |%s|\n",filename);
  else     fprintf(stderr,"LoadWeibull called with null filename\n");

}

std::string MetaRecognition::to_string() {
    std::stringstream oss;
    this->Save(oss);
    return oss.str();
}
void MetaRecognition::from_string(std::string input) {
    std::stringstream iss(input);
    this->Load(iss);
}


int MetaRecognition::set_fitting_size(int nsize){ isvalid=false; return fitting_size=nsize;}
int MetaRecognition::get_fitting_size(){ return fitting_size;}
int MetaRecognition::get_translate_amount(){ return translate_amount;}
int MetaRecognition::set_translate_amount(int ntrans) {isvalid=false; return translate_amount=ntrans;}
double MetaRecognition::get_small_score(){return small_score;}
double MetaRecognition::set_small_score(double nscore){isvalid=false;  return small_score=nscore;}
int MetaRecognition::get_sign(){return sign;}
int MetaRecognition::set_sign(int nsign){return sign=nsign;}
