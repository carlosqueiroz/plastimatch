#ifndef _RANSAC_H_
#define _RANSAC_H_

#include <set>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits>
#include "ParametersEstimator.h"
#if ITK_VERSION_MAJOR >= 5
#include <mutex>
#include "itkPlatformMultiThreader.h"
#else
#include "itkMultiThreader.h"
#include "itkSimpleFastMutexLock.h"
#endif
// #include "itkExceptionObject.h"
#include "itkMacro.h"

/**
 * This class implements a multi-threaded version of the RAndom SAmple 
 * Consensus (RANSAC) framework, a framework for robust parameter estimation.
 * Given data containing outliers we estimate the model parameters using subsets 
 * of the original data:
 * 1. Choose the minimal subset from the data for computing the exact model 
 *    parameters.
 * 2. See how much of the input data agrees with the computed parameters.
 * 3. Goto step 1. This can be done up to (N choose m) times, where m is the 
 *    number of data objects required for an exact estimate and N is the total 
 *    number of data objects.
 * 4. Take the largest subset of objects which agreed on the parameters and 
 *    compute a least squares fit using them.
 * 
 * This is based on:
 * Fischler M.A., Bolles R.C., 
 * "Random Sample Consensus: A Paradigm for Model Fitting with Applications to 
 * Image Analysis and Automated Cartography", 
 * Communications of the ACM, Vol. 24(6), 1981.
 *
 * Hartely R., Zisserman A., "Multiple View Geometry in Computer Vision", 2001.
 *
 * The class template parameters are T - objects used for the parameter estimation 
 *                                      (e.g. Point2D in line estimation, 
 *                                            std::pair<Point2D,Point2D> in 
 *                                            homography estimation).
 *                                   S - type of parameter (e.g. double).                          
 *
 * @author: Ziv Yaniv (zivy@isis.georgetown.edu)
 *
 */

namespace itk {

template<class T, class S>
class RANSAC : public Object {
public:
  
    typedef RANSAC                     Self;
    typedef Object                     Superclass;
    typedef SmartPointer<Self>         Pointer;
    typedef SmartPointer<const Self>   ConstPointer;
 
    itkTypeMacro( RANSAC, Object );
    /** New method for creating an object using a factory. */
    itkNewMacro( Self )
    
    /**
     * Set/Get the number of threads used by the RANSAC implementation. 
     *
     * @param numberOfThreads Number of threads the algorithm uses. Valid values
     *                        are in [1, #cores].
     */
    void SetNumberOfThreads( unsigned int numberOfThreads );
    unsigned int GetNumberOfThreads();

    /**
     * Set the function object that is able to estimate the desired parametric 
     * entity (e.g. PlaneParametersEstimator).
     * @param paramEstimator An object which can estimate the desired parameters 
     *                       using both an exact fit and a least squares fit.
     */ 
    void SetParametersEstimator( typename ParametersEstimator<T,S>::Pointer paramEstimator );

    /**
     * Set the data objects we want to use (e.g. point pairs for estimating a 
     * homography.
     * @param data The input from which the parameters will be estimated.
     */
    void SetData( std::vector<T> &data );

    /**
     * Estimate the model parameters using the RANSAC framework.
     * @param parameters A vector which will contain the estimated parameters.
     *                   If there is an error then this vector will be empty.
     *                   Errors are: 1. Parameter estimation object or data not
     *                                  set, see SetParameterEstimator and 
     *                                  SetData.
     *                               2. The given data is in a singular 
     *                                  configuration (e.g. trying to fit a circle
     *                                  to a set of colinear points).
     *                               3. The given parameter desiredProbabilityForNoOutliers 
     *                                  is not in (0,1)
     * @param desiredProbabilityForNoOutliers The probability that at least one of 
     *                                        the selected subsets doesn't contain 
     *                                        an outlier, must be in (0,1).
     * @return Returns the percentage of data used in the least squares estimate.
     */
    double Compute( std::vector<S> &parameters, 		             
	double desiredProbabilityForNoOutliers );

protected:

    /**
     * Construct an instance of the RANSAC algorithm. The number of threads used 
     * in the computation is 1, valid values are in [1, #cores].
     * 
     */
    RANSAC();
    ~RANSAC();

private:

    /**
     * Compute n choose m  [ n!/(m!*(n-m)!)]. 
     * If choose(n,m)>std::numeric_limits<unsigned int>::max(), or there is an
     * overflow during the computations then we return 
     * std::numeric_limits<unsigned int>::max(), otherwise the correct value
     * is returned.
     */
    unsigned int Choose( unsigned int n, unsigned int m );

    class SubSetIndexComparator {
    private:
	int length;
    public:
	SubSetIndexComparator(int arrayLength) : length(arrayLength){}
	bool operator()(const int *arr1, const int *arr2) const {
	    for(int i=0; i<this->length; i++) {
		if(arr1[i] < arr2[i])
		    return true;
		else if(arr1[i] > arr2[i]) 
		    return false;
	    }
	    return false;			
	}
    };
                   
    static 
#if ITK_VERSION_MAJOR >= 5
    ITK_THREAD_RETURN_FUNCTION_CALL_CONVENTION
#else
    ITK_THREAD_RETURN_TYPE
#endif
        RANSACThreadCallback( void *arg );

    //number of threads used in computing the RANSAC hypotheses
    unsigned int numberOfThreads;

    //the following variables are shared by all threads used in the RANSAC
    //computation

    //array corresponding to length of data array, data[i]== true if it 
    //agrees with the best model, otherwise false
    bool *bestVotes;
    unsigned int numVotesForBest;

    std::vector<T> data;

    //set which holds all of the subgroups/hypotheses already selected
    std::set<int *, SubSetIndexComparator > *chosenSubSets;
    //number of iterations, equivalent to desired number of hypotheses
    unsigned int numTries;

    double numerator;
    unsigned int allTries;

    /* GCS - best hypothesis during RANSAC, so we can skip the final 
       lsq estimate */
    std::vector<S> best_estimate;

    typename ParametersEstimator<T,S>::Pointer paramEstimator;
#if ITK_VERSION_MAJOR >= 5
    std::mutex hypothesisMutex;
    std::mutex resultsMutex;
#else
    itk::SimpleFastMutexLock hypothesisMutex;
    itk::SimpleFastMutexLock resultsMutex;
#endif
};

} // end namespace itk

#include "RANSAC.txx"

#endif //_RANSAC_H_

