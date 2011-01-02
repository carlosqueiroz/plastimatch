// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the kernel ridge regression 
    object from the dlib C++ Library.

    This example will train on data from the sinc function.

*/

#include <iostream>
#include <map>
#include <vector>

#include "dlib/svm.h"
#include "dlib/data_io.h"

using namespace std;
using namespace dlib;

int main()
{
    typedef std::map<unsigned long, double> sparse_sample_type;
    std::vector<sparse_sample_type> sparse_samples;
    std::vector<double> labels;

    load_libsvm_formatted_data (
        "/home/gsharp/Dropbox/autolabel/t-spine/t-spine.libsvm.all.scale", 
	sparse_samples, 
	labels
    );

#if defined (commentout)
    std::vector<sample_type>::iterator it;
    for (it = samples.begin(); 
	 it != samples.end();
	 it++)
    {
	sample_type& s = *it;
	sample_type::iterator map_it;
	for (map_it = s.begin();
	     map_it != s.end();
	     map_it++)
	{
	    unsigned long t = map_it->first;
	    std::cout << t << " (" << map_it->second << ") ";
	}
	std::cout << std::endl;
    }

    std::vector<double>::iterator labels_it;
    for (labels_it = labels.begin(); 
	 labels_it != labels.end();
	 labels_it++)
    {
	double t = (*labels_it);
	std::cout << t << std::endl;
    }
#endif

    typedef matrix<sparse_sample_type::value_type::second_type,0,1> 
	dense_sample_type;
    std::vector<dense_sample_type> dense_samples;
    dense_samples = sparse_to_dense (sparse_samples);

    typedef radial_basis_kernel<dense_sample_type> kernel_type;
    krr_trainer<kernel_type> trainer;

    // Here we set the kernel we want to use for training.   The radial_basis_kernel 
    // has a parameter called gamma that we need to determine.  As a rule of thumb, a good 
    // gamma to try is 1.0/(mean squared distance between your sample points).  So 
    // below we are using a similar value computed from at most 2000 randomly selected
    // samples.
    const double gamma = 3.0 / compute_mean_squared_distance(randomly_subsample(dense_samples, 2000));
    cout << "using gamma of " << gamma << endl;
    trainer.set_kernel(kernel_type(gamma));

#if defined (commentout)
    // now train a function based on our sample points
    decision_function<kernel_type> test = trainer.train(dense_samples, labels);
#endif

    // Note that the krr_trainer has the ability to tell us the leave-one-out cross-validation
    // accuracy.  The train() function has an optional 3rd argument and if we give it a double
    // it will give us back the LOO error.
    double loo_error;
    trainer.train(dense_samples, labels, loo_error);
    cout << "mean squared LOO error: " << loo_error << endl;
    // Which outputs the following:
    // mean squared LOO error: 8.29563e-07
}
