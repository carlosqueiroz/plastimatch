/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <time.h>
#include "itkSubtractImageFilter.h"
#include "itkImageRegionIterator.h"

#include "diff.h"
#include "itk_image_save.h"
#include "plm_image.h"
#include "print_and_exit.h"

DeformationFieldType::Pointer
diff_vf (
    const DeformationFieldType::Pointer& vf1,
    const DeformationFieldType::Pointer& vf2
) {
    
    typedef itk::SubtractImageFilter< DeformationFieldType, DeformationFieldType, 
				      DeformationFieldType > SubtractFilterType;
    SubtractFilterType::Pointer sub_filter = SubtractFilterType::New();

    sub_filter->SetInput1 (vf1);
    sub_filter->SetInput2 (vf2);

    try {
	sub_filter->Update();
    } catch (itk::ExceptionObject & excep) {
	std::cerr << "ITK exception caught: " << excep << std::endl;
	exit (-1);
    }
    DeformationFieldType::Pointer vf_diff = sub_filter->GetOutput ();
    return vf_diff;
}

Plm_image::Pointer
diff_image (
    const Plm_image::Pointer& pi1,
    const Plm_image::Pointer& pi2
) {
    
    FloatImageType::Pointer fi1 = pi1->itk_float ();
    FloatImageType::Pointer fi2 = pi2->itk_float ();

    typedef itk::SubtractImageFilter< FloatImageType, FloatImageType, 
				      FloatImageType > SubtractFilterType;
    SubtractFilterType::Pointer sub_filter = SubtractFilterType::New();

    sub_filter->SetInput1 (fi1);
    sub_filter->SetInput2 (fi2);

    try {
	sub_filter->Update();
    } catch (itk::ExceptionObject & excep) {
	std::cerr << "ITK exception caught: " << excep << std::endl;
	exit (-1);
    }
    FloatImageType::Pointer diff = sub_filter->GetOutput ();
    Plm_image::Pointer pi_diff = Plm_image::New (diff);
    return pi_diff;
}
