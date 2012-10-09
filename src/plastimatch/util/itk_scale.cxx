/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include "itkImage.h"
#include "itkMultiplyByConstantImageFilter.h"

#include "itk_image_type.h"
#include "itk_scale.h"

template <class T>
T
itk_scale (
    const T& image, 
    float weight)
{
    typedef typename T::ObjectType ImageType;
    typedef typename T::ObjectType::PixelType PixelType;

    typedef typename itk::MultiplyByConstantImageFilter< 
        ImageType, float, ImageType > MulFilterType;
    typename MulFilterType::Pointer multiply = MulFilterType::New();
    multiply->SetConstant (weight);
    multiply->SetInput (image);
    try {
        multiply->Update();
    }
    catch(itk::ExceptionObject & ex) {
	printf ("Exception running itkExtractImageFilter.\n");
	std::cout << ex << std::endl;
	getchar();
	exit(1);
    }
    return multiply->GetOutput();
}


/* Explicit instantiations */
template PLMUTIL_API UCharImageType::Pointer itk_scale (const UCharImageType::Pointer&, float);
template PLMUTIL_API ShortImageType::Pointer itk_scale (const ShortImageType::Pointer&, float);
template PLMUTIL_API UShortImageType::Pointer itk_scale (const UShortImageType::Pointer&, float);
template PLMUTIL_API UInt32ImageType::Pointer itk_scale (const UInt32ImageType::Pointer&, float);
template PLMUTIL_API FloatImageType::Pointer itk_scale (const FloatImageType::Pointer&, float);
template PLMUTIL_API DeformationFieldType::Pointer itk_scale (const DeformationFieldType::Pointer&, float);
