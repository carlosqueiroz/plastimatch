/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include "itkImageRegionIterator.h"
#include "itkShiftScaleImageFilter.h"

#include "itk_image_shift_scale.h"

template<class T> 
void
itk_image_scale (T img, float scale)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageRegionIterator< ImageType > IteratorType;
    typename ImageType::RegionType rg = img->GetLargestPossibleRegion ();
    IteratorType it (img, rg);

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	double v = (double) it.Get();
        it.Set (v * scale);
    }
}

template<class T>
void
itk_image_shift_scale (T img, float shift, float scale)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageRegionIterator< ImageType > IteratorType;
    typename ImageType::RegionType rg = img->GetLargestPossibleRegion();
    IteratorType it (img, rg);
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        double v = (double) it.Get();
        it.Set(v * scale + shift);
    }
}


/* Explicit instantiations */
template PLMBASE_API void itk_image_scale (FloatImageType::Pointer, float);

template PLMBASE_API void itk_image_shift_scale (Int32ImageType::Pointer, float, float);
template PLMBASE_API void itk_image_shift_scale (FloatImageType::Pointer, float, float);
