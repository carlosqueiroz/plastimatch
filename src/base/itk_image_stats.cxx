/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include "itkImage.h"
#include "itkOrientImageFilter.h"

#include "image_stats.h"
#include "itk_image_type.h"
#include "itk_image_stats.h"
#include "itk_image_header_compare.h"
#include "itk_resample.h"

/* -----------------------------------------------------------------------
   Statistics like min, max, etc.
   ----------------------------------------------------------------------- */
template<class T> 
void
itk_image_stats (const T& img, Image_stats *image_stats)
{
    int num_non_zero, num_vox;
    itk_image_stats(img, &(image_stats->min_val),
        &(image_stats->max_val), &(image_stats->avg_val),
        &num_non_zero, &num_vox);
    // manually assign incompatible types
    image_stats->num_non_zero = num_non_zero;
    image_stats->num_vox = num_vox;
}

template<class T> 
void
itk_image_stats (T img, double *min_val, double *max_val, 
    double *avg, int *non_zero, int *num_vox)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageRegionIterator< ImageType > IteratorType;
    typename ImageType::RegionType rg = img->GetLargestPossibleRegion ();
    IteratorType it (img, rg);

    int first = 1;
    double sum = 0.0;

    *non_zero = 0;
    *num_vox = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
	double v = (double) it.Get();
	if (first) {
	    *min_val = *max_val = v;
	    first = 0;
	}
	if (*min_val > v) *min_val = v;
	if (*max_val < v) *max_val = v;
	sum += v;
	(*num_vox) ++;
	if (v != 0.0) {
	    (*non_zero) ++;
	}
    }
    *avg = sum / (*num_vox);
}

template<class T>
void
itk_image_stats(T img, double *min_val, double *max_val,
    double *avg, int *non_zero, int *num_vox, double *sigma)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageRegionIterator< ImageType > IteratorType;
    typename ImageType::RegionType rg = img->GetLargestPossibleRegion ();
    IteratorType it (img, rg);

    itk_image_stats(img, min_val, max_val, avg, non_zero, num_vox);
    *sigma = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        double v = (double) it.Get();
        *sigma += (v - *avg) * (v - *avg);
    }
    *sigma = double(sqrtl(*sigma / *num_vox));
}

template<class T>
void itk_masked_image_stats(T img, UCharImageType::Pointer mask,
    Stats_operation stats_operation, double* min_val, double* max_val,
    double *avg, int *non_zero, int *num_vox)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageRegionIterator<ImageType> ImageIteratorType;
    typedef itk::ImageRegionIterator<UCharImageType> MaskIteratorType;

    if (!itk_image_header_compare(img, mask)) {
        mask = itk_resample_image(mask, img, 0, 0);
    }

    typename ImageType::RegionType rgn_img
        = img->GetLargestPossibleRegion();
    typename UCharImageType::RegionType rgn_mask
        = mask->GetLargestPossibleRegion();
    ImageIteratorType it_img(img, rgn_img);
    MaskIteratorType it_mask(mask, rgn_mask);

    int first = 1;
    double sum = 0.0;
    *non_zero = 0;
    *num_vox = 0;
    for (it_img.GoToBegin(), it_mask.GoToBegin(); !it_img.IsAtEnd();
         ++it_img, ++it_mask)
    {
        double v = (double) it_img.Get();
        unsigned char mask_value = it_mask.Get();
        if ((mask_value > 0) ^ (stats_operation == STATS_OPERATION_OUTSIDE)) {
            if (first) {
                *min_val = *max_val = v;
                first = 0;
            }
            if (*min_val > v) *min_val = v;
            if (*max_val < v) *max_val = v;
            sum += v;
            (*num_vox)++;
            if (v != 0.0) {
                (*non_zero)++;
            }
        }
    }
    *avg = sum / (*num_vox);
}

template<class T>
void itk_masked_image_stats(T img, UCharImageType::Pointer mask,
    Stats_operation stats_operation, double* min_val, double* max_val,
    double *avg, int *non_zero, int *num_vox, double *sigma)
{
    typedef typename T::ObjectType ImageType;
    typedef itk::ImageRegionIterator<ImageType> ImageIteratorType;
    typedef itk::ImageRegionIterator<UCharImageType> MaskIteratorType;

    if (!itk_image_header_compare (img, mask)) {
        mask = itk_resample_image (mask, img, 0, 0);
    }

    typename ImageType::RegionType rgn_img
        = img->GetLargestPossibleRegion();
    typename UCharImageType::RegionType rgn_mask
        = mask->GetLargestPossibleRegion();
    ImageIteratorType it_img (img, rgn_img);
    MaskIteratorType it_mask (mask, rgn_mask);
    itk_masked_image_stats(img, mask, stats_operation, min_val, max_val, avg, non_zero, num_vox);
    *sigma = 0;
    for (it_img.GoToBegin(), it_mask.GoToBegin(); !it_img.IsAtEnd(); ++it_img, ++it_mask) {
        double v = (double) it_img.Get();
        unsigned char mask_value = it_mask.Get();

        if ((mask_value > 0) ^ (stats_operation == STATS_OPERATION_OUTSIDE)) {
            *sigma += (v - *avg) * (v - *avg);
        }
    }
    *sigma = double(sqrtl(*sigma / *num_vox));
}
/* Explicit instantiations */
template PLMBASE_API void itk_image_stats (UCharImageType::Pointer, double*, double*, double*, int*, int*);
template PLMBASE_API void itk_image_stats (ShortImageType::Pointer, double*, double*, double*, int*, int*);
template PLMBASE_API void itk_image_stats (Int32ImageType::Pointer, double*, double*, double*, int*, int*);
template PLMBASE_API void itk_image_stats (FloatImageType::Pointer, double*, double*, double*, int*, int*);


template PLMBASE_API void itk_image_stats (UCharImageType::Pointer, double*, double*, double*, int*, int*, double*);
template PLMBASE_API void itk_image_stats (ShortImageType::Pointer, double*, double*, double*, int*, int*, double*);
template PLMBASE_API void itk_image_stats (Int32ImageType::Pointer, double*, double*, double*, int*, int*, double*);
template PLMBASE_API void itk_image_stats (FloatImageType::Pointer, double*, double*, double*, int*, int*, double*);

template PLMBASE_API void itk_image_stats (const UCharImageType::Pointer&, Image_stats*);
template PLMBASE_API void itk_image_stats (const ShortImageType::Pointer&, Image_stats*);
template PLMBASE_API void itk_image_stats (const UShortImageType::Pointer&, Image_stats*);
template PLMBASE_API void itk_image_stats (const Int32ImageType::Pointer&, Image_stats*);
template PLMBASE_API void itk_image_stats (const FloatImageType::Pointer&, Image_stats*);

template PLMBASE_API void itk_masked_image_stats (UCharImageType::Pointer, UCharImageType::Pointer,
		Stats_operation, double*, double*, double*, int*, int*, double*);
template PLMBASE_API void itk_masked_image_stats (ShortImageType::Pointer, UCharImageType::Pointer,
		Stats_operation, double*, double*, double*, int*, int*, double*);
template PLMBASE_API void itk_masked_image_stats (Int32ImageType::Pointer, UCharImageType::Pointer,
		Stats_operation, double*, double*, double*, int*, int*, double*);
template PLMBASE_API void itk_masked_image_stats (FloatImageType::Pointer, UCharImageType::Pointer,
		Stats_operation, double*, double*, double*, int*, int*, double*);

template PLMBASE_API void itk_masked_image_stats (UCharImageType::Pointer, UCharImageType::Pointer,
		Stats_operation, double*, double*, double*, int*, int*);
template PLMBASE_API void itk_masked_image_stats (ShortImageType::Pointer, UCharImageType::Pointer,
		Stats_operation, double*, double*, double*, int*, int*);
template PLMBASE_API void itk_masked_image_stats (Int32ImageType::Pointer, UCharImageType::Pointer,
		Stats_operation, double*, double*, double*, int*, int*);
template PLMBASE_API void itk_masked_image_stats (FloatImageType::Pointer, UCharImageType::Pointer,
		Stats_operation, double*, double*, double*, int*, int*);
