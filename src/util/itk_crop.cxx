/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include "itkContinuousIndex.h"
#include "itkRegionOfInterestImageFilter.h"
#include "itkImage.h"

#include "clamp.h"
#include "itk_bbox.h"
#include "itk_crop.h"
#include "itk_image_type.h"

template <class T>
T
itk_crop_by_index (
    T& image, 
    const int new_size[6])
{
    typedef typename T::ObjectType ImageType;
    typedef typename T::ObjectType::PixelType PixelType;
    typedef itk::RegionOfInterestImageFilter < ImageType, ImageType > FilterType;

    typename FilterType::Pointer filter = FilterType::New();
    typename ImageType::IndexType  extract_index;
    typename ImageType::SizeType   extract_size;
    typename ImageType::RegionType extract_region;

    // GCS FIX: Should use itk index
    typename ImageType::RegionType current_region
        = image->GetLargestPossibleRegion();

    for (int d = 0; d < 3; d++) {
	extract_index[d] = new_size[d*2];
        if (extract_index[d] < 0) {
            extract_index[d] = 0;
        }
	extract_size[d] = new_size[d*2+1] - extract_index[d] + 1;
        if (extract_size[d] > current_region.GetSize(d)) {
            extract_size[d] = current_region.GetSize(d);
        }
    }

    extract_region.SetSize (extract_size);
    extract_region.SetIndex (extract_index);

    filter->SetInput (image);
    filter->SetRegionOfInterest (extract_region);

    try {
	filter->UpdateLargestPossibleRegion ();
    }
    catch(itk::ExceptionObject & ex) {
	printf ("Exception running itkRegionOfInterestImageFilter.\n");
	std::cout << ex << std::endl;
	exit(1);
    }

    T out_image = filter->GetOutput();
    return out_image;
}

template <class T>
T
itk_crop_by_coord (
    T& image, 
    const float new_size[6])
{
    typedef typename T::ObjectType ImageType;
    typedef typename T::ObjectType::PixelType PixelType;
    typedef itk::RegionOfInterestImageFilter < ImageType, ImageType > FilterType;

    typename FilterType::Pointer filter = FilterType::New();
    typename ImageType::IndexType extract_index;
    typename ImageType::SizeType extract_size;
    typename ImageType::RegionType extract_region;

    // GCS FIX: Should use itk index
    typename ImageType::RegionType current_region
        = image->GetLargestPossibleRegion();
    
    itk::Point<double,3> p1, p2;
    itk::ContinuousIndex<double,3> i1, i2;
    p1[0] = new_size[0];
    p2[0] = new_size[1];
    p1[1] = new_size[2];
    p2[1] = new_size[3];
    p1[2] = new_size[4];
    p2[2] = new_size[5];
    //image->TransformPhysicalPointToIndex (p1, i1);
    //image->TransformPhysicalPointToIndex (p2, i2);
    image->TransformPhysicalPointToContinuousIndex (p1, i1);
    image->TransformPhysicalPointToContinuousIndex (p2, i2);

    CLAMP2 (i1[0], i2[0], 0, current_region.GetSize(0)-1);
    CLAMP2 (i1[1], i2[1], 0, current_region.GetSize(1)-1);
    CLAMP2 (i1[2], i2[2], 0, current_region.GetSize(2)-1);
    
    for (int d = 0; d < 3; d++) {
	extract_index[d] = std::ceil(i1[d]);
	extract_size[d] = std::floor(i2[d]) - std::ceil(i1[d]) + 1;
    }

    extract_region.SetSize (extract_size);
    extract_region.SetIndex (extract_index);

    filter->SetInput (image);
    filter->SetRegionOfInterest (extract_region);

    try {
	filter->UpdateLargestPossibleRegion ();
    }
    catch(itk::ExceptionObject & ex) {
	printf ("Exception running itkRegionOfInterestImageFilter.\n");
	std::cout << ex << std::endl;
	printf ("new_size = %f %f %f %f %f %f\n",
            new_size[0], new_size[1], new_size[2], 
            new_size[3], new_size[4], new_size[5]);
	std::cout << image->GetOrigin() << "\n";
	std::cout << image->GetSpacing() << "\n";
	std::cout << current_region << std::endl;
	std::cout << extract_region << std::endl;
	exit(1);
    }

    T out_image = filter->GetOutput();
    return out_image;
}


template <class T>
PLMUTIL_API
T
itk_crop_by_image (T& image, const UCharImageType::Pointer& bbox_image)
{
    int bbox_indices[6];
    float bbox_coordinates[6];
    itk_bbox (bbox_image, bbox_coordinates, bbox_indices);
    return itk_crop_by_index (image, bbox_indices);
}

/* Explicit instantiations */
template PLMUTIL_API CharImageType::Pointer itk_crop_by_index (CharImageType::Pointer&, const int*);
template PLMUTIL_API UCharImageType::Pointer itk_crop_by_index (UCharImageType::Pointer&, const int*);
template PLMUTIL_API ShortImageType::Pointer itk_crop_by_index (ShortImageType::Pointer&, const int*);
template PLMUTIL_API UShortImageType::Pointer itk_crop_by_index (UShortImageType::Pointer&, const int*);
template PLMUTIL_API Int32ImageType::Pointer itk_crop_by_index (Int32ImageType::Pointer&, const int*);
template PLMUTIL_API UInt32ImageType::Pointer itk_crop_by_index (UInt32ImageType::Pointer&, const int*);
template PLMUTIL_API FloatImageType::Pointer itk_crop_by_index (FloatImageType::Pointer&, const int*);
template PLMUTIL_API DoubleImageType::Pointer itk_crop_by_index (DoubleImageType::Pointer&, const int*);
template PLMUTIL_API UCharVecImageType::Pointer itk_crop_by_index (UCharVecImageType::Pointer&, const int*);

template PLMUTIL_API CharImageType::Pointer itk_crop_by_coord (CharImageType::Pointer&, const float*);
template PLMUTIL_API UCharImageType::Pointer itk_crop_by_coord (UCharImageType::Pointer&, const float*);
template PLMUTIL_API ShortImageType::Pointer itk_crop_by_coord (ShortImageType::Pointer&, const float*);
template PLMUTIL_API UShortImageType::Pointer itk_crop_by_coord (UShortImageType::Pointer&, const float*);
template PLMUTIL_API Int32ImageType::Pointer itk_crop_by_coord (Int32ImageType::Pointer&, const float*);
template PLMUTIL_API UInt32ImageType::Pointer itk_crop_by_coord (UInt32ImageType::Pointer&, const float*);
template PLMUTIL_API FloatImageType::Pointer itk_crop_by_coord (FloatImageType::Pointer&, const float*);
template PLMUTIL_API DoubleImageType::Pointer itk_crop_by_coord (DoubleImageType::Pointer&, const float*);
template PLMUTIL_API UCharVecImageType::Pointer itk_crop_by_coord (UCharVecImageType::Pointer&, const float*);

template PLMUTIL_API CharImageType::Pointer itk_crop_by_image (CharImageType::Pointer&, const UCharImageType::Pointer&);
template PLMUTIL_API UCharImageType::Pointer itk_crop_by_image (UCharImageType::Pointer&, const UCharImageType::Pointer&);
template PLMUTIL_API ShortImageType::Pointer itk_crop_by_image (ShortImageType::Pointer&, const UCharImageType::Pointer&);
template PLMUTIL_API UShortImageType::Pointer itk_crop_by_image (UShortImageType::Pointer&, const UCharImageType::Pointer&);
template PLMUTIL_API Int32ImageType::Pointer itk_crop_by_image (Int32ImageType::Pointer&, const UCharImageType::Pointer&);
template PLMUTIL_API UInt32ImageType::Pointer itk_crop_by_image (UInt32ImageType::Pointer&, const UCharImageType::Pointer&);
template PLMUTIL_API FloatImageType::Pointer itk_crop_by_image (FloatImageType::Pointer&, const UCharImageType::Pointer&);
template PLMUTIL_API DoubleImageType::Pointer itk_crop_by_image (DoubleImageType::Pointer&, const UCharImageType::Pointer&);
template PLMUTIL_API UCharVecImageType::Pointer itk_crop_by_image (UCharVecImageType::Pointer&, const UCharImageType::Pointer&);
