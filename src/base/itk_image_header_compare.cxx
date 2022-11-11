/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkCastImageFilter.h"
#include "itkOrientImageFilter.h"

#include "direction_cosines.h"
#include "itk_image.h"
#include "itk_volume_header.h"
#include "plm_image_header.h"
#include "plm_int.h"
#include "volume_header.h"

/* -----------------------------------------------------------------------
    Functions
   ----------------------------------------------------------------------- */

/* Return true if the headers are the same */
template<class T, class U>
bool 
itk_image_header_compare (T image1, U image2)
{
    typedef typename U::ObjectType I1ImageType;
    typedef typename T::ObjectType I2ImageType;

    const SizeType& i1_sz = image1->GetLargestPossibleRegion().GetSize ();
    const OriginType i1_og = itk_image_origin (image1);
    const typename I1ImageType::SpacingType& i1_sp = image1->GetSpacing();
    const typename I1ImageType::DirectionType& i1_dc = image1->GetDirection();

    const SizeType& i2_sz = image2->GetLargestPossibleRegion().GetSize ();
    const OriginType i2_og = itk_image_origin (image2);
    const typename I2ImageType::SpacingType& i2_sp = image2->GetSpacing();
    const typename I2ImageType::DirectionType& i2_dc = image2->GetDirection();

    if (i1_sz != i2_sz || i1_og != i2_og || i1_sp != i2_sp || i1_dc != i2_dc)
    {
        return false;
    } else {
        return true;
    }
}

/* -----------------------------------------------------------------------
   Explicit instantiations
   ----------------------------------------------------------------------- */
template PLMBASE_API bool itk_image_header_compare (UCharImageType::Pointer image1, UCharImageType::Pointer image2);
template PLMBASE_API bool itk_image_header_compare (UShortImageType::Pointer image1, UCharImageType::Pointer image2);
template PLMBASE_API bool itk_image_header_compare (ShortImageType::Pointer image1, UCharImageType::Pointer image2);
template PLMBASE_API bool itk_image_header_compare (UInt32ImageType::Pointer image1, UCharImageType::Pointer image2);
template PLMBASE_API bool itk_image_header_compare (Int32ImageType::Pointer image1, UCharImageType::Pointer image2);
template PLMBASE_API bool itk_image_header_compare (FloatImageType::Pointer image1, UCharImageType::Pointer image2);
template PLMBASE_API bool itk_image_header_compare (FloatImageType::Pointer image1, FloatImageType::Pointer image2);
template PLMBASE_API bool itk_image_header_compare (DeformationFieldType::Pointer image1, DeformationFieldType::Pointer image2);
