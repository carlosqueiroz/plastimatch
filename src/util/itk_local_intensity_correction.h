/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_local_intensity_correction_h_
#define _itk_local_intensity_correction_h_

#include "plmutil_config.h"
#include <utility>
#include "itk_image_type.h"


PLMUTIL_API FloatImageType::Pointer
itk_local_intensity_correction(
        FloatImageType::Pointer& source_image,
        FloatImageType::Pointer& reference_image,
        SizeType patch_size,
        bool blend,
        SizeType mediansize);

PLMUTIL_API FloatImageType::Pointer
itk_local_intensity_correction (
        FloatImageType::Pointer& source_image,
        FloatImageType::Pointer& reference_image, SizeType patch_size,
        FloatImageType::Pointer& shift_field, FloatImageType::Pointer& scale_field,
        bool blend,
        SizeType mediansize);

PLMUTIL_API FloatImageType::Pointer
itk_masked_local_intensity_correction(
        FloatImageType::Pointer& source_image,
        FloatImageType::Pointer& reference_image, SizeType patch_size,
        UCharImageType::Pointer& source_mask, UCharImageType::Pointer& reference_mask,
        FloatImageType::Pointer& shift_field, FloatImageType::Pointer& scale_field,
        bool blend,
        SizeType mediansize);
#endif
