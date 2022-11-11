/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_crop_h_
#define _itk_crop_h_

#include "plmutil_config.h"
#include "itk_image.h"

template <class T>
PLMUTIL_API
T
itk_crop_by_index (T& image, const int *new_size);

template <class T>
PLMUTIL_API
T
itk_crop_by_coord (T& image, const float *new_size);

template <class T>
PLMUTIL_API
T
itk_crop_by_image (T& image, const UCharImageType::Pointer& bbox_image);

#endif
