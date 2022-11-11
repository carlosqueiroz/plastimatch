/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_bbox_h_
#define _itk_bbox_h_

#include "plmbase_config.h"
#include "itk_image_type.h"

PLMBASE_API void
itk_bbox (UCharImageType::Pointer img, float *bbox_coordinates,
    int *bbox_indices);

#endif
