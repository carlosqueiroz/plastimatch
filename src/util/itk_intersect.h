/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_intersect_h_
#define _itk_intersect_h_

#include "plmutil_config.h"
#include "itk_image_type.h"

PLMUTIL_API
UCharImageType::Pointer
itk_intersect (const UCharImageType::Pointer image_1,
    const UCharImageType::Pointer image_2);

#endif
