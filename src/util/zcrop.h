/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _zcrop_h_
#define _zcrop_h_

#include "plmutil_config.h"
#include "plm_image.h"
#include "itk_image_type.h"  /* Not sure if needed or not  */

class Plm_image; /* Not sure if needed or not  */

PLMUTIL_API
void
zcrop (UCharImageType::Pointer& ref,
    UCharImageType::Pointer& cmp,
    float zcrop[2]);

#endif
