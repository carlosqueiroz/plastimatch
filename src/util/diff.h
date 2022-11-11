/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _diff_h_
#define _diff_h_

#include "plmutil_config.h"
#include <string>
#include "plm_image.h"

PLMUTIL_API Plm_image::Pointer
diff_image (const Plm_image::Pointer& pi1, const Plm_image::Pointer& pi2);

PLMUTIL_API DeformationFieldType::Pointer
diff_vf (const DeformationFieldType::Pointer& vf1,
    const DeformationFieldType::Pointer& vf2);
    

#endif
