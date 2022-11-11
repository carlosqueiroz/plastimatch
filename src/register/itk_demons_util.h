/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef ITK_DEMONS_UTIL_H
#define ITK_DEMONS_UTIL_H

#include "plmregister_config.h"
#include <itk_image_type.h>

class itk_demons_util
{
public:
    static void deformation_stats (DeformationFieldType::Pointer vf);
};

#endif
