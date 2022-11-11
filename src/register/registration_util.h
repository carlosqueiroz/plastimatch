/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _registration_util_h_
#define _registration_util_h_

#include "plmregister_config.h"
#include "itk_image_type.h"
#include "registration_data.h"
#include "stage_parms.h"

plm_long
count_fixed_voxels (Registration_data *regd,
    Stage_parms* stage, 
    FloatImageType::Pointer& fixed_ss);

#endif
