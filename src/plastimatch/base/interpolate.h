/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _interpolate_h_
#define _interpolate_h_

#include "plmbase_config.h"
#include "plm_int.h"

class Volume;

PLMBASE_C_API void li_clamp (
    float ma,
    plm_long dmax,
    plm_long* maf,
    plm_long* mar, 
    float* fa1, float* fa2
);

PLMBASE_C_API void li_clamp_3d (
    const float mijk[3],
    plm_long mijk_f[3],
    plm_long mijk_r[3],
    float li_frac_1[3],
    float li_frac_2[3],
    const Volume *moving
);

/* Compute only fractional components, do not solve for value.  
   Clamping is not done; instead, fractions are set to 0.f if xyz 
   lies outside limits of dim. */
PLMBASE_C_API void li_noclamp_3d (
    plm_long ijk_f[3],
    float li_frac_1[3],
    float li_frac_2[3],
    const float ijk[3],
    const plm_long dim[3]
);

PLMBASE_C_API void li_2d (
    plm_long *ijk_f,
    float *li_frac_1,
    float *li_frac_2,
    const float *ijk,
    const plm_long *dim
);

#if (COMMENTOUT)
PLMBASE_C_API float li_value (
    float f1[3],
    float f2[3],
    plm_long mvf, 
    float *m_img,
    Volume *moving
);

PLMBASE_C_API float li_value_dx (
    float f1[3],
    float f2[3],
    float rx,
    plm_long mvf, 
    float *m_img,
    Volume *moving
);

PLMBASE_C_API float li_value_dy (
    float f1[3],
    float f2[3],
    float ry,
    plm_long mvf, 
    float *m_img,
    Volume *moving
);

PLMBASE_C_API float li_value_dz (
    float f1[3],
    float f2[3],
    float rz,
    plm_long mvf, 
    float *m_img,
    Volume *moving
);
#endif (COMMENTOUT)
#endif
