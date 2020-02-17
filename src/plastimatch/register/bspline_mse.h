/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _bspline_mse_h_
#define _bspline_mse_h_

#include "plmregister_config.h"

class Bspline_optimize;

PLMREGISTER_API void bspline_score_c_mse (Bspline_optimize *bod);
PLMREGISTER_API void bspline_score_g_mse (Bspline_optimize *bod);
PLMREGISTER_API void bspline_score_h_mse (Bspline_optimize *bod);
PLMREGISTER_API void bspline_score_i_mse (Bspline_optimize *bod);
PLMREGISTER_API void bspline_score_k_mse (Bspline_optimize *bod);
PLMREGISTER_API void bspline_score_l_mse (Bspline_optimize *bod);
PLMREGISTER_API void bspline_score_m_mse (Bspline_optimize *bod);
PLMREGISTER_API void bspline_score_n_mse (Bspline_optimize *bod);
PLMREGISTER_API void bspline_score_o_mse (Bspline_optimize *bod);
PLMREGISTER_API void bspline_score_p_mse (Bspline_optimize *bod);
PLMREGISTER_API void bspline_score_q_mse (Bspline_optimize *bod);
PLMREGISTER_API void bspline_score_r_mse (Bspline_optimize *bod);

PLMREGISTER_API void
bspline_score_normalize (
    Bspline_optimize *bod,
    double raw_score
);

PLMREGISTER_API void bspline_score_mse (Bspline_optimize *bod);

PLMREGISTER_API float li_value (
    float f1[3],
    float f2[3],
    plm_long mvf, 
    float *m_img,
    Volume *moving
);

PLMREGISTER_API float li_value_dx (
    float f1[3],
    float f2[3],
    float rx,
    plm_long mvf, 
    float *m_img,
    Volume *moving
);

PLMREGISTER_API float li_value_dy (
    float f1[3],
    float f2[3],
    float ry,
    plm_long mvf, 
    float *m_img,
    Volume *moving
);

PLMREGISTER_API float li_value_dz (
    float f1[3],
    float f2[3],
    float rz,
    plm_long mvf, 
    float *m_img,
    Volume *moving
);

#endif
