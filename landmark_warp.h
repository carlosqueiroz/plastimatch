/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _landmark_warp_h_
#define _landmark_warp_h_

#include "plm_config.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "pointset.h"

class plastimatch1_EXPORT Landmark_warp
{
public:
    /* Inputs */
    Pointset *m_fixed_landmarks;
    Pointset *m_moving_landmarks;
    Plm_image *m_input_img;
    Plm_image_header m_pih;

    /* Config */
    float default_val;
    float rbf_radius;
    float young_modulus;
    int num_clusters; // if >0, use adaptive radius of RBF

    /* Internals */
    int *cluster_id;  // index of cluster the landmark belongs to
    float *adapt_radius; // adaptively found radius of RBF of each landmark

    /* Outputs */
    Plm_image *m_warped_img;
    Xform *m_vf;
    Pointset *m_warped_landmarks; // if regularized, warped l. may not exactly match fixed

public:
    Landmark_warp ();
    ~Landmark_warp ();
    void
    load_pointsets (
	const char *fixed_lm_fn, 
	const char *moving_lm_fn
    );
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT Landmark_warp*
landmark_warp_create (void);
gpuit_EXPORT void
landmark_warp_destroy (Landmark_warp *lw);
gpuit_EXPORT 
Landmark_warp*
landmark_warp_load_xform (const char *fn);
gpuit_EXPORT 
Landmark_warp*
landmark_warp_load_pointsets (const char *fixed_lm_fn, const char *moving_lm_fn);

#if defined __cplusplus
}
#endif

#endif
