/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _tps_h_
#define _tps_h_

#include "volume.h"

/* Incremental TPS only */
typedef struct tps_node Tps_node;
struct tps_node {
    float src[3];   /* (x,y,z) in fixed image */
    float tgt[3];   /* (x,y,z) in moving image */
    float dxyz[3];  /* tgt - src */
    float alpha;    /* RBF weight */
};

typedef struct tps_xform Tps_xform;
struct tps_xform {

    float img_origin[3];         /* Image origin (in mm) */
    float img_spacing[3];        /* Image spacing (in mm) */
    int img_dim[3];              /* Image size (in vox) */
    
    int num_tps_nodes;           /* Num control points */
    struct tps_node *tps_nodes;  /* Control point values */
};

#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT Tps_xform*
tps_xform_alloc (void);
gpuit_EXPORT Tps_xform*
tps_xform_load (char* fn);
gpuit_EXPORT void
tps_xform_save (Tps_xform *tps_xform, char *fn);
gpuit_EXPORT void
tps_xform_free (Tps_xform *tps_xform);

gpuit_EXPORT float
tps_default_alpha (float src[3], float tgt[3]);
gpuit_EXPORT void
tps_warp (
    Volume *vout,       /* Output image (sized and allocated) */
    Volume *vf_out,     /* Output vf (sized and allocated, can be null) */
    Tps_xform* tps,     /* TPS control points */
    Volume *moving,     /* Input image */
    int linear_interp,  /* 1 = trilinear, 0 = nearest neighbors */
    float default_val   /* Fill in this value outside of image */
);

#if defined __cplusplus
}
#endif

#endif
