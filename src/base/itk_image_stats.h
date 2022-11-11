/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_image_stats_h_
#define _itk_image_stats_h_

#include "plmbase_config.h"
#include "itk_image.h"
#include "itk_mask.h"

enum Stats_operation {
    STATS_OPERATION_INSIDE,
    STATS_OPERATION_OUTSIDE
};

class Image_stats;

/* -----------------------------------------------------------------------
   Function prototypes
   ----------------------------------------------------------------------- */
template<class T> PLMBASE_API void itk_image_stats (
    T img,
    double *min_val,
    double *max_val, 
    double *avg,
    int *non_zero,
    int *num_vox
);

template<class T> PLMBASE_API void itk_image_stats (
    T img,
    double *min_val,
    double *max_val,
    double *avg,
    int *non_zero,
    int *num_vox,
    double *sigma
);


template<class T> PLMBASE_API void itk_masked_image_stats (
    T img,
    UCharImageType::Pointer mask,
    Stats_operation stats_operation,
    double *min_val,
    double *max_val,
    double *avg,
    int *non_zero,
    int *num_vox,
    double *sigma
);

template<class T> PLMBASE_API void itk_masked_image_stats (
    T img,
    UCharImageType::Pointer mask,
    Stats_operation stats_operation,
    double *min_val,
    double *max_val,
    double *avg,
    int *non_zero,
    int *num_vox
);
template<class T> PLMBASE_API void itk_image_stats (
    const T& img,
    Image_stats *
);

#endif
