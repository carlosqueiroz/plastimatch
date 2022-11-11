/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _shared_parms_h_
#define _shared_parms_h_

#include "plmregister_config.h"
#include <map>
#include <string>
#include "metric_parms.h"
#include "plm_image_type.h"

#define IMG_OUT_FMT_AUTO                    0
#define IMG_OUT_FMT_DICOM                   1

class PLMREGISTER_API Shared_parms {
public:
    Shared_parms ();
    Shared_parms (const Shared_parms& s);
    ~Shared_parms ();

public:
    /* Similarity parms */
    std::map<std::string, Metric_parms> metric;
    
    /* ROI */
    bool fixed_roi_enable;
    bool moving_roi_enable;
    std::string valid_roi_out_fn;

    /* Stiffness map */
    bool fixed_stiffness_enable;
    std::string fixed_stiffness_fn;

    /* Subsampling */
    bool legacy_subsampling;

    /* Landmarks */
    std::string fixed_landmarks_fn;
    std::string moving_landmarks_fn;
    std::string fixed_landmarks_list;
    std::string moving_landmarks_list;
    std::string warped_landmarks_fn;

    /* Output files */
    bool img_out_resample_linear_xf;
    int img_out_fmt;
    Plm_image_type img_out_type;
    bool xf_out_itk;

public:
    void copy (const Shared_parms *s);
    void log ();
};

#endif
