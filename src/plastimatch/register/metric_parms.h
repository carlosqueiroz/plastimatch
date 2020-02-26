/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _metric_parms_h_
#define _metric_parms_h_

#include "plmregister_config.h"
#include <string>
#include <vector>
#include "plm_return_code.h"
#include "similarity_metric_type.h"

enum Bspline_lut_strategy {
    BSPLINE_LUT_STRATEGY_NONE,
    BSPLINE_LUT_STRATEGY_AUTO,
    BSPLINE_LUT_STRATEGY_3D,
    BSPLINE_LUT_STRATEGY_1D
};

class PLMREGISTER_API Metric_parms {
public:
    Metric_parms ();
public:
    Similarity_metric_type metric_type;
    float metric_lambda;
    Bspline_lut_strategy bspline_lut_strategy;

    std::string fixed_fn;
    std::string moving_fn;
    std::string fixed_roi_fn;
    std::string moving_roi_fn;
public:
    Plm_return_code set_metric_type (const std::string& val);
};

#endif
