/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _drr_options_h_
#define _drr_options_h_

#include "plmreconstruct_config.h"
#include <string>
#include "plm_math.h"
#include "threading.h"

class Drr_options;
class Proj_image;
class Volume;

#define OUTPUT_FORMAT_PFM                 0
#define OUTPUT_FORMAT_PGM                 1
#define OUTPUT_FORMAT_RAW                 2

//#define DRR_VERBOSE 1
//#define DRR_DEBUG_CALLBACK 1
//#define DRR_ULTRA_VERBOSE 1

enum drr_algorithm {
    DRR_ALGORITHM_EXACT,
    DRR_ALGORITHM_TRILINEAR_EXACT,
    DRR_ALGORITHM_TRILINEAR_APPROX,
    DRR_ALGORITHM_UNIFORM
};
typedef enum drr_algorithm Drr_algorithm;

enum hu_conversion_type {
    PREPROCESS_CONVERSION,
    INLINE_CONVERSION,
    NO_CONVERSION
};
typedef enum hu_conversion_type Hu_conversion;

class PLMRECONSTRUCT_API Drr_options {
public:
    Drr_options ();
public:
    Threading threading;
    int detector_resolution[2];      /* In pixels */
    float image_size[2];             /* In mm */
    int have_image_center;           /* Was image_center spec'd in options? */
    int image_resolution[2];         /* In pixels */
    float image_center[2];           /* In pixels */
    int have_image_window;           /* Was image_window spec'd in options? */
    int image_window[4];             /* In pixels */
    float isocenter[3];              /* In mm */

    float start_angle;               /* Source gantry angle */
    int num_angles;
    int have_angle_diff;             /* Was angle_diff spec'd in options? */
    float angle_diff;                /* In degrees */

    int have_nrm;                    /* Was nrm specified? */
    float nrm[3];                    /* Normal vector (unitless) */
    float vup[3];                    /* Direction vector (unitless) */

    float sad;			     /* In mm */
    float sid;			     /* In mm */
    int exponential_mapping;
    int output_format;
    Hu_conversion hu_conversion;
    Drr_algorithm algorithm;
    int geometry_only;
    std::string input_file;
    std::string output_file;
    std::string output_prefix;

    bool autoscale;
    float autoscale_range[2];
    float manual_scale;

    /* The option specified by the user goes in output_details_prefix, 
       and the individual filename for a specific angle goes in 
       output_details_fn */
    std::string output_details_prefix;
    std::string output_details_fn;
};

#endif
