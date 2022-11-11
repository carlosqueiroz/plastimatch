/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _registration_parms_h_
#define _registration_parms_h_

#include "plmregister_config.h"
#include <list>
#include <string>
#include <ctype.h>
#include <stdlib.h>
#include "bspline.h"    /* for enums */
#include "plm_image_type.h"
#include "plm_return_code.h"
#include "smart_pointer.h"
#include "threading.h"

class Groupwise_parms;
class Plm_image;
class Registration_parms_private;
class Shared_parms;
class Stage_parms;

#define DEFAULT_IMAGE_KEY "0"

class PLMREGISTER_API Registration_parms {
public:
    SMART_POINTER_SUPPORT (Registration_parms);
    Registration_parms_private *d_ptr;
public:
    int num_stages;

    /* Output files */
    std::string img_out_fn;
    std::string xf_in_fn;
    std::list<std::string> xf_out_fn;
    std::string vf_out_fn;
    std::string log_fn;
    float default_value;           /* Replacement when out-of-view */
    int init_type;
    double init[12];

    /* for 4D and atlas */
    std::string moving_dir;
    std::string fixed_dir;
    std::string img_out_dir;
    std::string vf_out_dir;

    /* for groupwise registration */
    std::string group_dir;

public:
    Registration_parms();
    ~Registration_parms();
public:
    Plm_return_code set_command_string (const std::string& command_string);
    Plm_return_code set_key_value (
        const std::string& section,
        const std::string& key, 
        const std::string& index, 
        const std::string& val);
    Plm_return_code parse_command_file (const char* options_fn);
    void set_job_paths (void);
public:
    Shared_parms* get_shared_parms ();
    void delete_all_stages ();
    std::list<Stage_parms*>& get_stages ();
    Stage_parms* append_stage ();
    Stage_parms* append_process_stage ();
    Groupwise_parms* get_groupwise_parms ();
};

#endif
