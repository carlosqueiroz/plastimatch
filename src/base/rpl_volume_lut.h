/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rpl_volume_lut_h_
#define _rpl_volume_lut_h_

#include "plmbase_config.h"
#include "smart_pointer.h"

class Rpl_volume;
class Rpl_volume_lut_private;
class Volume;

class PLMBASE_API Rpl_volume_lut 
{
public:
    SMART_POINTER_SUPPORT (Rpl_volume_lut);
    Rpl_volume_lut_private *d_ptr;
public:
    Rpl_volume_lut (Rpl_volume *rv, Volume *vol);
    ~Rpl_volume_lut ();
private:
    /* Should not be called */
    Rpl_volume_lut ();
public:
    void build_lut ();
protected:
    void set_lut_entry (
        const Ray_data* ray_data, 
        plm_long vox_idx, 
        const float *vox_ray, 
        plm_long ap_idx, 
        float li_frac, 
        float step_length, 
        int lut_entry_idx
    );
};

#endif
