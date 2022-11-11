/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <limits>

#include "float_pair_list.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "pwlut.h"
#include "volume.h"
#include "volume_adjust.h"

Volume::Pointer
volume_adjust (const Volume::Pointer& image_in, const Float_pair_list& al)
{
    Volume::Pointer vol_out = image_in->clone (PT_FLOAT);
    float *vol_img = vol_out->get_raw<float> ();
    
    Pwlut pwlut;
    pwlut.set_lut (al);

    for (plm_long v = 0; v < vol_out->npix; v++) {
        vol_img[v] = pwlut.lookup (vol_img[v]);
    }
    return vol_out;
}

Volume::Pointer
volume_adjust (const Volume::Pointer& image_in, const std::string& adj_string)
{
    Float_pair_list al = parse_float_pairs (adj_string);

    if (al.empty()) {
        print_and_exit ("Error: couldn't parse adjust string: %s\n",
            adj_string.c_str());
    }

    return volume_adjust (image_in, al);
}
