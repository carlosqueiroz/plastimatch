/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_adjust_h_
#define _volume_adjust_h_

#include "plmutil_config.h"
#include "float_pair_list.h"

PLMUTIL_API Volume::Pointer
volume_adjust (const Volume::Pointer& image_in, const Float_pair_list& al);
PLMUTIL_API Volume::Pointer
volume_adjust (const Volume::Pointer& image_in, const std::string& adj_string);

#endif
