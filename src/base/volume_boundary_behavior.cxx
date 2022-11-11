/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <string>
#include <string.h>
#include "volume_boundary_behavior.h"

Volume_boundary_behavior
volume_boundary_behavior_parse (const std::string& string)
{
    return volume_boundary_behavior_parse (string.c_str());
}

Volume_boundary_behavior
volume_boundary_behavior_parse (const char* string)
{
    if (!strcmp (string,"zero-pad")) {
        return ZERO_PADDING;
    }
    else if (!strcmp (string,"edge-pad")) {
        return EDGE_PADDING;
    }
    else if (!strcmp (string,"adaptive")) {
        return ADAPTIVE_PADDING;
    }
    else {
        /* ?? */
        return ADAPTIVE_PADDING;
    }
}
