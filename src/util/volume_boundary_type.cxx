/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <string.h>
#include "volume_boundary_type.h"

Volume_boundary_type
volume_boundary_type_parse (const std::string& string)
{
    return volume_boundary_type_parse (string.c_str());
}

Volume_boundary_type
volume_boundary_type_parse (const char* string)
{
    if (!strcmp (string,"interior-edge")) {
        return INTERIOR_EDGE;
    }
    else if (!strcmp (string,"interior-face")) {
        return INTERIOR_FACE;
    }
    else {
        /* ?? */
        return INTERIOR_EDGE;
    }
}

