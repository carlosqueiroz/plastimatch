/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _distance_map_cuda_h_
#define _distance_map_cuda_h_

#include "plmutil_config.h"
#include "delayload.h"

PLMUTILCUDA_API
DELAYLOAD_WRAP (
void distance_map_cuda,
    void *dummy_var
);

#endif
