/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cuda_probe_h_
#define _cuda_probe_h_

#include "plm_config.h"
#include "delayload.h"

#if defined __cplusplus
extern "C" {
#endif

plmcuda_EXPORT (
bool cuda_probe,
    void
);

#if defined __cplusplus
}
#endif

#endif
