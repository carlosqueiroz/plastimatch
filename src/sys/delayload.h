/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _delayload_h_
#define _delayload_h_

#include "plmsys_config.h"
#if PLM_CONFIG_LEGACY_CUDA_DELAYLOAD
#include "delayload_legacy.h"
#else
#include "cuda_delayload.h"
#endif

#endif
