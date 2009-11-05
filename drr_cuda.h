/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _drr_cuda_h_
#define _drr_cuda_h_

#include "drr_opts.h"
#include "fdk_opts.h"  /* Temporary */
#include "volume.h"

#if defined __cplusplus
extern "C" {
#endif

int CUDA_DRR3 (Volume *vol, Fdk_options *options);
int CUDA_DRR (Volume *vol, Fdk_options *options);

#if defined __cplusplus
}
#endif

#endif
