/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "delayload.h"
#include "drr.h"
#include "drr_cuda.h"
#include "drr_opencl.h"
#include "drr_opts.h"
#include "drr_options.h"
#include "plm_image.h"
#include "plm_math.h"
#include "plm_timer.h"
#include "proj_image.h"
#include "proj_matrix.h"
#include "ray_trace.h"
#include "string_util.h"
#include "threading.h"
#include "volume.h"

int
main (int argc, char* argv[])
{
    Drr_options options;

    parse_args (&options, argc, argv);

    drr_compute (&options);
    
    printf ("Done.\n");
    return 0;
}
