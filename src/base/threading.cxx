/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <string>
#include <string.h>
#include "threading.h"

Threading
threading_parse (const std::string& string)
{
    return threading_parse (string.c_str());
}

Threading
threading_parse (const char* string)
{
    if (!strcmp (string,"single")) {
        return THREADING_CPU_SINGLE;
    }
    else if (!strcmp (string,"cpu") || !strcmp (string,"openmp")) {
#if (OPENMP_FOUND)
        return THREADING_CPU_OPENMP;
#else
        return THREADING_CPU_SINGLE;
#endif
    }
    else if (!strcmp (string,"cuda") || !strcmp (string,"gpu")) {
#if (CUDA_FOUND)
        return THREADING_CUDA;
#elif (OPENMP_FOUND)
        return THREADING_CPU_OPENMP;
#else
        return THREADING_CPU_SINGLE;
#endif
    }
    else if (!strcmp (string,"opencl")) {
#if (OPENCL_FOUND)
        return THREADING_OPENCL;
#elif (CUDA_FOUND)
        return THREADING_CUDA;
#elif (OPENMP_FOUND)
        return THREADING_CPU_OPENMP;
#else
        return THREADING_CPU_SINGLE;
#endif
    }
    else {
#if (OPENMP_FOUND)
        return THREADING_CPU_OPENMP;
#else
        return THREADING_CPU_SINGLE;
#endif
    }
}
