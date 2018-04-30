#include "plm_config.h"
#include <stdio.h>
#include "delayload.h"

int
main (int argc, char* argv[])
{
#if PLM_CONFIG_LEGACY_CUDA_DELAYLOAD
    int found = delayload_libplmcuda ();
#else
    bool found = check_library ("nvcuda.dll");
#endif
    printf ("CUDA library was %sfound\n", found ? "" : "not ");
    return 0;
}
