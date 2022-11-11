/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "cuda_probe.h"

bool
cuda_probe (void)
{
    int devicecount = -1;
    printf ("Testing for CUDA...\n");

    int driver_version = 0, runtime_version = 0;
    cudaRuntimeGetVersion(&runtime_version);
    cudaDriverGetVersion(&driver_version);
    printf ("CUDA driver version: %d.%d\n",
        driver_version / 1000, (driver_version % 100)/10);
    printf ("CUDA runtime version: %d.%d\n",
        runtime_version / 1000, (runtime_version % 100)/10);
    
    cudaError_t rc = cudaGetDeviceCount (&devicecount);
    if (rc != cudaSuccess) {
	printf("Call to cudaGetDeviceCount() returned failure (%d)\n",
            (int) rc);
	return false;
    }

    if (devicecount == 0) {
	printf("Suitable CUDA environment not detected!\n");
	return false;
    }
    printf ("Devices found: %d\n", devicecount);

    // Get CUDA device properties.
    cudaDeviceProp props;
    cudaGetDeviceProperties (&props, 0);
    printf ("Compute Capability %d.%d\n", props.major, props.minor);
    if (props.major < 1) {
        printf ("Device is presumed NOT cuda capable.\n");
        return false;
    }

    printf ("Device is presumed cuda capable.\n");
    return true;
    // Unless proven otherwise, we assume no CUDA.
    printf ("Device is presumed NOT cuda capable.\n");
}
