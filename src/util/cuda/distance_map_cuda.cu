/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_util.h"
#include "distance_map_cuda.h"

__global__ void 
myFirstKernel(int *d_a)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    d_a[idx] = idx;  
}


void
distance_map_cuda (void *dummy_var)
{
    // pointer for host memory
    int *h_a;

    // pointer for device memory
    int *d_a;

    // define grid and block size
    int numBlocks = 8;
    int numThreadsPerBlock = 8;

    // Part 1 of 5: allocate host and device memory
    size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);
    h_a = (int *) malloc(memSize);
    cudaMalloc((void **) &d_a, memSize);

    CUDA_check_error("cudaMalloc");

    // Part 2 of 5: launch kernel
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(numThreadsPerBlock, 1, 1);

    // Part 3 of 5: implement the kernel
    myFirstKernel<<<dimGrid, dimBlock>>>(d_a);

    // block until the device has completed
    cudaDeviceSynchronize();

    // check if kernel execution generated an error
    CUDA_check_error("kernel execution");

    // Part 4 of 5: device to host copy
    cudaMemcpy( h_a, d_a, memSize, cudaMemcpyDeviceToHost );

    // Check for any CUDA errors
    CUDA_check_error("cudaMemcpy");

    // Part 5 of 5: verify the data returned to the host is correct
    for (int i = 0; i < numBlocks; i++)	{
	for (int j = 0; j < numThreadsPerBlock; j++) {
	    assert (h_a[i * numThreadsPerBlock + j] == i * numThreadsPerBlock + j);
	}
    }

    // free device memory
    cudaFree(d_a);

    // free host memory
    free(h_a);

    // If the program makes it this far, then the results are correct and
    // there are no run-time errors.  Good work!
    printf("Correct!\n");
    exit (0);
}
