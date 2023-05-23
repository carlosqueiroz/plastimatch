/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_kernel_util.h"
#include "cuda_util.h"

// NOTE: Helper functions in this file accept parameters that use CUDA specific
//       types (dim3, float4, etc).  As a result, these helper functions can
//       only be called from within other CUDA files (.cu)
//
//       The file 'cuda_util.cu' provides utility functions that do not use CUDA
//       specific types as parameters; and can, therefore, be used from
//       standard C or C++ files.
//
//       __device__ functions cannot be placed in here due to nvcc limitations.
//       Please place __device__ functions in cuda_kernel_util.inc

void
CUDA_array2vec_int3 (
    int3* vec,
    plm_long* array
)
{
    vec->x = array[0];
    vec->y = array[1];
    vec->z = array[2];
}

void
CUDA_array2vec_float3 (
    float3* vec,
    float* array
)
{
    vec->x = array[0];
    vec->y = array[1];
    vec->z = array[2];
}

// Builds execution configurations for kernels that
// assign one thread per element (1tpe).
int
CUDA_exec_conf_1tpe (
    dim3 *dimGrid,          // OUTPUT: Grid  dimensions
    dim3 *dimBlock,         // OUTPUT: Block dimensions
    int num_threads,        // INPUT: Total # of threads
    int threads_per_block,  // INPUT: Threads per block
    bool negotiate          // INPUT: Is threads per block negotiable?
)
{
    int i;
    int Grid_x = 0;
    int Grid_y = 0;
    int sqrt_num_blocks;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    if (negotiate) {
        int found_flag = 0;
        int j = 0;

        // Search for a valid execution configuration for the required # of blocks.
        // Block size has been specified as changable.  This helps if the
        // number of blocks required is a prime number > 65535.  Changing the
        // # of threads per block will change the # of blocks... which hopefully
        // won't be prime again.
        for (j = threads_per_block; j > 32; j -= 32) {
            num_blocks = (num_threads + j - 1) / j;
            sqrt_num_blocks = (int)sqrt((float)num_blocks);

            for (i = sqrt_num_blocks; i < GRID_LIMIT_X; i++) {
                if (num_blocks % i == 0) {
                    Grid_x = i;
                    Grid_y = num_blocks / Grid_x;
                    found_flag = 1;
                    break;
                }
            }

            if (found_flag == 1) {
                threads_per_block = j;
                break;
            }
        }

    } else {

        // Search for a valid execution configuration for the required # of blocks.
        // The calling algorithm has specifed that # of threads per block
        // is non negotiable.
        sqrt_num_blocks = (int)sqrt((float)num_blocks);

        for (i = sqrt_num_blocks; i < GRID_LIMIT_X; i++) {
            if (num_blocks % i == 0) {
                Grid_x = i;
                Grid_y = num_blocks / Grid_x;
                break;
            }
        }
    }



    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
        printf ("\n");
        printf ("[GPU KERNEL PANIC] Unable to find suitable execution configuration!");
        printf ("Terminating...\n");
        exit (0);
    } else {
        // callback function could be added
        // to arguments and called here if you need
        // to do something fancy upon success.
#if VERBOSE
        printf ("Grid [%i,%i], %d threads_per_block.\n", 
            Grid_x, Grid_y, threads_per_block);
#endif
    }

    // Pass configuration back by reference
    dimGrid->x = Grid_x;
    dimGrid->y = Grid_y;
    dimGrid->z = 1;

    dimBlock->x = threads_per_block;
    dimBlock->y = 1;
    dimBlock->z = 1;

    // Return the # of blocks we decided on just
    // in case we need it later to allocate shared memory, etc.
    return num_blocks;
}

// Builds execution configurations for kernels that
// assign one block per element (1bpe).
void
CUDA_exec_conf_1bpe (
    dim3 *dimGrid,          // OUTPUT: Grid  dimensions
    dim3 *dimBlock,         // OUTPUT: Block dimensions
    int num_blocks,         // INPUT: Number of blocks
    int threads_per_block)  // INPUT: Threads per block
{
    int i;
    int Grid_x = 0;
    int Grid_y = 0;

    // Search for a valid execution configuration for the required # of blocks.
    int sqrt_num_blocks = (int)sqrt((float)num_blocks);

    for (i = sqrt_num_blocks; i < 65535; i++) {
        if (num_blocks % i == 0) {
            Grid_x = i;
            Grid_y = num_blocks / Grid_x;
            break;
        }
    }


    // Were we able to find a valid exec config?
    if (Grid_x == 0) {
        printf ("\n");
        printf ("[GPU KERNEL PANIC] Unable to find suitable execution configuration!");
        printf ("Terminating...\n");
        exit (0);
    } else {
        // callback function could be added
        // to arguments and called here if you need
        // to do something fancy upon success.
#if VERBOSE
        printf ("Grid [%i,%i], %d threads_per_block.\n", 
            Grid_x, Grid_y, threads_per_block);
#endif
    }

    // Pass configuration back by reference
    dimGrid->x = Grid_x;
    dimGrid->y = Grid_y;
    dimGrid->z = 1;

    dimBlock->x = threads_per_block;
    dimBlock->y = 1;
    dimBlock->z = 1;
}

void
CUDA_timer_start (cuda_timer *timer)
{
    cudaEventCreate (&timer->start);
    cudaEventCreate (&timer->stop);
    cudaEventRecord (timer->start, 0);
}

// Returns time in milliseconds
float
CUDA_timer_report (cuda_timer *timer)
{
    float time;
    cudaEventRecord (timer->stop, 0);
    cudaEventSynchronize (timer->stop);
    cudaEventElapsedTime (&time, timer->start, timer->stop);
    cudaEventDestroy (timer->start);
    cudaEventDestroy (timer->stop);
    return time;
}

// https://stackoverflow.com/questions/59899751/memset-cuarray-for-surface-memory
__global__ void
CUDA_clear_3d_array_kernel (cudaSurfaceObject_t surf, dim3 kdim)
{
    // calculate surface coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;

    // write to memeory
    if (x < kdim.x && y < kdim.y && z < kdim.z) {
        surf3Dwrite<float> (0.f, surf, x*4, y, z, cudaBoundaryModeTrap);
    }
}

void
CUDA_clear_3d_array (cudaSurfaceObject_t *surf, const plm_long *dim)
{
    int threadX = 256;  // BLOCK_SIZE
    int threadY = 1;
    int threadZ = 1;
    int blockX = (dim[0] + threadX - 1) / threadX;
    int blockY = (dim[1] + threadY - 1) / threadY;
    int blockZ = (dim[2] + threadZ - 1) / threadZ;
    dim3 block_dim = dim3(threadX, threadY, threadZ);
    dim3 grid_dim = dim3(blockX, blockY, blockZ);
    dim3 kdim (dim[0], dim[1], dim[2]);
    CUDA_clear_3d_array_kernel <<< grid_dim, block_dim >>> (*surf, kdim);
}

void
CUDA_memcpy_to_3d_array (
    cudaArray_t *dev_array,
    const plm_long *dim,
    const float *source    
)
{
    cudaExtent ca_extent;
    ca_extent.width  = dim[0];
    ca_extent.height = dim[1];
    ca_extent.depth  = dim[2];

    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.extent   = ca_extent;
    cpy_params.kind     = cudaMemcpyHostToDevice;
    cpy_params.dstArray = *dev_array;

    cpy_params.srcPtr = make_cudaPitchedPtr ((void*)source,
        ca_extent.width * sizeof(float), ca_extent.width , ca_extent.height);

    cudaMemcpy3D (&cpy_params);
    CUDA_check_error ("cudaMemcpy3D (to) failed");
}

void
CUDA_memcpy_from_3d_array (
    float *dest, 
    const plm_long *dim,
    cudaArray_t *dev_array
)
{
    cudaExtent ca_extent;
    ca_extent.width  = dim[0];
    ca_extent.height = dim[1];
    ca_extent.depth  = dim[2];

    cudaMemcpy3DParms cpy_params = {0};
    cpy_params.extent   = ca_extent;
    cpy_params.kind     = cudaMemcpyDeviceToHost;
    cpy_params.srcArray = *dev_array;

    cpy_params.dstPtr = make_cudaPitchedPtr ((void*)dest,
        ca_extent.width * sizeof(float), ca_extent.width , ca_extent.height);

    cudaMemcpy3D (&cpy_params);
    CUDA_check_error ("cudaMemcpy3D (from) failed");
}

void
CUDA_malloc_3d_array (
    cudaArray_t *dev_array,
    const plm_long *dim
)
{
    cudaChannelFormatDesc ca_descriptor;
    cudaExtent ca_extent;

    ca_descriptor = cudaCreateChannelDesc<float>();
    ca_extent.width  = dim[0];
    ca_extent.height = dim[1];
    ca_extent.depth  = dim[2];
    cudaMalloc3DArray (dev_array, &ca_descriptor, ca_extent);
    CUDA_check_error ("cudaMalloc3DArray failed");
}

void
CUDA_bind_texture (
    cudaTextureObject_t *tex_obj,
    cudaArray_t dev_array
)
{
    if (*tex_obj != 0) {
        cudaDestroyTextureObject (*tex_obj);
        *tex_obj = 0;
    }
    
    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dev_array;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.addressMode[2] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaCreateTextureObject (tex_obj, &resDesc, &texDesc, NULL);
    CUDA_check_error ("cudaCreateTextureObject failed");
}

void
CUDA_bind_surface (
    cudaSurfaceObject_t *surf,
    cudaArray_t dev_array
)
{
    if (*surf != 0) {
        cudaDestroySurfaceObject (*surf);
        *surf = 0;
    }

    // Specify surface
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dev_array;
    
    // Create the surface objects
    cudaCreateSurfaceObject(surf, &resDesc);
    CUDA_check_error ("cudaCreateSurfaceObject failed");
}
