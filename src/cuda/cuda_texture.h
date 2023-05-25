/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cuda_texture_h_
#define _cuda_texture_h_

#include "plm_config.h"
#include "cuda_kernel_util.h"

/* From https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html

"In the case of texture access, if a texture reference is bound to a
linear array in global memory, then the device code can write to the
underlying array. Texture references that are bound to CUDA arrays can
be written to via surface-write operations by binding a surface to the
same underlying CUDA array storage). Reading from a texture while
writing to its underlying global memory array in the same kernel
launch should be avoided because the texture caches are read-only and
are not invalidated when the associated global memory is modified."

This paragraph leaves two questions unaddressed. (1) Does using a 
surface-write operation count as "writing to its underlying global 
memory array"?  (2) Is it safe to write to global memory in one 
kernel and then read from a texture in a subsequent kernel?

The answer to (1) is unknown, and I assume is not safe.  However, 
the answer to (2) is "yes", and described as follows:

"Within a kernel call, the texture cache is not kept coherent with
respect to global memory writes, so texture fetches from addresses
that have been written via global stores in the same kernel call
return undefined data. That is, a thread can safely read a memory
location via texture if the location has been updated by a previous
kernel call or memory copy, but not if it has been previously updated
by the same thread or another thread within the same kernel call."

*/

class Cuda_texture {
public:
    cudaArray_t dev;
    cudaTextureObject_t tex;
    cudaSurfaceObject_t surf;
    Cuda_texture ();
    ~Cuda_texture ();
    void make_and_bind (const plm_long *dim, const float *source);
    float probe (int i, int j, int k, const plm_long *dim);
};

#endif
