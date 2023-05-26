/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include "cuda_texture.h"

Cuda_texture::Cuda_texture ()
{
    dev = 0;
    tex = 0;
    surf = 0;
}

Cuda_texture::~Cuda_texture ()
{
    // GCS TODO: Implement destructor
}

void
Cuda_texture::make_and_bind (const plm_long *dim, const float *source)
{
    CUDA_malloc_3d_array (&dev, dim);
    CUDA_bind_texture (&tex, dev);
    CUDA_bind_surface (&surf, dev);
    if (source) {
        CUDA_memcpy_to_3d_array (&dev, dim, source);
    } else {
        CUDA_clear_3d_array (&surf, dim);
    }
}

__global__ void
probe_kernel (
    cudaSurfaceObject_t surf,
    cudaTextureObject_t tex,
    dim3 kdim, dim3 idx, float* value)
{
    // calculate surface coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;

    *value = 0.f;
    if (x < kdim.x && y < kdim.y && z < kdim.z) {

        // N.b.
        // When using textures, you need to add 0.5.  The following codes are equivalent
        // surf3Dread<float> (&value, surf, x * 4, y , z)
        // value = tex3D<float> (tex, idx.x+0.5, idx.y+0.5, idx.z+0.5)

        surf3Dread<float> (value, surf, idx.x * 4, idx.y , idx.z, cudaBoundaryModeTrap);
    }
}

float
Cuda_texture::probe (int i, int j, int k, const plm_long *dim)
{
    dim3 kdim (dim[0], dim[1], dim[2]);
    dim3 idx (i, j, k);

    float probe_value;
    float *probe_pointer;
    cudaMalloc (&probe_pointer, sizeof (float));
    probe_kernel <<< 1,1 >>> (surf, tex, kdim, idx, probe_pointer);
    cudaMemcpy (&probe_value, probe_pointer, sizeof(float),
        cudaMemcpyDeviceToHost);
    cudaFree (probe_pointer);
    return probe_value;
}
