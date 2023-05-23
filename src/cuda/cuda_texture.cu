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
