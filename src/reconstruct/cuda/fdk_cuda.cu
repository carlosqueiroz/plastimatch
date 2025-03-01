/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmreconstruct_config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

#include "cuda_util.h"
#include "fdk_cuda_p.h"
#include "plm_math.h"
#include "proj_image_dir.h"
#include "volume.h"

/****************************************************\
* Uncomment the line below to enable verbose output. *
* Enabling this should not nerf performance.         *
\****************************************************/
//#define VERBOSE

/**********************************************************\
* Uncomment the line below to enable detailed performance  *
* reporting.  This measurement alters the system, however, *
* resulting in significantly slower kernel execution.      *
\**********************************************************/
//#define TIME_KERNEL

// P R O T O T Y P E S ////////////////////////////////////////////////////
__global__ void kernel_fdk (float *dev_vol, int2 img_dim, float2 ic, float3 nrm, float sad, float scale, float3 vol_origin, int3 vol_dim, float3 vol_spacing, unsigned int Blocks_Y, float invBlocks_Y);
///////////////////////////////////////////////////////////////////////////



// T E X T U R E S ////////////////////////////////////////////////////////
#if defined (commentout)
texture<float, 1, cudaReadModeElementType> tex_img;
texture<float, 1, cudaReadModeElementType> tex_matrix;
#endif

#if defined (commentoout)
cudaTextureObject_t tex_img = 0;
cudaTextureObject_t tex_matrix = 0;
#endif


//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__
void kernel_fdk_gmem (
    float *dev_vol,
    float *pimg,
    float *pmat,
    int2 img_dim,
    float2 ic,
    float3 nrm,
    float sad,
    float scale,
    float3 vol_origin,
    int3 vol_dim,
    float3 vol_spacing,
    unsigned int Blocks_Y,
    float invBlocks_Y)
{
    // CUDA 2.0 does not allow for a 3D grid, which severely
    // limits the manipulation of large 3D arrays of data.  The
    // following code is a hack to bypass this implementation
    // limitation.
    unsigned int blockIdx_z = __float2uint_rd(blockIdx.y * invBlocks_Y);
    unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
    unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
    unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

    if (i >= vol_dim.x || j >= vol_dim.y || k >= vol_dim.z) {
        return; 
    }

    // Index row major into the volume
    long int vol_idx = i + ( j*(vol_dim.x) ) + ( k*(vol_dim.x)*(vol_dim.y) );

    float3 vp;
    float3 ip;
    int2 ip_r;
    float  s;
    float voxel_data;

    // origin volume coords
    vp.x = vol_origin.x + i * vol_spacing.x;
    vp.y = vol_origin.y + j * vol_spacing.y;
    vp.z = vol_origin.z + k * vol_spacing.z;

    // matrix multiply
    ip.x = pmat[0]*vp.x + pmat[1]*vp.y + pmat[2]*vp.z + pmat[3];
    ip.y = pmat[4]*vp.x + pmat[5]*vp.y + pmat[6]*vp.z + pmat[7];
    ip.z = pmat[8]*vp.x + pmat[9]*vp.y + pmat[10]*vp.z + pmat[11];

    // Change coordinate systems
    ip.x = ic.x + ip.x / ip.z;
    ip.y = ic.y + ip.y / ip.z;

    // Get pixel from 2D image
    ip_r.x = __float2int_rd(ip.x);
    ip_r.y = __float2int_rd(ip.y);

    // Clip against image dimensions
    if (ip_r.x < 0 || ip_r.x >= img_dim.x || ip_r.y < 0 || ip_r.y >= img_dim.y) {
        return;
    }
    voxel_data = pimg[ip_r.x*img_dim.x + ip_r.y];

    // Dot product
    s = nrm.x*vp.x + nrm.y*vp.y + nrm.z*vp.z;

    // Conebeam weighting factor
    s = sad - s;
    s = (sad * sad) / (s * s);

    // Place it into the volume
    dev_vol[vol_idx] += scale * s * voxel_data;
}


__global__
void kernel_fdk (
    float *dev_vol,
    cudaTextureObject_t tex_img,
    //cudaArray_t dev_img, 
    //float *dev_image,
    float *dev_matrix,
    int2 img_dim,
    float2 ic,
    float3 nrm,
    float sad,
    float scale,
    float3 vol_origin,
    int3 vol_dim,
    float3 vol_spacing,
    unsigned int Blocks_Y,
    float invBlocks_Y
)
{
    // CUDA 2.0 does not allow for a 3D grid, which severely
    // limits the manipulation of large 3D arrays of data.  The
    // following code is a hack to bypass this implementation
    // limitation.
    unsigned int blockIdx_z = __float2uint_rd(blockIdx.y * invBlocks_Y);
    unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
    unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
    unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

    if (i >= vol_dim.x || j >= vol_dim.y || k >= vol_dim.z) {
        return; 
    }

    // Index row major into the volume
    long int vol_idx = i + ( j*(vol_dim.x) ) + ( k*(vol_dim.x)*(vol_dim.y) );

    float3 vp;
    float3 ip;
    float  s;
    float voxel_data;

    // origin volume coords
    vp.x = vol_origin.x + i * vol_spacing.x;
    vp.y = vol_origin.y + j * vol_spacing.y;
    vp.z = vol_origin.z + k * vol_spacing.z;

    // matrix multiply
#if defined (commentout)
    ip.x = tex1Dfetch(tex_matrix, 0)*vp.x + tex1Dfetch(tex_matrix, 1)*vp.y + tex1Dfetch(tex_matrix, 2)*vp.z + tex1Dfetch(tex_matrix, 3);
    ip.y = tex1Dfetch(tex_matrix, 4)*vp.x + tex1Dfetch(tex_matrix, 5)*vp.y + tex1Dfetch(tex_matrix, 6)*vp.z + tex1Dfetch(tex_matrix, 7);
    ip.z = tex1Dfetch(tex_matrix, 8)*vp.x + tex1Dfetch(tex_matrix, 9)*vp.y + tex1Dfetch(tex_matrix, 10)*vp.z + tex1Dfetch(tex_matrix, 11);
#endif
    ip.x = dev_matrix[0]*vp.x + dev_matrix[1]*vp.y + dev_matrix[2]*vp.z + dev_matrix[3];
    ip.y = dev_matrix[4]*vp.x + dev_matrix[5]*vp.y + dev_matrix[6]*vp.z + dev_matrix[7];
    ip.z = dev_matrix[8]*vp.x + dev_matrix[9]*vp.y + dev_matrix[10]*vp.z + dev_matrix[11];

    // Change coordinate systems
    ip.x = ic.x + ip.x / ip.z;
    ip.y = ic.y + ip.y / ip.z;

    // Get pixel from 2D image
    ip.x = __float2int_rd(ip.x);
    ip.y = __float2int_rd(ip.y);

    // Clip against image dimensions
    if (ip.x < 0 || ip.x >= img_dim.x || ip.y < 0 || ip.y >= img_dim.y) {
        return;
    }
    voxel_data = tex2D<float>(tex_img, ip.y, ip.x);

    // Dot product
    s = nrm.x*vp.x + nrm.y*vp.y + nrm.z*vp.z;

    // Conebeam weighting factor
    s = sad - s;
    s = (sad * sad) / (s * s);

    // Place it into the volume
    dev_vol[vol_idx] += scale * s * voxel_data;
}

void*
fdk_cuda_state_create_cu (
    Volume *vol, 
    unsigned int image_npix, 
    float scale, 
    Fdk_parms *parms
)
{
    Fdk_cuda_state *state;
    Fdk_cuda_kernel_args *kargs;

    // Thread Block Dimensions
    int tBlock_x = 16;
    int tBlock_y = 4;
    int tBlock_z = 4;

    // Each element in the volume (each voxel) gets 1 thread
    int blocksInX = (vol->dim[0]+tBlock_x-1)/tBlock_x;
    int blocksInY = (vol->dim[1]+tBlock_y-1)/tBlock_y;
    int blocksInZ = (vol->dim[2]+tBlock_z-1)/tBlock_z;
    //dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
    //dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

    int vol_size_malloc = vol->npix * sizeof(float);

#if defined (VERBOSE)
    // First, we need to allocate memory on the host device
    // for the 3D volume of voxels that will hold our reconstruction.
    printf("========================================\n");
    printf("Allocating %dMB of video memory...", vol_size_malloc/1048576);
#endif

    /* Allocate memory */
    state = (Fdk_cuda_state*) malloc (sizeof (Fdk_cuda_state));
    cudaMalloc ((void**) &state->dev_matrix, 12*sizeof(float));
    cudaMalloc ((void**) &state->dev_kargs, sizeof(Fdk_cuda_kernel_args));
    cudaMalloc ((void**) &state->dev_vol, vol_size_malloc);
    cudaMemset ((void*) state->dev_vol, 0, vol_size_malloc);  
    CUDA_check_error ("Unable to allocate memory for volume");
    cudaMalloc ((void**) &state->dev_img, image_npix * sizeof(float));
    CUDA_check_error ("Unable to allocate memory for projection image");

#if defined (VERBOSE)
    printf(" done.\n\n");
#endif

    /* Set grid and block size */
    state->dimGrid = dim3 (blocksInX, blocksInY * blocksInZ);
    state->dimBlock = dim3 (tBlock_x, tBlock_y, tBlock_z);
    state->blocksInY = blocksInY;

    /* Set static kernel arguments */
    kargs = &state->kargs;
    kargs->scale = scale;
    kargs->vol_origin.x = vol->origin[0];
    kargs->vol_origin.y = vol->origin[1];
    kargs->vol_origin.z = vol->origin[2];
    kargs->vol_dim.x = vol->dim[0];
    kargs->vol_dim.y = vol->dim[1];
    kargs->vol_dim.z = vol->dim[2];
    kargs->vol_spacing.x = vol->spacing[0];
    kargs->vol_spacing.y = vol->spacing[1];
    kargs->vol_spacing.z = vol->spacing[2];

#if defined (VERBOSE)
    // State the kernel execution parameters
    printf("kernel parameters:\n dimGrid: %u, %u\n dimBlock: %u, %u, %u\n", 
        state->dimGrid.x, state->dimGrid.y,
        state->dimBlock.x, state->dimBlock.y, state->dimBlock.z);
    printf("%u voxels in volume\n", vol->npix);
    printf ("parms->last_img: %i\n", parms->last_img);

    printf("%u projections to process\n", 1+(parms->last_img - parms->first_img) / parms->skip_img);
    printf("%u Total Operations\n", vol->npix * (1+(parms->last_img - parms->first_img) / parms->skip_img));
    printf("========================================\n\n");
#endif

    return (void*) state;
}

void
fdk_cuda_state_destroy_cu (
    void *dev_state
)
{
    Fdk_cuda_state *state = (Fdk_cuda_state*) dev_state;
    cudaFree (state->dev_img);
    cudaFree (state->dev_kargs);
    cudaFree (state->dev_matrix);
    cudaFree (state->dev_vol);
    free (dev_state);
}

void
fdk_cuda_queue_image_cu (
    void *dev_state, 
    int *dim, 
    double *ic, 
    double *nrm, 
    double sad, 
    double sid, 
    double *matrix, 
    float *img
)
{
    Fdk_cuda_state *state = (Fdk_cuda_state*) dev_state;
    Fdk_cuda_kernel_args *kargs = &state->kargs;

    /* Set dynamic kernel arguments */
    kargs->img_dim.x = dim[0];
    kargs->img_dim.y = dim[1];
    kargs->ic.x = ic[0];
    kargs->ic.y = ic[1];
    kargs->nrm.x = nrm[0];
    kargs->nrm.y = nrm[1];
    kargs->nrm.z = nrm[2];
    kargs->sad = sad;
    kargs->sid = sid;

    for (int j = 0; j < 12; j++) {
        kargs->matrix[j] = (float) matrix[j];
    }

    // Copy image pixel data to device memory
    cudaChannelFormatDesc channel =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&state->dev_img, &channel, dim[0], dim[1]);
    const size_t spitch = dim[0] * sizeof(float);
    cudaMemcpy2DToArray(state->dev_img, 0, 0, img, spitch,
        dim[0] * sizeof(float), dim[1], cudaMemcpyHostToDevice);

    // Copy matrix to device memory
    cudaMemcpy (
        state->dev_matrix,
        kargs->matrix,
        sizeof(kargs->matrix), 
        cudaMemcpyHostToDevice
    );

#if defined (commentoout)
    resDesc.res.linear.devPtr = dev->dev_matrix;
    cudaCreateTextureObject(&tex_matrix, &resDesc, &texDesc, NULL);
#endif
}

void
fdk_cuda_backproject_cu (void *dev_state)
{
    Fdk_cuda_state *state = (Fdk_cuda_state*) dev_state;
    Fdk_cuda_kernel_args *kargs = &state->kargs;

    cudaTextureObject_t tex_img = 0;
    // Create texture objects
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = state->dev_img;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaCreateTextureObject(&tex_img, &resDesc, &texDesc, NULL);
    
    // Note: cbi->img AND cbi->matrix are passed via texture memory
    kernel_fdk <<< state->dimGrid, state->dimBlock >>> (
        state->dev_vol,
        tex_img,
        state->dev_matrix,
        kargs->img_dim,
        kargs->ic,
        kargs->nrm,
        kargs->sad,
        kargs->scale,
        kargs->vol_origin,
        kargs->vol_dim,
        kargs->vol_spacing,
        state->blocksInY,
        1.0f / (float) state->blocksInY
    );

    CUDA_check_error ("Kernel Panic!");

#if FDK_CUDA_TIME_KERNEL
    // CUDA kernel calls are asynchronous...
    // In order to accurately time the kernel
    // execution time we need to set a thread
    // barrier here after its execution.
    cudaThreadSynchronize();
#endif

#if defined (VERBOSE)
    printf ("done.\n");
#endif

    // Unbind the image and projection matrix textures
    cudaDestroyTextureObject (tex_img);
    //cudaUnbindTexture (tex_matrix);
}

void
fdk_cuda_fetch_volume_cu (
    void *dev_state, 
    void *host_buf, 
    unsigned int copy_size
)
{
    Fdk_cuda_state *state = (Fdk_cuda_state*) dev_state;

    // Copy reconstructed volume from device to host
    //cudaMemcpy (vol->img, dev_vol, vol->npix * vol->pix_size, cudaMemcpyDeviceToHost);
    cudaMemcpy (host_buf, state->dev_vol, copy_size, cudaMemcpyDeviceToHost);
    CUDA_check_error ("Error: Unable to retrieve data volume.");
}
