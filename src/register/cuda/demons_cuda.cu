/* -----------------------------------------------------------------------
   see copyright.txt and license.txt for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdio.h>
#include <cuda.h>

#include "cuda_kernel_util.h"
#include "cuda_texture.h"
#include "cuda_util.h"
#include "demons.h"
#include "demons_cuda.h"
#include "demons_state.h"
#include "gaussian.h"
#include "plm_cuda_math.h"
#include "plm_timer.h"
#include "volume.h"

/* GCS 2023-05-09.  Why is this redefined from cuda_kernel_util.h ? */
#undef block_size
#define block_size 256
#define BLOCK_SIZE 256

/* Texture Memory */
class Demons_cuda_state {
public:
    Cuda_texture fixed;
    Cuda_texture moving;
    Cuda_texture grad_x;
    Cuda_texture grad_y;
    Cuda_texture grad_z;
    Cuda_texture grad_mag;
    Cuda_texture vf_est_x;
    Cuda_texture vf_est_y;
    Cuda_texture vf_est_z;
    Cuda_texture vf_smooth_x;
    Cuda_texture vf_smooth_y;
    Cuda_texture vf_smooth_z;
};

/*
Constant Memory
*/
__constant__ int c_dim[3];
__constant__ int c_moving_dim[3];
__constant__ float c_spacing_div2[3];
__constant__ float c_f2mo[3];
__constant__ float c_f2ms[3];
__constant__ float c_invmps[3];


/*
Constant Memory Functions
*/
void 
setConstantDimension (plm_long *h_dim)
{
    int i_dim[3] = { (int) h_dim[0], (int) h_dim[1], (int) h_dim[2] };
    cudaMemcpyToSymbol (c_dim, i_dim, sizeof(int3));
    //cudaMemcpyToSymbol(c_dim, h_dim, sizeof(int3));
}

void 
setConstantMovingDimension (plm_long *h_dim)
{
    int i_dim[3] = { (int) h_dim[0], (int) h_dim[1], (int) h_dim[2] };
    cudaMemcpyToSymbol (c_moving_dim, i_dim, sizeof(int3));
}

void setConstantPixelSpacing(float *h_spacing_div2)
{
    cudaMemcpyToSymbol(c_spacing_div2, h_spacing_div2, sizeof(float3));
}

void setConstantF2mo(float *h_f2mo)
{
    cudaMemcpyToSymbol(c_f2mo, h_f2mo, sizeof(float3));
}

void setConstantF2ms(float *h_f2ms)
{
    cudaMemcpyToSymbol(c_f2ms, h_f2ms, sizeof(float3));
}

void setConstantInvmps(float *h_invmps)
{
    cudaMemcpyToSymbol(c_invmps, h_invmps, sizeof(float3));
}

/*
Device Functions
*/
__device__ int volume_index_cuda (int *dims, int i, int j, int k)
{
    return i + (dims[0] * (j + dims[1] * k));
}

/*
Kernels
*/
__global__ void
calculate_gradient_magnitude_image_kernel (
    cudaTextureObject_t grad_x,
    cudaTextureObject_t grad_y,
    cudaTextureObject_t grad_z,
    cudaSurfaceObject_t grad_mag)
{
    // calculate surface coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= c_dim[0] || y >= c_dim[1] || z >= c_dim[2])
        return;

    long v = (z * c_dim[1] * c_dim[0]) + (y * c_dim[0]) + x;
    long v3 = v * 3;

    float vox_grad_x = tex3D<float> (grad_x, x, y, z);
    float vox_grad_y = tex3D<float> (grad_y, x, y, z);
    float vox_grad_z = tex3D<float> (grad_z, x, y, z);
    float val = vox_grad_x * vox_grad_x
        + vox_grad_y * vox_grad_y + vox_grad_z * vox_grad_z;

    surf3Dwrite (val, grad_mag, x * 4, y, z);
}

__global__ void 
estimate_displacements_kernel (
    cudaSurfaceObject_t vf_est_x,
    cudaSurfaceObject_t vf_est_y,
    cudaSurfaceObject_t vf_est_z,
    cudaTextureObject_t vf_smooth_x,
    cudaTextureObject_t vf_smooth_y,
    cudaTextureObject_t vf_smooth_z,
    cudaTextureObject_t fixed,
    cudaTextureObject_t moving,
    cudaTextureObject_t grad_x,
    cudaTextureObject_t grad_y,
    cudaTextureObject_t grad_z,
    cudaTextureObject_t grad_mag,
    float *ssd, 
    int *inliers, 
    float homog, 
    float denominator_eps, 
    float accel, 
    int blockY, 
    float invBlockY
)
{
    // calculate surface coordinates
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (i >= c_dim[0] || j >= c_dim[1] || k >= c_dim[2])
	return;

    long fv = (k * c_dim[1] * c_dim[0]) + (j * c_dim[0]) + i;
    long f3v = 3 * fv;

    float mi = c_f2mo[0] + i * c_f2ms[0];
    float mj = c_f2mo[1] + j * c_f2ms[1];
    float mk = c_f2mo[2] + k * c_f2ms[2];

    /* Find correspondence with nearest neighbor interpolation 
       and boundary checking */
    int mz = __float2int_rn (mk + c_invmps[2] 
	* tex3D<float> (vf_smooth_z, i, j, k));
    if (mz < 0 || mz >= c_moving_dim[2])
	return;

    int my = __float2int_rn (mj + c_invmps[1] 
	* tex3D<float> (vf_smooth_y, i, j, k));
    if (my < 0 || my >= c_moving_dim[1])
	return;

    int mx = __float2int_rn (mi + c_invmps[0] 
	* tex3D<float> (vf_smooth_x, i, j, k));
    if (mx < 0 || mx >= c_moving_dim[0])
	return;

    int mv = (mz * c_moving_dim[1] + my) * c_moving_dim[0] + mx;
    int m3v = 3 * mv;

    /* Find image difference at this correspondence */
    float diff = tex3D<float> (fixed, i, j, k)
        - tex3D<float> (moving, mx, my, mz);

    /* Compute denominator */
    float denom = tex3D<float> (grad_mag, mx, my, mz) + homog * diff * diff;

    /* Compute SSD for statistics */
    inliers[fv] = 1;
    ssd[fv] = diff * diff;

    /* Threshold the denominator to stabilize estimation */
    if (denom < denominator_eps) 
	return;

    /* Compute new estimate of displacement */
    float mult = accel * diff / denom;
    float data;
    surf3Dread (&data, vf_est_x, mx * 4, my, mz);
    data += mult * tex3D<float>(grad_x, mx, my, mz);
    surf3Dwrite (data, vf_est_x, mx * 4, my, mz);
    surf3Dread (&data, vf_est_y, mx * 4, my, mz);
    data += mult * tex3D<float>(grad_y, mx, my, mz);
    surf3Dwrite (data, vf_est_y, mx * 4, my, mz);
    surf3Dread (&data, vf_est_z, mx * 4, my, mz);
    data += mult * tex3D<float>(grad_z, mx, my, mz);
    surf3Dwrite (data, vf_est_z, mx * 4, my, mz);
}

template <class T> __global__ void
reduction(T *vectorData, int totalElements)
{
    __shared__ T vector[BLOCK_SIZE * 2];

    /* Find position in vector */
    int threadID = threadIdx.x;
    int blockID = blockIdx.x;
    int xInVector = BLOCK_SIZE * blockID * 2 + threadID;

    vector[threadID] = (xInVector < totalElements) ? vectorData[xInVector] : 0;
    vector[threadID + BLOCK_SIZE] = (xInVector + BLOCK_SIZE < totalElements) ? vectorData[xInVector + BLOCK_SIZE] : 0;
    __syncthreads();

    /* Calculate partial sum */
    for (int stride = BLOCK_SIZE; stride > 0; stride >>= 1) {
        if (threadID < stride)
            vector[threadID] += vector[threadID + stride];
        __syncthreads();
    }
    __syncthreads();

    if (threadID == 0)
        vectorData[blockID] = vector[0];
}

__device__ void
vf_conv_x (
    cudaSurfaceObject_t vf_out,
    cudaTextureObject_t vf_in,
    float *ker,
    int x,
    int y,
    int z,
    int i1,
    int j1,
    int j2
)
{
    float sum = 0.0;
    for (int i = i1, j = j1; j <= j2; i++, j++) {
        float data = tex3D<float> (vf_in, i, y, z);
        sum += ker[j] * data;
    }
    surf3Dwrite (sum, vf_out, x * 4, y, z);
}

__device__ void
vf_conv_y (
    cudaSurfaceObject_t vf_out,
    cudaTextureObject_t vf_in,
    float *ker,
    int x,
    int y,
    int z,
    int i1,
    int j1,
    int j2
)
{
    float sum = 0.0;
    for (int i = i1, j = j1; j <= j2; i++, j++) {
        float data = tex3D<float> (vf_in, x, i, z);
        sum += ker[j] * data;
    }
    surf3Dwrite (sum, vf_out, x * 4, y, z);
}

__device__ void
vf_conv_z (
    cudaSurfaceObject_t vf_out,
    cudaTextureObject_t vf_in,
    float *ker,
    int x,
    int y,
    int z,
    int i1,
    int j1,
    int j2
)
{
    float sum = 0.0;
    for (int i = i1, j = j1; j <= j2; i++, j++) {
        float data = tex3D<float> (vf_in, x, y, i);
        sum += ker[j] * data;
    }
    surf3Dwrite (sum, vf_out, x * 4, y, z);
}

__global__ void
vf_convolve_x_kernel (
    cudaSurfaceObject_t vf_out_x,
    cudaSurfaceObject_t vf_out_y,
    cudaSurfaceObject_t vf_out_z,
    cudaTextureObject_t vf_in_x,
    cudaTextureObject_t vf_in_y,
    cudaTextureObject_t vf_in_z,
    float *ker, int half_width, int blockY, float invBlockY)
{
    int i, i1;		/* i is the offset in the vf */
    int j, j1, j2;	/* j is the index of the kernel */
    int d;			/* d is the vector field direction */

    // calculate surface coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= c_dim[0] || y >= c_dim[1] || z >= c_dim[2])
        return;

    long v3 = 3 * ((z * c_dim[1] * c_dim[0]) + (y * c_dim[0]) + x);

    j1 = x - half_width;
    j2 = x + half_width;
    if (j1 < 0) j1 = 0;
    if (j2 >= c_dim[0]) {
        j2 = c_dim[0] - 1;
    }
    i1 = j1 - x;
    j1 = j1 - x + half_width;
    j2 = j2 - x + half_width;

    vf_conv_x (vf_out_x, vf_in_x, ker, x, y, z, i1, j1, j2);
    vf_conv_x (vf_out_y, vf_in_y, ker, x, y, z, i1, j1, j2);
    vf_conv_x (vf_out_z, vf_in_z, ker, x, y, z, i1, j1, j2);
}

__global__ void
vf_convolve_y_kernel (
    cudaSurfaceObject_t vf_out_x,
    cudaSurfaceObject_t vf_out_y,
    cudaSurfaceObject_t vf_out_z,
    cudaTextureObject_t vf_in_x,
    cudaTextureObject_t vf_in_y,
    cudaTextureObject_t vf_in_z,
    float *ker, int half_width, int blockY, float invBlockY)
{
    int i, i1;		/* i is the offset in the vf */
    int j, j1, j2;	/* j is the index of the kernel */
    int d;			/* d is the vector field direction */

    // calculate surface coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= c_dim[0] || y >= c_dim[1] || z >= c_dim[2])
        return;

    long v3 = 3 * ((z * c_dim[1] * c_dim[0]) + (y * c_dim[0]) + x);

    j1 = y - half_width;
    j2 = y + half_width;
    if (j1 < 0) j1 = 0;
    if (j2 >= c_dim[1]) {
        j2 = c_dim[1] - 1;
    }
    i1 = j1 - y;
    j1 = j1 - y + half_width;
    j2 = j2 - y + half_width;

    vf_conv_y (vf_out_x, vf_in_x, ker, x, y, z, i1, j1, j2);
    vf_conv_y (vf_out_y, vf_in_y, ker, x, y, z, i1, j1, j2);
    vf_conv_y (vf_out_z, vf_in_z, ker, x, y, z, i1, j1, j2);
}

__global__ void
vf_convolve_z_kernel (
    cudaSurfaceObject_t vf_out_x,
    cudaSurfaceObject_t vf_out_y,
    cudaSurfaceObject_t vf_out_z,
    cudaTextureObject_t vf_in_x,
    cudaTextureObject_t vf_in_y,
    cudaTextureObject_t vf_in_z,
    float *ker, int half_width, int blockY, float invBlockY)
{
    int i, i1;		/* i is the offset in the vf */
    int j, j1, j2;	/* j is the index of the kernel */
    int d;			/* d is the vector field direction */

    // calculate surface coordinates
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;

    if (x >= c_dim[0] || y >= c_dim[1] || z >= c_dim[2])
        return;

    long v3 = 3 * ((z * c_dim[1] * c_dim[0]) + (y * c_dim[0]) + x);

    j1 = z - half_width;
    j2 = z + half_width;
    if (j1 < 0) j1 = 0;
    if (j2 >= c_dim[2]) {
        j2 = c_dim[2] - 1;
    }
    i1 = j1 - z;
    j1 = j1 - z + half_width;
    j2 = j2 - z + half_width;

    vf_conv_z (vf_out_x, vf_in_x, ker, x, y, z, i1, j1, j2);
    vf_conv_z (vf_out_y, vf_in_y, ker, x, y, z, i1, j1, j2);
    vf_conv_z (vf_out_z, vf_in_z, ker, x, y, z, i1, j1, j2);
}

__global__ void
volume_calc_grad_kernel (
    cudaTextureObject_t moving,
    cudaSurfaceObject_t grad_x,
    cudaSurfaceObject_t grad_y,
    cudaSurfaceObject_t grad_z
)
{
    // calculate surface coordinates
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

    if (i >= c_dim[0] || j >= c_dim[1] || k >= c_dim[2])
        return;

    /* p is prev, n is next */
    int i_p = (i == 0) ? 0 : i - 1;
    int i_n = (i == c_dim[0] - 1) ? c_dim[0] - 1 : i + 1;
    int j_p = (j == 0) ? 0 : j - 1;
    int j_n = (j == c_dim[1] - 1) ? c_dim[1] - 1 : j + 1;
    int k_p = (k == 0) ? 0 : k - 1;
    int k_n = (k == c_dim[2] - 1) ? c_dim[2] - 1 : k + 1;

    long v3 = 3 * ((k * c_dim[1] * c_dim[0]) + (j * c_dim[0]) + i);

    long gi = v3;
    long gj = v3 + 1;
    long gk = v3 + 2;

    float val = 0;
    val = c_spacing_div2[0] *
        (tex3D<float>(moving, i_p, j, k) - tex3D<float>(moving, i_n, j, k));
    surf3Dwrite (val, grad_x, i * 4, j, k);

    val = c_spacing_div2[1] * 
        (tex3D<float>(moving, i, j_p, k) - tex3D<float>(moving, i, j_p, k));
    surf3Dwrite (val, grad_y, i * 4, j, k);

    val = c_spacing_div2[2] *
        (tex3D<float>(moving, i, j, k_p) - tex3D<float>(moving, i, j, k_n));
    surf3Dwrite (val, grad_z, i * 4, j, k);
}

void
demons_cuda (
    Demons_state *demons_state,
    Volume* fixed, 
    Volume* moving, 
    Volume* moving_grad, 
    Volume* vf_init, 
    Demons_parms* parms
)
{
    int i;
    int	it;		/* Iterations */
    float f2mo[3];	/* Origin difference (in cm) from fixed to moving */
    float f2ms[3];	/* Slope to convert fixed to moving */
    float invmps[3];	/* 1/pixel spacing of moving image */
    float *kerx, *kery, *kerz;
    int fw[3];
    double diff_run;
    //Volume *vf_est, *vf_smooth;
    int inliers;
    float ssd;

    Plm_timer* timer = new Plm_timer;
    Plm_timer* gpu_timer = new Plm_timer;
    Plm_timer* kernel_timer = new Plm_timer;

    int num_elements, half_num_elements, reductionBlocks;
    size_t vol_size, interleaved_vol_size, inlier_size;
    int *d_inliers;
    float total_runtime, spacing_div2[3];
    float *d_kerx, *d_kery, *d_kerz, *d_swap, *d_ssd;
    dim3 reductionGrid;

    Demons_cuda_state dcstate;
    
    printf ("Hello from demons_cuda()\n");

    // This code uses planar format
    vf_convert_to_planar (demons_state->vf_smooth);
    
    /* Initialize GPU timers */
    double gpu_time = 0;
    double kernel_time = 0;
	
    /* Determine GPU execution environment */
    int threadX = BLOCK_SIZE;
    int threadY = 1;
    int threadZ = 1;
    int blockX = (fixed->dim[0] + threadX - 1) / threadX;
    int blockY = (fixed->dim[1] + threadY - 1) / threadY;
    int blockZ = (fixed->dim[2] + threadZ - 1) / threadZ;
    dim3 block_dim = dim3(threadX, threadY, threadZ);
    dim3 grid_dim = dim3(blockX, blockY, blockZ);

    for (i = 0; i < 3; i++)
	spacing_div2[i] = 0.5 / moving->spacing[i];

    /* Determine size of device memory */
    vol_size = moving->dim[0] * moving->dim[1] * moving->dim[2] * sizeof(float);
    interleaved_vol_size = 3 * fixed->dim[0] * fixed->dim[1] * fixed->dim[2] * sizeof(float);
    inlier_size = moving->dim[0] * moving->dim[1] * moving->dim[2] * sizeof(int);

    /* Allocate device memory */
    gpu_timer->start ();
    cudaMalloc((void**)&d_ssd, vol_size);
    cudaMalloc((void**)&d_inliers, inlier_size);

    // Allocate device memory and bind to textures
    dcstate.fixed.make_and_bind (fixed->dim, (float*) fixed->img);
    dcstate.moving.make_and_bind (moving->dim, (float*) moving->img);
    dcstate.grad_x.make_and_bind (moving->dim, 0);
    dcstate.grad_y.make_and_bind (moving->dim, 0);
    dcstate.grad_z.make_and_bind (moving->dim, 0);
    dcstate.grad_mag.make_and_bind (moving->dim, 0);
    dcstate.vf_est_x.make_and_bind (moving->dim, 0);
    dcstate.vf_est_y.make_and_bind (moving->dim, 0);
    dcstate.vf_est_z.make_and_bind (moving->dim, 0);
    /* GCS FIX: initialize vf_smooth to initial guess if supplied */
    dcstate.vf_smooth_x.make_and_bind (moving->dim, 0);
    dcstate.vf_smooth_y.make_and_bind (moving->dim, 0);
    dcstate.vf_smooth_z.make_and_bind (moving->dim, 0);
    
    /* Copy/Initialize device memory */
    gpu_time += gpu_timer->report ();

    /* Set device constant memory */
    setConstantDimension(fixed->dim);
    setConstantMovingDimension(moving->dim);
    setConstantPixelSpacing(spacing_div2);

    /* Bind device texture memory */
    //cudaBindTexture(0, tex_fixed, d_fixed, vol_size);
    //cudaBindTexture(0, tex_moving, d_moving, vol_size);
    gpu_time += gpu_timer->report ();

    /* Check for any errors prekernel execution */
    CUDA_check_error("Error before kernel execution");

    /* Call kernel */
    kernel_timer->start ();
    volume_calc_grad_kernel<<< grid_dim, block_dim >>>(
        dcstate.moving.tex, dcstate.grad_x.surf,
        dcstate.grad_y.surf, dcstate.grad_z.surf);

    cudaDeviceSynchronize();
    kernel_time += kernel_timer->report ();

    /* Check for any errors postkernel execution */
    CUDA_check_error("Kernel execution failed");

    /* Call kernel */
    kernel_timer->start ();
    calculate_gradient_magnitude_image_kernel<<< grid_dim, block_dim >>> (
        dcstate.grad_x.tex, dcstate.grad_y.tex, dcstate.grad_z.tex,
        dcstate.grad_mag.surf);
    cudaDeviceSynchronize();
    kernel_time += kernel_timer->report ();

    /* Check for any errors postkernel execution */
    CUDA_check_error("Kernel execution failed");

    /* Validate filter widths */
    validate_filter_widths (fw, parms->filter_width);

    /* Create the seperable smoothing kernels for the x, y, and z directions */
    kerx = create_ker (parms->filter_std / fixed->spacing[0], fw[0]/2);
    kery = create_ker (parms->filter_std / fixed->spacing[1], fw[1]/2);
    kerz = create_ker (parms->filter_std / fixed->spacing[2], fw[2]/2);
    kernel_stats (kerx, kery, kerz, fw);

    /* Compute some variables for converting pixel sizes / origins */
    for (i = 0; i < 3; i++) {
	invmps[i] = 1 / moving->spacing[i];
	f2mo[i] = (fixed->origin[i] - moving->origin[i]) / moving->spacing[i];
	f2ms[i] = fixed->spacing[i] / moving->spacing[i];
    }

    /* Allocate device memory */
    gpu_timer->start ();
    printf ("Doing cudaMalloc\n");
    cudaMalloc ((void**)&d_kerx, fw[0] * sizeof(float));
    cudaMalloc ((void**)&d_kery, fw[1] * sizeof(float));
    cudaMalloc ((void**)&d_kerz, fw[2] * sizeof(float));

    /* Copy/Initialize device memory */
    printf ("Doing cudaMemcpy\n");
    cudaMemcpy (d_kerx, kerx, fw[0] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy (d_kery, kery, fw[1] * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy (d_kerz, kerz, fw[2] * sizeof(float), cudaMemcpyHostToDevice);

    /* Set device constant memory */
    setConstantF2ms (f2mo);
    setConstantF2ms (f2ms);
    setConstantInvmps (invmps);

    /* Bind device texture memory */
    printf ("Doing cudaBindTexture\n");
    gpu_time += gpu_timer->report ();

    timer->start ();

    /* Main loop through iterations.  At the start of each iteration, 
       the current displacement field will be in vf_smooth. */
    for (it = 0; it < parms->max_its; it++) {
	printf ("Looping...\n");
	inliers = 0; ssd = 0.0;

	/* Check for any errors prekernel execution */
	CUDA_check_error ("Error before kernel execution");

	gpu_timer->start ();
	cudaMemset(d_ssd, 0, vol_size);
	cudaMemset(d_inliers, 0, inlier_size);
	gpu_time += gpu_timer->report ();

	// Call kernel, new displacement estimates will be stored into vf_est.
	kernel_timer->start ();
	estimate_displacements_kernel<<< grid_dim, block_dim >>> (
            dcstate.vf_est_x.surf,
            dcstate.vf_est_y.surf,
            dcstate.vf_est_z.surf,
            dcstate.vf_smooth_x.tex,
            dcstate.vf_smooth_y.tex,
            dcstate.vf_smooth_z.tex,
            dcstate.fixed.tex,
            dcstate.moving.tex,
            dcstate.grad_x.tex,
            dcstate.grad_y.tex,
            dcstate.grad_z.tex,
            dcstate.grad_mag.tex,
	    d_ssd, 
	    d_inliers, 
	    parms->homog, 
	    parms->denominator_eps, 
	    parms->accel, 
	    blockY, 
	    1.0f / (float)blockY);
	cudaDeviceSynchronize ();
	kernel_time += kernel_timer->report ();

	/* Check for any errors postkernel execution */
	CUDA_check_error ("Kernel execution failed");

	// Calculate statistics for display
	num_elements = moving->dim[0] * moving->dim[1] * moving->dim[2];
	while (num_elements > 1) {
	    half_num_elements = num_elements / 2;
	    reductionBlocks = (half_num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

	    /* Invoke kernels */
	    dim3 reductionGrid(reductionBlocks, 1);
	    kernel_timer->start ();
	    reduction<float><<< reductionGrid, block_dim >>>(d_ssd, num_elements);
	    cudaDeviceSynchronize();
	    reduction<int><<< reductionGrid, block_dim >>>(d_inliers, num_elements);
	    cudaDeviceSynchronize();
	    kernel_time += kernel_timer->report ();

	    /* Check for any errors postkernel execution */
	    CUDA_check_error("Kernel execution failed");

	    num_elements = reductionBlocks;
	}

	gpu_timer->start ();
	cudaMemcpy(&ssd, d_ssd, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&inliers, d_inliers, sizeof(int), cudaMemcpyDeviceToHost);
	gpu_time += gpu_timer->report ();

	/* Print statistics */
	printf ("----- SSD = %.01f (%d/%d)\n", ssd/inliers, inliers, fixed->npix);

	/* Check for any errors prekernel execution */
	CUDA_check_error("Error before kernel execution");

	/* Smooth vf_est into vf_smooth.  The volumes are ping-ponged. */
	kernel_timer->start ();
	vf_convolve_x_kernel<<< grid_dim, block_dim >>> (
            dcstate.vf_smooth_x.surf,
            dcstate.vf_smooth_y.surf,
            dcstate.vf_smooth_z.surf,
            dcstate.vf_est_x.tex,
            dcstate.vf_est_y.tex,
            dcstate.vf_est_z.tex,
            d_kerx, fw[0] / 2, blockY, 1.0f / (float)blockY);
	cudaDeviceSynchronize();
	kernel_time += kernel_timer->report ();

	/* Check for any errors postkernel execution */
	CUDA_check_error("Kernel execution failed");

	/* Smooth vf_smooth into vf_est.  The volumes are ping-ponged. */
	kernel_timer->start ();
	vf_convolve_y_kernel<<< grid_dim, block_dim >>> (
            dcstate.vf_est_x.surf,
            dcstate.vf_est_y.surf,
            dcstate.vf_est_z.surf,
            dcstate.vf_smooth_x.tex,
            dcstate.vf_smooth_y.tex,
            dcstate.vf_smooth_z.tex,
            d_kery, fw[1] / 2, blockY, 1.0f / (float)blockY);
	cudaDeviceSynchronize();
	kernel_time += kernel_timer->report ();

	/* Check for any errors postkernel execution */
	CUDA_check_error("Kernel execution failed");

	/* Smooth vf_est into vf_smooth.  The volumes are ping-ponged. */
	vf_convolve_z_kernel<<< grid_dim, block_dim >>> (
            dcstate.vf_smooth_x.surf,
            dcstate.vf_smooth_y.surf,
            dcstate.vf_smooth_z.surf,
            dcstate.vf_est_x.tex,
            dcstate.vf_est_y.tex,
            dcstate.vf_est_z.tex,
            d_kerz, fw[2] / 2, blockY, 1.0f / (float)blockY);
	cudaDeviceSynchronize();
	kernel_time += kernel_timer->report ();

	/* Check for any errors postkernel execution */
	CUDA_check_error("Kernel execution failed");
    }

    /* Copy final output from device to host */
    float **img = (float**) demons_state->vf_smooth->img;
    gpu_timer->start ();
    CUDA_memcpy_from_3d_array (img[0], fixed->dim, &(dcstate.vf_smooth_x.dev));
    CUDA_memcpy_from_3d_array (img[1], fixed->dim, &(dcstate.vf_smooth_y.dev));
    CUDA_memcpy_from_3d_array (img[2], fixed->dim, &(dcstate.vf_smooth_z.dev));
    gpu_time += gpu_timer->report ();

    // Host expects interleaved
    vf_convert_to_interleaved (demons_state->vf_smooth);

    /* Print statistics */
    diff_run = timer->report ();
    printf("Time for %d iterations = %f (%f sec / it)\n", parms->max_its, diff_run, diff_run / parms->max_its);
    total_runtime = gpu_time + kernel_time;
    printf("\nTransfer run time: %f ms\n", gpu_time * 1000);
    printf("Kernel run time: %f ms\n", kernel_time * 1000);
    printf("Total CUDA run time: %f s\n\n", total_runtime);

    // Clean up
    free(kerx);
    free(kery);
    free(kerz);

    delete timer;
    delete kernel_timer;
    delete gpu_timer;

    /* Free device global memory */
    cudaFree(d_ssd);
    cudaFree(d_inliers);
}
