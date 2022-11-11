/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <deque>
#include <vector>
#include "itkImage.h"

#include "image_boundary.h"
#include "distance_map.h"
#include "distance_map_cuda.h"
#include "itk_distance_map.h"
#include "itk_image_type.h"
#include "plm_image.h"
#include "volume.h"
#include "volume_header.h"

class Distance_map_private {
public:
    Distance_map_private () {
        inside_is_positive = false;
        absolute_distance = false;
        use_squared_distance = false;
        maximum_distance = FLT_MAX;
        algorithm = Distance_map::DANIELSSON;
        vbb = ADAPTIVE_PADDING;
        vbt = INTERIOR_EDGE;
        threading = THREADING_CPU_OPENMP;
    }
public:
    Distance_map::Algorithm algorithm;
    bool inside_is_positive;
    bool absolute_distance;
    bool use_squared_distance;
    float maximum_distance;
    Volume_boundary_behavior vbb;
    Volume_boundary_type vbt;
    Threading threading;

    UCharImageType::Pointer input;
    FloatImageType::Pointer output;
public:
    void run_native_danielsson ();
    void run_native_maurer ();
    void run_itk_signed_danielsson ();
    void run_itk_signed_maurer ();
    void run_itk_signed_native ();
    void run_song_maurer ();
    void run ();
protected:
    void native_danielsson_initialize_face_distances (
        Volume::Pointer& vb, float *dm);
    void forward_propagate_i (
        float *dm,
        const Volume::Pointer& vb,
        const float* sp2,
        plm_long j, 
        plm_long k);
    void backward_propagate_i (
        float *dm,
        const Volume::Pointer& vb,
        const float* sp2,
        plm_long j, 
        plm_long k);
    void forward_propagate_j (
        float *dm,
        const Volume::Pointer& vb,
        const float* sp2,
        plm_long k);
    void backward_propagate_j (
        float *dm,
        const Volume::Pointer& vb,
        const float* sp2,
        plm_long k);
    void maurerFT (
        unsigned char *vol,
	float *sp2,
	int height, int width, int depth,
	float *output);
    void voronoiFT (
	int dim, 
        unsigned char *vol, 
        float *sp2,
	int height, int width, int depth, 
	float *output);
    void runVoronoiFT1D (
        unsigned char *vol, 
	float *sp2,
	int height, int width, int depth, 
	float *output);
    void runVoronoiFT2D ( 
	float *sp2,
	int height, int width, int depth, 
	float *output);
    void runVoronoiFT3D (
	float *sp2,
	int height, int width, int depth, 
	float *output);
    int removeFT2D (
	float *sp2,
	std::deque<std::vector<int>> &g_nodes,
	int *w, int *Rd);
    int removeFT3D (
	float *sp2,
	std::deque<std::vector<int>> &g_nodes,
	int *w, int *Rd);
    double ED (
	float *sp2,
	int vol_i, int vol_j, int vol_k,
	std::vector<int> &fv);
    void distTransform (
	unsigned char *vol, 
	float *sp2,
	int height, int width, int depth, 
	float *ed_out);
    double calcDist (
	float *sp2,
	double i, double j, double k,
	double target_i, double target_j, double target_k);
};

/* Define some macros */
#define SQ_DIST(idx,sp2)                        \
    dm[3*idx+0]*dm[3*idx+0]*sp2[0]              \
        + dm[3*idx+1]*dm[3*idx+1]*sp2[1]        \
        + dm[3*idx+2]*dm[3*idx+2]*sp2[2]
#define SQ_DIST_I(idx,sp2)                      \
    (dm[3*idx+0]+1)*(dm[3*idx+0]+1)*sp2[0]      \
        + dm[3*idx+1]*dm[3*idx+1]*sp2[1]        \
        + dm[3*idx+2]*dm[3*idx+2]*sp2[2]
#define SQ_DIST_J(idx,sp2)                              \
    dm[3*idx+0]*dm[3*idx+0]*sp2[0]                      \
        + (dm[3*idx+1]+1)*(dm[3*idx+1]+1)*sp2[1]        \
        + dm[3*idx+2]*dm[3*idx+2]*sp2[2]
#define SQ_DIST_K(idx,sp2)                              \
    dm[3*idx+0]*dm[3*idx+0]*sp2[0]                      \
        + dm[3*idx+1]*dm[3*idx+1]*sp2[1]                \
        + (dm[3*idx+2]+1)*(dm[3*idx+2]+1)*sp2[2]

#define COPY_I(new_idx,old_idx)                 \
    dm[3*new_idx+0] = dm[3*old_idx+0] + 1;      \
    dm[3*new_idx+1] = dm[3*old_idx+1];          \
    dm[3*new_idx+2] = dm[3*old_idx+2];
#define COPY_J(new_idx,old_idx)                 \
    dm[3*new_idx+0] = dm[3*old_idx+0];          \
    dm[3*new_idx+1] = dm[3*old_idx+1] + 1;      \
    dm[3*new_idx+2] = dm[3*old_idx+2];
#define COPY_K(new_idx,old_idx)                 \
    dm[3*new_idx+0] = dm[3*old_idx+0];          \
    dm[3*new_idx+1] = dm[3*old_idx+1];          \
    dm[3*new_idx+2] = dm[3*old_idx+2] + 1;


void
Distance_map_private::forward_propagate_i (
    float *dm,
    const Volume::Pointer& vb,
    const float* sp2,
    plm_long j, 
    plm_long k)
{
    /* Forward propagate i */
    for (plm_long i = 1; i < vb->dim[0]; i++) {
        plm_long vo = vb->index (i-1, j, k);   /* "old" voxel */
        plm_long vn = vb->index (i, j, k);     /* "new" voxel */
        if (dm[3*vo] == FLT_MAX) {
            continue;
        }
        if (dm[3*vn] == FLT_MAX) {
            COPY_I (vn, vo);
            continue;
        }
        float odist = SQ_DIST_I(vo,sp2);
        float ndist = SQ_DIST(vn,sp2);
        if (odist < ndist) {
            if (i == 14 && j == 19 && k == 20) {
                printf (">>> %f %f\n", odist, ndist);
            }
            COPY_I (vn, vo);
        }
    }
}

void
Distance_map_private::backward_propagate_i (
    float *dm,
    const Volume::Pointer& vb,
    const float* sp2,
    plm_long j, 
    plm_long k)
{
    /* Backward propagate i */
    for (plm_long i = vb->dim[0] - 2; i >= 0; i--) {
        plm_long vo = vb->index (i+1, j, k);   /* "old" voxel */
        plm_long vn = vb->index (i, j, k);     /* "new" voxel */
        if (dm[3*vo] == FLT_MAX) {
            continue;
        }
        if (dm[3*vn] == FLT_MAX) {
            COPY_I (vn, vo);
            continue;
        }
        float odist = SQ_DIST_I(vo,sp2);
        float ndist = SQ_DIST(vn,sp2);
        if (odist < ndist) {
            COPY_I (vn, vo);
        }
    }
}

void 
Distance_map_private::forward_propagate_j (
    float *dm,
    const Volume::Pointer& vb,
    const float* sp2,
    plm_long k)
{
    /* Propagate within j = 0 */
    this->forward_propagate_i (dm, vb, sp2, 0, k);
    this->backward_propagate_i (dm, vb, sp2, 0, k);

    /* Forward scan j */
    for (plm_long j = 1; j < vb->dim[1]; j++) {
        /* Propagate j */
        for (plm_long i = 0; i < vb->dim[0]; i++) {
            plm_long vo = vb->index (i, j-1, k);   /* "old" voxel */
            plm_long vn = vb->index (i, j, k);     /* "new" voxel */
            if (dm[3*vo] == FLT_MAX) {
                continue;
            }
            if (dm[3*vn] == FLT_MAX) {
                COPY_J (vn, vo);
                continue;
            }
            float odist = SQ_DIST_J(vo,sp2);
            float ndist = SQ_DIST(vn,sp2);
            if (odist < ndist) {
                COPY_J (vn, vo);
            }
        }
        /* Propagate along i */
        this->forward_propagate_i (dm, vb, sp2, j, k);
        this->backward_propagate_i (dm, vb, sp2, j, k);
    }
}

void 
Distance_map_private::backward_propagate_j (
    float *dm,
    const Volume::Pointer& vb,
    const float* sp2,
    plm_long k)
{
    /* Backward scan j */
    for (plm_long j = vb->dim[1] - 2; j >= 0; j--) {
        /* Propagate j */
        for (plm_long i = 0; i < vb->dim[0]; i++) {
            plm_long vo = vb->index (i, j+1, k);   /* "old" voxel */
            plm_long vn = vb->index (i, j, k);     /* "new" voxel */
            if (dm[3*vo] == FLT_MAX) {
                continue;
            }
            if (dm[3*vn] == FLT_MAX) {
                COPY_J (vn, vo);
                continue;
            }
            float odist = SQ_DIST_J(vo,sp2);
            float ndist = SQ_DIST(vn,sp2);
            if (odist < ndist) {
                COPY_J (vn, vo);
            }
        }
        /* Propagate along i */
        this->forward_propagate_i (dm, vb, sp2, j, k);
        this->backward_propagate_i (dm, vb, sp2, j, k);
    }
}

void
Distance_map_private::native_danielsson_initialize_face_distances (
    Volume::Pointer& vb, float *dm
)
{
    unsigned char *imgb = (unsigned char*) vb->img;

    float sp2[3] = {
        vb->spacing[0] * vb->spacing[0],
        vb->spacing[1] * vb->spacing[1],
        vb->spacing[2] * vb->spacing[2]
    };

    /* Yuck.  Loop through face image, and initialize voxels that have 
       faces abutting the segment.  Initialize the distance to the 
       nearest face, keeping in mind that a voxel may abut a segment 
       on multiple faces. */
    for (plm_long k = 0, v = 0; k < vb->dim[2]; k++) {
        for (plm_long j = 0; j < vb->dim[1]; j++) {
            for (plm_long i = 0; i < vb->dim[0]; i++, v++) {
                /* I */
                if (imgb[v] & VBB_MASK_NEG_I) {
                    if (SQ_DIST(v,sp2) == 0 || sp2[0] < SQ_DIST(v,sp2)) {
                        dm[3*v+0] = 0.5;
                        dm[3*v+1] = 0;
                        dm[3*v+2] = 0;
                    }
                    if (i != 0) {
                        int v2 = vb->index (i-1, j, k);
                        if (dm[3*v2] == FLT_MAX || sp2[0] < SQ_DIST(v2,sp2)) {
                            dm[3*v2+0] = 0.5;
                            dm[3*v2+1] = 0;
                            dm[3*v2+2] = 0;
                        }
                    }
                }
                if (imgb[v] & VBB_MASK_POS_I) {
                    if (SQ_DIST(v,sp2) == 0 || sp2[0] < SQ_DIST(v,sp2)) {
                        dm[3*v+0] = 0.5;
                        dm[3*v+1] = 0;
                        dm[3*v+2] = 0;
                    }
                    if (i != vb->dim[0]-1) {
                        int v2 = vb->index (i+1, j, k);
                        if (dm[3*v2] == FLT_MAX || sp2[0] < SQ_DIST(v2,sp2)) {
                            dm[3*v2+0] = 0.5;
                            dm[3*v2+1] = 0;
                            dm[3*v2+2] = 0;
                        }
                    }
                }
                /* J */
                if (imgb[v] & VBB_MASK_NEG_J) {
                    if (SQ_DIST(v,sp2) == 0 || sp2[1] < SQ_DIST(v,sp2)) {
                        dm[3*v+0] = 0;
                        dm[3*v+1] = 0.5;
                        dm[3*v+2] = 0;
                    }
                    if (j != 0) {
                        int v2 = vb->index (i, j-1, k);
                        if (dm[3*v2] == FLT_MAX || sp2[1] < SQ_DIST(v2,sp2)) {
                            dm[3*v2+0] = 0;
                            dm[3*v2+1] = 0.5;
                            dm[3*v2+2] = 0;
                        }
                    }
                }
                if (imgb[v] & VBB_MASK_POS_J) {
                    if (SQ_DIST(v,sp2) == 0 || sp2[1] < SQ_DIST(v,sp2)) {
                        dm[3*v+0] = 0;
                        dm[3*v+1] = 0.5;
                        dm[3*v+2] = 0;
                    }
                    if (j != vb->dim[1]-1) {
                        int v2 = vb->index (i+1, j, k);
                        if (dm[3*v2] == FLT_MAX || sp2[1] < SQ_DIST(v2,sp2)) {
                            dm[3*v2+0] = 0;
                            dm[3*v2+1] = 0.5;
                            dm[3*v2+2] = 0;
                        }
                    }
                }
                /* K */
                if (imgb[v] & VBB_MASK_NEG_K) {
                    if (SQ_DIST(v,sp2) == 0 || sp2[2] < SQ_DIST(v,sp2)) {
                        dm[3*v+0] = 0;
                        dm[3*v+1] = 0;
                        dm[3*v+2] = 0.5;
                    }
                    if (k != 0) {
                        int v2 = vb->index (i, j, k-1);
                        if (dm[3*v2] == FLT_MAX || sp2[2] < SQ_DIST(v2,sp2)) {
                            dm[3*v2+0] = 0;
                            dm[3*v2+1] = 0;
                            dm[3*v2+2] = 0.5;
                        }
                    }
                }
                if (imgb[v] & VBB_MASK_POS_K) {
                    if (SQ_DIST(v,sp2) == 0 || sp2[2] < SQ_DIST(v,sp2)) {
                        dm[3*v+0] = 0;
                        dm[3*v+1] = 0;
                        dm[3*v+2] = 0.5;
                    }
                    if (k != vb->dim[2]-1) {
                        int v2 = vb->index (i, j, k+1);
                        if (dm[3*v2] == FLT_MAX || sp2[2] < SQ_DIST(v2,sp2)) {
                            dm[3*v2+0] = 0;
                            dm[3*v2+1] = 0;
                            dm[3*v2+2] = 0.5;
                        }
                    }
                }
            }
        }
    }
}

void
Distance_map_private::run_native_danielsson ()
{
    /* Compute boundary of image
       vb = volume of boundary, imgb = img of boundary */
    Image_boundary ib;
    ib.set_volume_boundary_type (vbt);
    ib.set_volume_boundary_behavior (vbb);
    ib.set_input_image (this->input);
    ib.run ();
    UCharImageType::Pointer itk_ib = ib.get_output_image ();
    Plm_image pib (itk_ib);
    Volume::Pointer vb = pib.get_volume_uchar();
    unsigned char *imgb = (unsigned char*) vb->img;
    
    /* Convert image to native volume 
       vs = volume of set, imgs = img of set */
    Plm_image pi (this->input);
    Volume::Pointer vs = pi.get_volume_uchar();
    unsigned char *imgs = (unsigned char*) vs->img;

    /* Sort dimensions by voxel spacing (bubble sort) */
    int spacing_order[3] = { 0, 1, 2 };
    if (vb->spacing[spacing_order[0]] > vb->spacing[spacing_order[1]]) {
        std::swap (spacing_order[0], spacing_order[1]);
    }
    if (vb->spacing[spacing_order[1]] > vb->spacing[spacing_order[2]]) {
        std::swap (spacing_order[1], spacing_order[2]);
    }
    if (vb->spacing[spacing_order[0]] > vb->spacing[spacing_order[1]]) {
        std::swap (spacing_order[0], spacing_order[1]);
    }

    float sp2[3] = {
        vb->spacing[0] * vb->spacing[0],
        vb->spacing[1] * vb->spacing[1],
        vb->spacing[2] * vb->spacing[2]
    };
    
    /* Allocate and initialize "Danielsson array" */
    float *dm = new float[3*vb->npix];
    for (plm_long v = 0; v < vb->npix; v++) {
        bool inside = (bool) imgb[v];
        if (inside) {
            dm[3*v+0] = 0;
            dm[3*v+1] = 0;
            dm[3*v+2] = 0;
        } else {
            dm[3*v+0] = FLT_MAX;
            dm[3*v+1] = FLT_MAX;
            dm[3*v+2] = FLT_MAX;
        }
    }
    if (vbt == INTERIOR_FACE) {
        native_danielsson_initialize_face_distances (vb, dm);
    }

    /* GCS FIX -- I'm not entirely sure if it is required to scan 
       both forward and backward for j direction.  Need to test. */

    /* Propagate within k = 0 */
    this->forward_propagate_j (dm, vb, sp2, 0);
    this->backward_propagate_j (dm, vb, sp2, 0);

    /* Forward scan k */
    for (plm_long k = 1; k < vb->dim[2]; k++) {
        /* Propagate from prev to curr k */
        for (plm_long j = 0; j < vb->dim[1]; j++) {
            for (plm_long i = 0; i < vb->dim[0]; i++) {
                plm_long vo = vb->index (i, j, k-1);   /* "old" voxel */
                plm_long vn = vb->index (i, j, k);     /* "new" voxel */
                if (dm[3*vo] == FLT_MAX) {
                    continue;
                }
                if (dm[3*vn] == FLT_MAX) {
                    COPY_K (vn, vo);
                    continue;
                }
                float odist = SQ_DIST_K(vo,sp2);
                float ndist = SQ_DIST(vn,sp2);
                if (odist < ndist) {
                    COPY_K (vn, vo);
                }
            }
        }
        /* Propagate within curr k */
        this->forward_propagate_j (dm, vb, sp2, k);
        this->backward_propagate_j (dm, vb, sp2, k);
    }

    /* Backward scan k */
    for (plm_long k = vb->dim[2] - 2; k >= 0; k--) {
        /* Propagate from prev to curr k */
        for (plm_long j = 0; j < vb->dim[1]; j++) {
            for (plm_long i = 0; i < vb->dim[0]; i++) {
                plm_long vo = vb->index (i, j, k+1);   /* "old" voxel */
                plm_long vn = vb->index (i, j, k);     /* "new" voxel */
                if (dm[3*vo] == FLT_MAX) {
                    continue;
                }
                if (dm[3*vn] == FLT_MAX) {
                    COPY_K (vn, vo);
                    continue;
                }
                float odist = SQ_DIST_K(vo,sp2);
                float ndist = SQ_DIST(vn,sp2);
                if (odist < ndist) {
                    COPY_K (vn, vo);
                }
            }
        }
        /* Propagate within curr k */
        this->forward_propagate_j (dm, vb, sp2, k);
        this->backward_propagate_j (dm, vb, sp2, k);
    }

    /* Fill in output image */
    Plm_image::Pointer dmap = Plm_image::New (
        new Plm_image (
            new Volume (Volume_header (vb), PT_FLOAT, 1)));
    Volume::Pointer dmap_vol = dmap->get_volume_float ();
    float *dmap_img = (float*) dmap_vol->img;
    for (plm_long v = 0; v < vb->npix; v++) {
        if (!this->use_squared_distance) {
            dmap_img[v] = sqrt(SQ_DIST(v,sp2));
        }
        if (dmap_img[v] >= maximum_distance) {
            dmap_img[v] = maximum_distance;
        }
        if ((this->inside_is_positive && !imgs[v])
            || (!this->inside_is_positive && imgs[v]))
        {
            dmap_img[v] = -dmap_img[v];
        }
        if (this->absolute_distance) {
            dmap_img[v] = fabs(dmap_img[v]);
        }
    }
    
    /* Free temporary memory */
    delete[] dm;

    /* Fixate distance map into private class */
    this->output = dmap->itk_float ();
}

void
Distance_map_private::run_native_maurer ()
{
#if CUDA_FOUND
    if (threading == THREADING_CUDA) {
        distance_map_cuda (0);
        return;
    }
#endif
}
void
Distance_map_private::run_itk_signed_danielsson ()
{
    this->output = itk_distance_map_danielsson (
        this->input,
        this->use_squared_distance,
        this->inside_is_positive);
}
void 
Distance_map_private::runVoronoiFT1D(unsigned char *vol, float *sp2,
		int height, int width, int depth,
		float *output)
{
	// GNodes
	std::deque<std::vector<int>> g_nodes;

	// Distance between slices
	int slice_stride = height * width;

	int k;

	#pragma omp parallel shared(vol, sp2, output) private(g_nodes,k)
	{
	#pragma omp for
	for (k = 0; k < depth; k++){
		int i;
		for (i = 0; i < height; i++){
			int j;
			for (j = 0; j < width; j++){
				if (vol[k * slice_stride + i * width + j] != 0){
					std::vector<int> fv {i, j, k};
					g_nodes.push_back(fv);
				}
			}

			if (g_nodes.size() == 0){
				continue;
			}

			// Query partial voronoi diagram
			for (j = 0; j < width; j++){
				int ite = 0;
				while ((ite < (g_nodes.size() - 1)) &&
					(ED(sp2, i, j, k, g_nodes[ite]) >
					ED(sp2, i, j, k, g_nodes[ite+1]))){
					ite++;
				}

				output[k * slice_stride + i * width + j] =
					double(g_nodes[ite][2] * slice_stride +
					g_nodes[ite][0] * width +
					g_nodes[ite][1]);
			}
			g_nodes.clear();
		}
	}
	}
}

void 
Distance_map_private::runVoronoiFT2D(float *sp2, 
		int height, int width, int depth, 
		float *vol)
{
	std::deque<std::vector<int>> g_nodes;
	
	int Rd[3];
	int w[3];

	// Distance between slices
	int slice_stride = height * width;

	int k;
	#pragma omp parallel shared(sp2, vol) private(g_nodes, Rd, w, k)
    	{
	#pragma omp for
	for (k = 0; k < depth; k++){
		int j;
		for (j = 0; j < width; j++){
			int i;
			for (i = 0; i < height; i++){
				if (vol[k * slice_stride + i * width + j] != -1.0){
					int fv_k = int(vol[k * slice_stride + i * width + j])
							/ slice_stride;
					
					int fv_i = (int(vol[k * slice_stride + i * width + j])
							% slice_stride) / width;
					
					int fv_j = (int(vol[k * slice_stride + i * width + j])
							% slice_stride) % width;
					
					if(g_nodes.size() < 2){
						std::vector<int> fv {fv_i, fv_j, fv_k};

						g_nodes.push_back(fv);
					}
					else{
						w[0] = fv_i;
						w[1] = fv_j;
						w[2] = fv_k;

						Rd[0] = i;
						Rd[1] = j;
						Rd[2] = k;

						while (g_nodes.size() >= 2 && 
							removeFT2D(sp2, g_nodes, w, Rd)){
							g_nodes.pop_back();
						}

						std::vector<int> fv {fv_i, fv_j, fv_k};

						g_nodes.push_back(fv);

					}
				}
			}

			if (g_nodes.size() == 0){
				continue;
			}

			// Query partial voronoi diagram
			for (i = 0; i < height; i++){
				double minDist = DBL_MAX;
				int minIndex = -1;

				int ite = 0;
				while(ite < g_nodes.size()){
					double tempDist = ED(sp2, i, j, k, g_nodes[ite]);

					if(tempDist < minDist){
						minDist = tempDist;
						minIndex = ite;
					}
					ite++;
				}

				vol[k * slice_stride + i * width + j] = \
					double(g_nodes[minIndex][2] * slice_stride +
					g_nodes[minIndex][0] * width +
					g_nodes[minIndex][1]);
			}
			g_nodes.clear();
		}
	}
	}
}

void 
Distance_map_private::runVoronoiFT3D(float *sp2, int height, int width, int depth, float *vol)
{
	std::deque<std::vector<int>> g_nodes;

	int Rd[3];
	int w[3];

	// Distance between slices
	int slice_stride = height * width;

	int i;
	#pragma omp parallel shared(sp2, vol) private(g_nodes, Rd, w, i)
    	{
	#pragma omp for
	for (i = 0; i < height; i++){
		int j;
		for (j = 0; j < width; j++){
			int k;
			for (k = 0; k < depth; k++){
				if (vol[k * slice_stride + i * width + j] != -1.0){
					int fv_k = int(vol[k * slice_stride + i * width + j])
							/ slice_stride;

					int fv_i = (int(vol[k * slice_stride + i * width + j])
							% slice_stride) / width;

					int fv_j = (int(vol[k * slice_stride + i * width + j])
							% slice_stride) % width;
					
					if(g_nodes.size() < 2){
						std::vector<int> fv {fv_i, fv_j, fv_k};

						g_nodes.push_back(fv);
					}
					else{
						w[0] = fv_i;
						w[1] = fv_j;
						w[2] = fv_k;

						Rd[0] = i;
						Rd[1] = j;
						Rd[2] = k;

						while (g_nodes.size() >= 2 && 
							removeFT3D(sp2, g_nodes, w, Rd)){
							g_nodes.pop_back();
						}

						std::vector<int> fv {fv_i, fv_j, fv_k};

						g_nodes.push_back(fv);

					}
				}
			}
				
			if (g_nodes.size() == 0){
				continue;
			}

			// Query partial voronoi diagram
			for (k = 0; k < depth; k++){
				double minDist = DBL_MAX;
				int minIndex = -1;

				int ite = 0;
				while(ite < g_nodes.size()){
					double tempDist = ED(sp2, i, j, k, g_nodes[ite]);

					if(tempDist < minDist){
						minDist = tempDist;
						minIndex = ite;
					}
					ite++;
				}

				vol[k * slice_stride + i * width + j] = \
					double(g_nodes[minIndex][2] * slice_stride + \
					g_nodes[minIndex][0] * width + \
					g_nodes[minIndex][1]);
			}
			g_nodes.clear();
		}
	}
	}
}

int 
Distance_map_private::removeFT2D(float *sp2, std::deque<std::vector<int>> &g_nodes, int *w, int *Rd)
{
	std::vector<int> u = g_nodes[g_nodes.size() - 2];
	std::vector<int> v = g_nodes[g_nodes.size() - 1];

	double a = (v[0] - u[0]) * sqrt(sp2[0]);
	double b = (w[0] - v[0]) * sqrt(sp2[0]);
	double c = a + b;

	double vRd = 0.0;
	double uRd = 0.0;
	double wRd = 0.0;

	int i = 1;
	for (i; i < 3; i++){
		vRd += (v[i] - Rd[i]) * (v[i] - Rd[i]) * sp2[i];
		uRd += (u[i] - Rd[i]) * (u[i] - Rd[i]) * sp2[i];
		wRd += (w[i] - Rd[i]) * (w[i] - Rd[i]) * sp2[i];
	}

	return (c * vRd - b * uRd - a * wRd - a * b * c > 0.0);

}

int 
Distance_map_private::removeFT3D(float *sp2, std::deque<std::vector<int>> &g_nodes, int *w, int *Rd)
{
	std::vector<int> u = g_nodes[g_nodes.size() - 2];
	std::vector<int> v = g_nodes[g_nodes.size() - 1];

	double a = (v[2] - u[2]) * sqrt(sp2[2]);
	double b = (w[2] - v[2]) * sqrt(sp2[2]);
	double c = a + b;

	double vRd = 0;
	double uRd = 0;
	double wRd = 0;

	int i = 0;
	for (i; i < 2; i++){
		vRd += (v[i] - Rd[i]) * (v[i] - Rd[i]) * sp2[i];
		uRd += (u[i] - Rd[i]) * (u[i] - Rd[i]) * sp2[i];
		wRd += (w[i] - Rd[i]) * (w[i] - Rd[i]) * sp2[i];
	}

	return (c * vRd - b * uRd - a * wRd - a * b * c > 0.0);
}

double 
Distance_map_private::ED(float *sp2, 
		int vol_i, int vol_j, int vol_k, 
		std::vector<int> &fv)
{
	double temp = 0;

        temp = (fv[0] - vol_i) * (fv[0] - vol_i) * sp2[0] +\
                        (fv[1] - vol_j) * (fv[1] - vol_j) * sp2[1] +\
                        (fv[2] - vol_k) * (fv[2] - vol_k) * sp2[2];

        return sqrt(temp);
}
void 
Distance_map_private::voronoiFT(int dim, unsigned char *vol, float *sp2,
		int height, int width, int depth, float *output)
{
	switch (dim)
	{
		case 1:
			runVoronoiFT1D(vol, sp2, height, width, depth, output);
			break;	
		case 2:				
			runVoronoiFT2D(sp2, height, width, depth, output);
			break;
		case 3:					
			runVoronoiFT3D(sp2, height, width, depth, output);
			break;									
		default:								
			break;								
	}
}
void
Distance_map_private::maurerFT(unsigned char *vol, float *sp2, int height,
		int width, int depth, float *output)
{
	int dim;
	for (dim = 1; dim < 4; dim++){
	       voronoiFT(dim, vol, sp2, height, width, depth, output);
	}
}
void 
Distance_map_private::distTransform(unsigned char *vol, float *sp2,
		int height, int width, int depth, float *ed_out)
{
	int slice_stride = height * width;

	int k;
	#pragma omp parallel private(k)
	{
	
	#pragma omp for
	for (k = 0; k < depth; k++){
		int i;
		for (i = 0; i < height; i++){
			int j;
			for (j = 0; j < width; j++){
				int dep_id = int(ed_out[k * slice_stride + i * width + j])
					    	 / slice_stride;
				
				int row_id = int(ed_out[k * slice_stride + i * width + j])
						% slice_stride / width;

				int col_id = int(ed_out[k * slice_stride + i * width + j])
					     	% slice_stride % width;
				
				if (row_id == i && col_id == j && k == dep_id){
					ed_out[k * slice_stride + i * width + j] = 0.0;
				}
				else{
                                        ed_out[k * slice_stride + i * width + j] =
						calcDist(
						sp2,
						i, j, k,
						row_id, col_id, dep_id); 
				}
			}
		}
	}
	}
}

double 
Distance_map_private::calcDist(float *sp2,
		double i, double j, double k,
		double target_i, double target_j, double target_k)
{
	double result = (i - target_i) * (i - target_i) * sp2[0] +
			(j - target_j) * (j - target_j) * sp2[1] +
			(k - target_k) * (k - target_k) * sp2[2];

	return sqrt(result);
}
void
Distance_map_private::run_song_maurer ()
{
    /* Compute boundary of image
       vb = volume of boundary, imgb = img of boundary */
    Image_boundary ib;
    ib.set_volume_boundary_type (vbt);
    ib.set_volume_boundary_behavior (vbb);
    ib.set_input_image (this->input);
    ib.run ();
    UCharImageType::Pointer itk_ib = ib.get_output_image ();
    Plm_image pib (itk_ib);
    Volume::Pointer vb = pib.get_volume_uchar();
    unsigned char *imgb = (unsigned char*) vb->img;
    
    /* Convert image to native volume 
       vs = volume of set, imgs = img of set */
    Plm_image pi (this->input);
    Volume::Pointer vs = pi.get_volume_uchar();
    unsigned char *imgs = (unsigned char*) vs->img;

    float sp2[3] = {
        vb->spacing[0] * vb->spacing[0],
        vb->spacing[1] * vb->spacing[1],
        vb->spacing[2] * vb->spacing[2]
    };
    /* Fill in output image */
    Plm_image::Pointer dmap = Plm_image::New (
        new Plm_image (
            new Volume (Volume_header (vb), PT_FLOAT, 1)));
    Volume::Pointer dmap_vol = dmap->get_volume_float ();
    float *dmap_img = (float*) dmap_vol->img;
    //float *dmap_img = (float *)malloc(vb->spacing[0] * vb->spacing[1] *
//		    vb->spacing[2] * sizeof(float));
    for (int i = 0; i < vb->dim[0] * vb->dim[1] *
		    vb->dim[2]; i++){
	    dmap_img[i] = -1.0;
    }

    maurerFT(imgb, sp2, vb->dim[0], vb->dim[1], vb->dim[2],
		    dmap_img);

    distTransform(imgb, sp2, vb->dim[0], vb->dim[1], vb->dim[2],
		    dmap_img);
    
    /* Fixate distance map into private class */

    this->output = dmap->itk_float();

}

void
Distance_map_private::run_itk_signed_maurer ()
{
    this->output = itk_distance_map_maurer (
        this->input,
        this->use_squared_distance,
        this->inside_is_positive);
}

void
Distance_map_private::run ()
{
    switch (this->algorithm) {
    case Distance_map::DANIELSSON:
        this->run_native_danielsson ();
        break;
    case Distance_map::ITK_DANIELSSON:
        this->run_itk_signed_danielsson ();
        break;
    case Distance_map::MAURER:
        this->run_native_maurer ();
        break;
    case Distance_map::SONG_MAURER:
        this->run_song_maurer ();
        break;	
    case Distance_map::ITK_MAURER:
    default:
        this->run_itk_signed_maurer ();
        break;
    }
}

Distance_map::Distance_map () {
    d_ptr = new Distance_map_private;
}

Distance_map::~Distance_map () {
    delete d_ptr;
}

void
Distance_map::set_input_image (const std::string& image_fn)
{
    Plm_image pli (image_fn);
    d_ptr->input = pli.itk_uchar();
}

void
Distance_map::set_input_image (const char* image_fn)
{
    Plm_image pli (image_fn);
    d_ptr->input = pli.itk_uchar();
}

void
Distance_map::set_input_image (UCharImageType::Pointer image)
{
    d_ptr->input = image;
}

void
Distance_map::set_input_image (const Plm_image::Pointer& image)
{
    Plm_image::Pointer pi_clone = image->clone ();
    d_ptr->input = pi_clone->itk_uchar ();
}

void 
Distance_map::set_use_squared_distance (bool use_squared_distance)
{
    d_ptr->use_squared_distance = use_squared_distance;
}

void 
Distance_map::set_maximum_distance (float maximum_distance)
{
    d_ptr->maximum_distance = maximum_distance;
}

void
Distance_map::set_volume_boundary_behavior (Volume_boundary_behavior vbb)
{
    d_ptr->vbb = vbb;
}

void
Distance_map::set_volume_boundary_type (Volume_boundary_type vbt)
{
    d_ptr->vbt = vbt;
}

void
Distance_map::set_threading (Threading threading)
{
    d_ptr->threading = threading;
}

void 
Distance_map::set_inside_is_positive (bool inside_is_positive)
{
    d_ptr->inside_is_positive = inside_is_positive;
}

void 
Distance_map::set_absolute_distance (bool absolute_distance)
{
    d_ptr->absolute_distance = absolute_distance;
}

void 
Distance_map::set_algorithm (const std::string& algorithm)
{
    if (algorithm == "danielsson" || algorithm == "native_danielsson") {
        d_ptr->algorithm = Distance_map::DANIELSSON;
    }
    else if (algorithm == "itk-danielsson") {
        d_ptr->algorithm = Distance_map::ITK_DANIELSSON;
    }
    else if (algorithm == "maurer") {
        d_ptr->algorithm = Distance_map::ITK_MAURER;
    }
    else if (algorithm == "native_maurer") {
        d_ptr->algorithm = Distance_map::MAURER;
    }
    else if (algorithm == "itk-maurer" || algorithm == "itk_maurer") {
        d_ptr->algorithm = Distance_map::ITK_MAURER;
    }
    else if (algorithm == "song-maurer" || algorithm == "song_maurer") {
        d_ptr->algorithm = Distance_map::SONG_MAURER;
    }
    /* Else do nothing */
}

void
Distance_map::run ()
{
    d_ptr->run ();
}

FloatImageType::Pointer
Distance_map::get_output_image ()
{
    return d_ptr->output;
}
