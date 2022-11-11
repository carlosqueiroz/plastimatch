/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#if (OPENMP_FOUND)
#include <omp.h>
#endif

#include "bspline.h"
#include "bspline_regularize.h"
#include "bspline_regularize_analytic.h"
#include "bspline_score.h"
#include "bspline_xform.h"
#include "plm_math.h"
#include "plm_timer.h"
#include "volume.h"

void
print_matrix (double* mat, int m, int n)
{
    int i,j;

    for (j=0; j<n; j++) {
        for (i=0; i<m; i++) {
            printf ("%1.3e ", mat[m*j+i]);
        }
        printf ("\n");
    }
}

void
compute_coeff_from_vf (Bspline_xform* bxf, Volume* vol)
{
    plm_long i, j, k;
    int a,b,c,z;
    int idx_poi, cidx, pidx, qidx;
    float *vec_poi;
    float *img = (float*) vol->img;

    plm_long p[3];
    float q[3];
    float* q_lut;
    plm_long* c_lut;

    for (k = 0; k < vol->dim[2]; k++) {
        p[2] = k / bxf->vox_per_rgn[2];
        q[2] = k % bxf->vox_per_rgn[2];
        for (j = 0; j < vol->dim[1]; j++) {
            p[1] = j / bxf->vox_per_rgn[1];
            q[1] = j % bxf->vox_per_rgn[1];
            for (i = 0; i < vol->dim[0]; i++) {
                p[0] = i / bxf->vox_per_rgn[0];
                q[0] = i % bxf->vox_per_rgn[0];

                pidx = volume_index (bxf->rdims, p[0], p[1], p[2]);
                qidx = volume_index (bxf->vox_per_rgn, q[0], q[1], q[2]);

                idx_poi = volume_index (vol->dim, i, j, k);
                vec_poi = &img[3*idx_poi];

                q_lut = &bxf->q_lut[qidx*64];
                c_lut = &bxf->c_lut[pidx*64];

                z = 0;
                for (c = 0; c < 4; c++) {
                    for (b = 0; b < 4; b++) {
                        for (a = 0; a < 4; a++) {
                            cidx = 3 * c_lut[z];
                            bxf->coeff[cidx+0] += vec_poi[0] * q_lut[z];
                            bxf->coeff[cidx+1] += vec_poi[1] * q_lut[z];
                            bxf->coeff[cidx+2] += vec_poi[2] * q_lut[z];
                            z++;
                        }
                    }
                }

            } /* i < vol-dim[0] */
        } /* j < vol->dim[1] */
    } /* k < vol->dim[2] */
}

#if (OPENMP_FOUND)
void
reg_sort_sets (
    double* cond,
    double* sets,
    plm_long* k_lut,
    const Bspline_xform* bxf
)
{
    int sidx, kidx;

    /* Rackem' Up */
    for (sidx=0; sidx<64; sidx++) {
        kidx = k_lut[sidx];

        cond[3*(64*kidx+sidx)+0] = sets[3*sidx+0];
        cond[3*(64*kidx+sidx)+1] = sets[3*sidx+1];
        cond[3*(64*kidx+sidx)+2] = sets[3*sidx+2];
    }
}
#endif


#if (OPENMP_FOUND)
void
reg_update_grad (
    Bspline_score* ssd,
    double* cond,
    const Bspline_xform* bxf
)
{
    int kidx, sidx;

    for (kidx=0; kidx < bxf->num_knots; kidx++) {
        for (sidx=0; sidx<64; sidx++) {
            ssd->total_grad[3*kidx+0] += cond[3*(64*kidx+sidx)+0];
            ssd->total_grad[3*kidx+1] += cond[3*(64*kidx+sidx)+1];
            ssd->total_grad[3*kidx+2] += cond[3*(64*kidx+sidx)+2];
        }
    }
}
#endif


void 
find_knots_3 (plm_long* knots, plm_long tile_num, const plm_long* cdims)
{
    int tile_loc[3];
    int i, j, k;
    int idx = 0;
    int num_tiles_x = cdims[0] - 3;
    int num_tiles_y = cdims[1] - 3;
    int num_tiles_z = cdims[2] - 3;

    // First get the [x,y,z] coordinate of
    // the tile in the control grid.
    tile_loc[0] = tile_num % num_tiles_x;
    tile_loc[1] = ((tile_num - tile_loc[0]) / num_tiles_x) % num_tiles_y;
    tile_loc[2] = ((((tile_num - tile_loc[0]) / num_tiles_x) / 
			    num_tiles_y) % num_tiles_z);

    /* GCS 2011-07-14: Why not remove the below three lines, and let i,j,k 
       run from 0 to 3? */
    // Tiles do not start on the edges of the grid, so we
    // push them to the center of the control grid.
    tile_loc[0]++;
    tile_loc[1]++;
    tile_loc[2]++;

    // Find 64 knots' [x,y,z] coordinates
    // and convert into a linear knot index
    for (k = -1; k < 3; k++)
    	for (j = -1; j < 3; j++)
    	    for (i = -1; i < 3; i++) {
    		knots[idx++] = (cdims[0]*cdims[1]*(tile_loc[2]+k)) +
                           (cdims[0]*(tile_loc[1]+j)) +
                           (tile_loc[0]+i);
        }

}

void
eval_integral (double* V, double* Qn1, double* Qn2, double gs)
{
    int i,j;
    double S[16];

    double I[7] = {
        gs,
        (1.0/2.0) * (gs * gs),
        (1.0/3.0) * (gs * gs * gs),
        (1.0/4.0) * (gs * gs * gs * gs),
        (1.0/5.0) * (gs * gs * gs * gs * gs),
        (1.0/6.0) * (gs * gs * gs * gs * gs * gs),
        (1.0/7.0) * (gs * gs * gs * gs * gs * gs * gs)
    };

    // Generate 16 4x4 matrix by taking the outer
    // product of the each row in the Q matrix with
    // every other row in the Q matrix. We use these
    // to generate each element in V.
    for (j=0; j<4; j++) {
        for (i=0; i<4; i++) {
            vec_outer (S, Qn1+(4*j), Qn2+(4*i), 4);
            V[4*j + i] = (I[0] *  S[ 0])
                       + (I[1] * (S[ 1] + S[ 4]))
                       + (I[2] * (S[ 2] + S[ 5] + S[ 8]))
                       + (I[3] * (S[ 3] + S[ 6] + S[ 9] + S[12]))
                       + (I[4] * (S[ 7] + S[10] + S[13]))
                       + (I[5] * (S[11] + S[14]))
                       + (I[6] * (S[15]));
        }
    }
}

void
init_analytic (double **QX, double **QY, double **QZ, 
    const Bspline_xform* bxf)
{
    double rx, ry, rz;

    double B[16] = {
        1.0/6.0, -1.0/2.0,  1.0/2.0, -1.0/6.0,
        2.0/3.0,  0.0    , -1.0    ,  1.0/2.0,
        1.0/6.0,  1.0/2.0,  1.0/2.0, -1.0/2.0,
        0.0    ,  0.0    ,  0.0    ,  1.0/6.0
    };

    /* grid spacing */
    rx = 1.0/bxf->grid_spac[0];
    ry = 1.0/bxf->grid_spac[1];
    rz = 1.0/bxf->grid_spac[2];

    double RX[16] = {
        1.0, 0.0,   0.0,      0.0,
        0.0,  rx,   0.0,      0.0,
        0.0, 0.0, rx*rx,      0.0,
        0.0, 0.0,   0.0, rx*rx*rx
    };

    double RY[16] = {
        1.0, 0.0,   0.0,      0.0,
        0.0,  ry,   0.0,      0.0,
        0.0, 0.0, ry*ry,      0.0,
        0.0, 0.0,   0.0, ry*ry*ry
    };

    double RZ[16] = {
        1.0, 0.0,   0.0,      0.0,
        0.0,  rz,   0.0,      0.0,
        0.0, 0.0, rz*rz,      0.0,
        0.0, 0.0,   0.0, rz*rz*rz
    };

    double delta1[16] = {
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 3.0, 0.0
    };

    double delta2[16] = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        2.0, 0.0, 0.0, 0.0,
        0.0, 6.0, 0.0, 0.0
    };
    
    double delta3[16] = {
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        6.0, 0.0, 0.0, 0.0
    };

    // Let's call Q the product of the recripocal grid spacing
    // matrix (R) and the B-spline coefficient matrix (B).
    mat_mult_mat (QX[0], B, 4, 4, RX, 4, 4);
    mat_mult_mat (QY[0], B, 4, 4, RY, 4, 4);
    mat_mult_mat (QZ[0], B, 4, 4, RZ, 4, 4);

    // Get the product of QX, QY, QZ and delta.
    //   QX1 is the  first-order derivative of X
    //   QX2 is the second-order derivative of X
    //   QY1 is the  first-order derivative of Y
    //   ... etc
    mat_mult_mat (QX[1], QX[0], 4, 4, delta1, 4, 4);    
    mat_mult_mat (QX[2], QX[0], 4, 4, delta2, 4, 4);
    mat_mult_mat (QX[3], QX[0], 4, 4, delta3, 4, 4);
    mat_mult_mat (QY[1], QY[0], 4, 4, delta1, 4, 4);    
    mat_mult_mat (QY[2], QY[0], 4, 4, delta2, 4, 4);    
    mat_mult_mat (QY[3], QY[0], 4, 4, delta3, 4, 4);
    mat_mult_mat (QZ[1], QZ[0], 4, 4, delta1, 4, 4);    
    mat_mult_mat (QZ[2], QZ[0], 4, 4, delta2, 4, 4);    
    mat_mult_mat (QZ[3], QZ[0], 4, 4, delta3, 4, 4);
}

void
get_Vmatrix (double* V, double* X, double* Y, double* Z)
{
    int i,j;
    double tmp[256];       /* 16 x 16 matrix */

    /* Calculate the temporary 16*16 matrix */
    for (j=0; j<4; j++) {
        for (i=0; i<4; i++) {
            tmp[16*(j+ 0) + (i+ 0)] = Y[4*0 + 0] * Z[4*j + i];
            tmp[16*(j+ 0) + (i+ 4)] = Y[4*0 + 1] * Z[4*j + i];
            tmp[16*(j+ 0) + (i+ 8)] = Y[4*0 + 2] * Z[4*j + i];
            tmp[16*(j+ 0) + (i+12)] = Y[4*0 + 3] * Z[4*j + i];

            tmp[16*(j+ 4) + (i+ 0)] = Y[4*1 + 0] * Z[4*j + i];
            tmp[16*(j+ 4) + (i+ 4)] = Y[4*1 + 1] * Z[4*j + i];
            tmp[16*(j+ 4) + (i+ 8)] = Y[4*1 + 2] * Z[4*j + i];
            tmp[16*(j+ 4) + (i+12)] = Y[4*1 + 3] * Z[4*j + i];

            tmp[16*(j+ 8) + (i+ 0)] = Y[4*2 + 0] * Z[4*j + i];
            tmp[16*(j+ 8) + (i+ 4)] = Y[4*2 + 1] * Z[4*j + i];
            tmp[16*(j+ 8) + (i+ 8)] = Y[4*2 + 2] * Z[4*j + i];
            tmp[16*(j+ 8) + (i+12)] = Y[4*2 + 3] * Z[4*j + i];

            tmp[16*(j+12) + (i+ 0)] = Y[4*3 + 0] * Z[4*j + i];
            tmp[16*(j+12) + (i+ 4)] = Y[4*3 + 1] * Z[4*j + i];
            tmp[16*(j+12) + (i+ 8)] = Y[4*3 + 2] * Z[4*j + i];
            tmp[16*(j+12) + (i+12)] = Y[4*3 + 3] * Z[4*j + i];
        }
    }

    /* Calculate the 64*64 V matrix */
    for (j=0; j<16; j++) {
        for (i=0; i<16; i++) {
            V[64*(j+ 0) + (i+ 0)] = X[4*0 + 0] * tmp[16*j + i];
            V[64*(j+ 0) + (i+16)] = X[4*0 + 1] * tmp[16*j + i];
            V[64*(j+ 0) + (i+32)] = X[4*0 + 2] * tmp[16*j + i];
            V[64*(j+ 0) + (i+48)] = X[4*0 + 3] * tmp[16*j + i];

            V[64*(j+16) + (i+ 0)] = X[4*1 + 0] * tmp[16*j + i];
            V[64*(j+16) + (i+16)] = X[4*1 + 1] * tmp[16*j + i];
            V[64*(j+16) + (i+32)] = X[4*1 + 2] * tmp[16*j + i];
            V[64*(j+16) + (i+48)] = X[4*1 + 3] * tmp[16*j + i];

            V[64*(j+32) + (i+ 0)] = X[4*2 + 0] * tmp[16*j + i];
            V[64*(j+32) + (i+16)] = X[4*2 + 1] * tmp[16*j + i];
            V[64*(j+32) + (i+32)] = X[4*2 + 2] * tmp[16*j + i];
            V[64*(j+32) + (i+48)] = X[4*2 + 3] * tmp[16*j + i];

            V[64*(j+48) + (i+ 0)] = X[4*3 + 0] * tmp[16*j + i];
            V[64*(j+48) + (i+16)] = X[4*3 + 1] * tmp[16*j + i];
            V[64*(j+48) + (i+32)] = X[4*3 + 2] * tmp[16*j + i];
            V[64*(j+48) + (i+48)] = X[4*3 + 3] * tmp[16*j + i];
        }
    }
}

void
scale_Vmatrix(double* V, double lambda)
{
	int j;
	for (j=0;j<4096;j++) {
		V[j] = lambda * V[j];
	}
}
void
scale_Vmatrix_omp(double* V, double lambda)
{
	int j;
#pragma omp parallel for 
	for (j=0;j<4096;j++) {
		V[j] = lambda * V[j];
	}
}

/* Employs my "world famous" thread safe "condense" control-point
 * update method for mulit-core acceleration.  Also, score is
 * returned so as to take advantage of OpenMP's built in
 * sum-reduction capabilities */
#if (OPENMP_FOUND)
double
region_smoothness_omp (
    double* sets,
    const Regularization_parms* reg_parms,    
    const Bspline_xform* bxf,
    double* V, 
    plm_long* knots
)
{
    double S = 0.0;         /* Region smoothness */
    double X[64] = {0};
    double Y[64] = {0};
    double Z[64] = {0};
    int i, j;

    for (j=0; j<64; j++) {
    	/* S = pVp operation ----------------------------- */
        for (i=0; i<64; i++) {
            X[j] += bxf->coeff[3*knots[i]+0] * V[64*j + i];
            Y[j] += bxf->coeff[3*knots[i]+1] * V[64*j + i];
            Z[j] += bxf->coeff[3*knots[i]+2] * V[64*j + i];
        }

        S += X[j] * bxf->coeff[3*knots[j]+0];
        S += Y[j] * bxf->coeff[3*knots[j]+1];
        S += Z[j] * bxf->coeff[3*knots[j]+2];
        /* ------------------------------------------------ */

        /* dS/dp = 2Vp operation */
        sets[3*j+0] += 2 * X[j];
        sets[3*j+1] += 2 * Y[j];
        sets[3*j+2] += 2 * Z[j];
    }

    return S;
}
#endif

void
region_smoothness_elastic (
    Bspline_score *bspline_score, 
    const Regularization_parms* reg_parms,    
    const Bspline_xform* bxf,
    double* V9,
    double* V10,
    double* V11,
    double* V12,
    double* V13,
    double* V14,
    double* V15,
    double* V16,
    double* V17,
    double* V18,
    double* V19,
    double* V20,	    
    plm_long* knots)
{
    double S = 0.0;         /* Region smoothness */
    double X1[64] = {0};
    double Y1[64] = {0};
    double X2[64] = {0};
    double Y2[64] = {0};
    double X3[64] = {0};
    double Z3[64] = {0};
    double X4[64] = {0};
    double Z4[64] = {0};
    double Y5[64] = {0};
    double Z5[64] = {0};
    double Y6[64] = {0};
    double Z6[64] = {0};
    double X7[64] = {0};
    double Y7[64] = {0};
    double Z7[64] = {0};
    double X8[64] = {0};
    double Y8[64] = {0};
    double Z8[64] = {0};
    double X9[64] = {0};
    double Y9[64] = {0};
    double Z9[64] = {0};
    
     int i,j;

    for (j=0; j<64; j++) {
    	/* S = pVp operation ----------------------------- */
        for (i=0; i<64; i++) {
            X1[j] += bxf->coeff[3*knots[i]+0] * V9[64*j + i];
            X2[j] += bxf->coeff[3*knots[i]+0] * V10[64*j + i];
            X3[j] += bxf->coeff[3*knots[i]+0] * V11[64*j + i];
            X4[j] += bxf->coeff[3*knots[i]+0] * V12[64*j + i];
            Y5[j] += bxf->coeff[3*knots[i]+1] * V13[64*j + i];
            Y6[j] += bxf->coeff[3*knots[i]+1] * V14[64*j + i];
            Y1[j] += bxf->coeff[3*knots[i]+1] * V9[64*j + i];
            Y2[j] += bxf->coeff[3*knots[i]+1] * V10[64*j + i];
            Z3[j] += bxf->coeff[3*knots[i]+2] * V11[64*j + i];
            Z4[j] += bxf->coeff[3*knots[i]+2] * V12[64*j + i];
            Z5[j] += bxf->coeff[3*knots[i]+2] * V13[64*j + i];
            Z6[j] += bxf->coeff[3*knots[i]+2] * V14[64*j + i];
	    X7[j] += bxf->coeff[3*knots[i]+0] * V15[64*j + i];
	    Y8[j] += bxf->coeff[3*knots[i]+1] * V16[64*j + i];
	    Z9[j] += bxf->coeff[3*knots[i]+2] * V17[64*j + i];
	    Y7[j] += bxf->coeff[3*knots[i]+1] * V18[64*j + i];
	    Z7[j] += bxf->coeff[3*knots[i]+2] * V18[64*j + i];
	    X8[j] += bxf->coeff[3*knots[i]+0] * V19[64*j + i];
            Z8[j] += bxf->coeff[3*knots[i]+2] * V19[64*j + i];
            X9[j] += bxf->coeff[3*knots[i]+0] * V20[64*j + i];
	    Y9[j] += bxf->coeff[3*knots[i]+1] * V20[64*j + i];
	}

        S += X1[j] * bxf->coeff[3*knots[j]+1];
        S += X2[j] * bxf->coeff[3*knots[j]+1];
        S += X3[j] * bxf->coeff[3*knots[j]+2];
	S += X4[j] * bxf->coeff[3*knots[j]+2];
        S += Y5[j] * bxf->coeff[3*knots[j]+2];
        S += Y6[j] * bxf->coeff[3*knots[j]+2];
	S += X7[j] * bxf->coeff[3*knots[j]+0];
	S += Y8[j] * bxf->coeff[3*knots[j]+1];
	S += Z9[j] * bxf->coeff[3*knots[j]+2];
        S += Y7[j] * bxf->coeff[3*knots[j]+1];
    	S += Z7[j] * bxf->coeff[3*knots[j]+2];
    	S += X8[j] * bxf->coeff[3*knots[j]+0];
    	S += Z8[j] * bxf->coeff[3*knots[j]+2];
    	S += X9[j] * bxf->coeff[3*knots[j]+0];
    	S += Y9[j] * bxf->coeff[3*knots[j]+1];

	/* ------------------------------------------------ */

        /* dS/dp = Vp operation */
	bspline_score->total_grad[3*knots[j]+0] += Y1[j] + Y2[j] + Z3[j] + 
		Z4[j] + 2 * X7[j] + 2 * X8[j] + 2 * X9[j];
        bspline_score->total_grad[3*knots[j]+1] += X1[j] + X2[j] + Z5[j] + 
		Z6[j] + 2 * Y7[j] + 2 * Y8[j] + 2 * Y9[j];
        bspline_score->total_grad[3*knots[j]+2] += X3[j] + X4[j] + Y5[j] + 
		Y6[j] + 2 * Z7[j] + 2 * Z8[j] + 2 * Z9[j];
    }

    bspline_score->rmetric += S;
}
#if (OPENMP_FOUND)
double
region_smoothness_elastic_omp (
    double* sets,
    const Regularization_parms* reg_parms,    
    const Bspline_xform* bxf,
    double* V9,
    double* V10,
    double* V11,
    double* V12,
    double* V13,
    double* V14,
    double* V15,
    double* V16,
    double* V17,
    double* V18,
    double* V19,
    double* V20,

    plm_long* knots
)
{
    double S = 0.0;         /* Region smoothness */
    double X1[64] = {0};
    double Y1[64] = {0};
    double X2[64] = {0};
    double Y2[64] = {0};
    double X3[64] = {0};
    double Z3[64] = {0};
    double X4[64] = {0};
    double Z4[64] = {0};
    double Y5[64] = {0};
    double Z5[64] = {0};
    double Y6[64] = {0};
    double Z6[64] = {0};
    double X7[64] = {0};
    double Y7[64] = {0};
    double Z7[64] = {0};
    double X8[64] = {0};
    double Y8[64] = {0};
    double Z8[64] = {0};
    double X9[64] = {0};
    double Y9[64] = {0};
    double Z9[64] = {0};
    
    int i, j;

    for (j=0; j<64; j++) {
    	/* S = pVp operation ----------------------------- */
        for (i=0; i<64; i++) {
            X1[j] += bxf->coeff[3*knots[i]+0] * V9[64*j + i];
            X2[j] += bxf->coeff[3*knots[i]+0] * V10[64*j + i];
            X3[j] += bxf->coeff[3*knots[i]+0] * V11[64*j + i];
            X4[j] += bxf->coeff[3*knots[i]+0] * V12[64*j + i];
            Y5[j] += bxf->coeff[3*knots[i]+1] * V13[64*j + i];
            Y6[j] += bxf->coeff[3*knots[i]+1] * V14[64*j + i];
            Y1[j] += bxf->coeff[3*knots[i]+1] * V9[64*j + i];
            Y2[j] += bxf->coeff[3*knots[i]+1] * V10[64*j + i];
            Z3[j] += bxf->coeff[3*knots[i]+2] * V11[64*j + i];
            Z4[j] += bxf->coeff[3*knots[i]+2] * V12[64*j + i];
            Z5[j] += bxf->coeff[3*knots[i]+2] * V13[64*j + i];
            Z6[j] += bxf->coeff[3*knots[i]+2] * V14[64*j + i];
            X7[j] += bxf->coeff[3*knots[i]+0] * V15[64*j + i]; //(mu + lambda/2)
	    Y8[j] += bxf->coeff[3*knots[i]+1] * V16[64*j + i];
	    Z9[j] += bxf->coeff[3*knots[i]+2] * V17[64*j + i];
	    Y7[j] += bxf->coeff[3*knots[i]+1] * V18[64*j + i]; //(mu/2)
       	    Z7[j] += bxf->coeff[3*knots[i]+2] * V18[64*j + i];
	    X8[j] += bxf->coeff[3*knots[i]+0] * V19[64*j + i];
	    Z8[j] += bxf->coeff[3*knots[i]+2] * V19[64*j + i];
     	    X9[j] += bxf->coeff[3*knots[i]+0] * V20[64*j + i];
	    Y9[j] += bxf->coeff[3*knots[i]+1] * V20[64*j + i];
        }

        S += X1[j] * bxf->coeff[3*knots[j]+1];
        S += X2[j] * bxf->coeff[3*knots[j]+1];
        S += X3[j] * bxf->coeff[3*knots[j]+2];
	S += X4[j] * bxf->coeff[3*knots[j]+2];
        S += Y5[j] * bxf->coeff[3*knots[j]+2];
        S += Y6[j] * bxf->coeff[3*knots[j]+2];
        S += X7[j] * bxf->coeff[3*knots[j]+0];
	S += Y8[j] * bxf->coeff[3*knots[j]+1];
	S += Z9[j] * bxf->coeff[3*knots[j]+2];
        S += Y7[j] * bxf->coeff[3*knots[j]+1];
        S += Z7[j] * bxf->coeff[3*knots[j]+2];
        S += X8[j] * bxf->coeff[3*knots[j]+0];
        S += Z8[j] * bxf->coeff[3*knots[j]+2];
        S += X9[j] * bxf->coeff[3*knots[j]+0];
        S += Y9[j] * bxf->coeff[3*knots[j]+1];
 
	/* ------------------------------------------------ */

        /* dS/dp = 2Vp operation */
        sets[3*j+0] += Y1[j] + Y2[j] + Z3[j] + Z4[j] + 2 * X7[j] + 2 * X8[j] + 
		2 * X9[j];
        sets[3*j+1] += X1[j] + X2[j] + Z5[j] + Z6[j] + 2 * Y8[j] + 2 * Y7[j] + 
		2 * Y9[j];
        sets[3*j+2] += X3[j] + X4[j] + Y5[j] + Y6[j] + 2 * Z9[j] + 2 * Z7[j] + 
		2 * Z8[j];
    }
    return S;
}
#endif

void
region_smoothness (
    Bspline_score *bspline_score, 
    const Regularization_parms* reg_parms,    
    const Bspline_xform* bxf,
    double* V, 
    plm_long* knots)
{
    double S = 0.0;         /* Region smoothness */
    double X[64] = {0};
    double Y[64] = {0};
    double Z[64] = {0};
    int i,j;

    for (j=0; j<64; j++) {
    	/* S = pVp operation ----------------------------- */
        for (i=0; i<64; i++) {
            X[j] += bxf->coeff[3*knots[i]+0] * V[64*j + i];
            Y[j] += bxf->coeff[3*knots[i]+1] * V[64*j + i];
            Z[j] += bxf->coeff[3*knots[i]+2] * V[64*j + i];
        }

        S += X[j] * bxf->coeff[3*knots[j]+0];
        S += Y[j] * bxf->coeff[3*knots[j]+1];
        S += Z[j] * bxf->coeff[3*knots[j]+2];
        /* ------------------------------------------------ */

        /* dS/dp = 2Vp operation */
	bspline_score->total_grad[3*knots[j]+0] += 2 * X[j];
        bspline_score->total_grad[3*knots[j]+1] += 2 * Y[j];
        bspline_score->total_grad[3*knots[j]+2] += 2 * Z[j];
    }

    bspline_score->rmetric += S;
}
void
Bspline_regularize::analytic_init (
    const Bspline_xform* bxf,
    const Regularization_parms* reg_parms)
{
    double X[256];                      /* 16 x 16 matrix */
    double Y[256];                      /* 16 x 16 matrix */
    double Z[256];                      /* 16 x 16 matrix */
    double gs[3];

    this->cond = (double*)malloc(3*64*bxf->num_knots*sizeof(double));

    gs[0] = (double)bxf->grid_spac[0];
    gs[1] = (double)bxf->grid_spac[1];
    gs[2] = (double)bxf->grid_spac[2];

    this->QX_mats = (double*)malloc (4 * 16 * sizeof (double));
    this->QY_mats = (double*)malloc (4 * 16 * sizeof (double));
    this->QZ_mats = (double*)malloc (4 * 16 * sizeof (double));

    memset (this->QX_mats, 0, 4*16*sizeof(double));
    memset (this->QY_mats, 0, 4*16*sizeof(double));
    memset (this->QZ_mats, 0, 4*16*sizeof(double));

    this->QX = (double**)malloc (4 * sizeof (double*));
    this->QY = (double**)malloc (4 * sizeof (double*));
    this->QZ = (double**)malloc (4 * sizeof (double*));

    /* 4x4 matrices */
    this->QX[0] = this->QX_mats;
    this->QX[1] = this->QX[0] + 16;
    this->QX[2] = this->QX[1] + 16;
    this->QX[3] = this->QX[2] + 16;

    this->QY[0] = this->QY_mats;
    this->QY[1] = this->QY[0] + 16;
    this->QY[2] = this->QY[1] + 16;
    this->QY[3] = this->QY[2] + 16;

    this->QZ[0] = this->QZ_mats;
    this->QZ[1] = this->QZ[0] + 16;
    this->QZ[2] = this->QZ[1] + 16;
    this->QZ[3] = this->QZ[2] + 16;

    init_analytic (this->QX, this->QY, this->QZ, bxf);

    /* The below should probably be wrapped into init_analytic() */
    this->V_mats = (double*)malloc (32*4096 * sizeof (double));
    this->V = (double**)malloc (32 * sizeof (double*));

    /* The fifteen 64 x 64 V matrices */
    /*V[0] - V[5] -> Curvature, V[6] - V[8] -> Diffusion, 
     * V[9] - V[14] -> Linear Elastic , 
     * V[15] - V[20] -> Linear Elastic with complex weights (Diffusion)*/
    this->V[0] = this->V_mats;
    this->V[1] = this->V[0] + 4096;
    this->V[2] = this->V[1] + 4096;
    this->V[3] = this->V[2] + 4096;
    this->V[4] = this->V[3] + 4096;
    this->V[5] = this->V[4] + 4096;
    this->V[6] = this->V[5] + 4096;
    this->V[7] = this->V[6] + 4096;
    this->V[8] = this->V[7] + 4096;
    this->V[9] = this->V[8] + 4096;
    this->V[10] = this->V[9] + 4096;
    this->V[11] = this->V[10] + 4096;
    this->V[12] = this->V[11] + 4096;
    this->V[13] = this->V[12] + 4096;
    this->V[14] = this->V[13] + 4096;   
    this->V[15] = this->V[14] + 4096;
    this->V[16] = this->V[15] + 4096;
    this->V[17] = this->V[16] + 4096;
    this->V[18] = this->V[17] + 4096;
    this->V[19] = this->V[18] + 4096;
    this->V[20] = this->V[19] + 4096;
    this->V[21] = this->V[20] + 4096;
    this->V[22] = this->V[21] + 4096;
    this->V[23] = this->V[22] + 4096;
    this->V[24] = this->V[23] + 4096;   
    this->V[25] = this->V[24] + 4096;
    this->V[26] = this->V[25] + 4096;
    this->V[27] = this->V[26] + 4096;
    this->V[28] = this->V[27] + 4096;
    this->V[29] = this->V[28] + 4096;
    this->V[30] = this->V[29] + 4096;
    this->V[31] = this->V[30] + 4096;


    eval_integral (X, this->QX[2], this->QX[2], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[0], X, Y, Z);
    scale_Vmatrix(this->V[0], reg_parms->curvature_penalty); 

    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[2], this->QY[2], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[1], X, Y, Z);
    scale_Vmatrix(this->V[1], reg_parms->curvature_penalty);
    
    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[2], this->QZ[2], gs[2]);
    get_Vmatrix (this->V[2], X, Y, Z);
    scale_Vmatrix(this->V[2], reg_parms->curvature_penalty);
    
    eval_integral (X, this->QX[1], this->QX[1], gs[0]);
    eval_integral (Y, this->QY[1], this->QY[1], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[3], X, Y, Z);
    scale_Vmatrix(this->V[3], reg_parms->curvature_penalty * 
		    reg_parms->curvature_mixed_weight);
    
    eval_integral (X, this->QX[1], this->QX[1], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[1], this->QZ[1], gs[2]);
    get_Vmatrix (this->V[4], X, Y, Z);
    scale_Vmatrix(this->V[4], reg_parms->curvature_penalty *
		    reg_parms->curvature_mixed_weight);
    
    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[1], this->QY[1], gs[1]);
    eval_integral (Z, this->QZ[1], this->QZ[1], gs[2]);
    get_Vmatrix (this->V[5], X, Y, Z);
    scale_Vmatrix(this->V[5], reg_parms->curvature_penalty *
		    reg_parms->curvature_mixed_weight);
    
    eval_integral (X, this->QX[1], this->QX[1], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[6], X, Y, Z);
    scale_Vmatrix(this->V[6], reg_parms->diffusion_penalty);
    
    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[1], this->QY[1], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[7], X, Y, Z);
    scale_Vmatrix(this->V[7], reg_parms->diffusion_penalty);
    
    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[1], this->QZ[1], gs[2]);
    get_Vmatrix (this->V[8], X, Y, Z);
    scale_Vmatrix(this->V[8], reg_parms->diffusion_penalty);
    
    eval_integral (X, this->QX[1], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[1], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[9], X, Y, Z);
    scale_Vmatrix(this->V[9], reg_parms->lame_coefficient_1 * 
		    reg_parms->linear_elastic_multiplier);
    
    eval_integral (X, this->QX[0], this->QX[1], gs[0]);
    eval_integral (Y, this->QY[1], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[10], X, Y, Z);
    scale_Vmatrix(this->V[10], reg_parms->lame_coefficient_2 * 
		    reg_parms->linear_elastic_multiplier);
    
    eval_integral (X, this->QX[1], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[1], gs[2]);
    get_Vmatrix (this->V[11], X, Y, Z);
    scale_Vmatrix(this->V[11], reg_parms->lame_coefficient_1 * 
		    reg_parms->linear_elastic_multiplier);
    
    eval_integral (X, this->QX[0], this->QX[1], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[1], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[12], X, Y, Z);
    scale_Vmatrix(this->V[12], reg_parms->lame_coefficient_2 * 
		    reg_parms->linear_elastic_multiplier);
    
    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[1], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[1], gs[2]);
    get_Vmatrix (this->V[13], X, Y, Z);
    scale_Vmatrix(this->V[13], reg_parms->lame_coefficient_1 * 
		    reg_parms->linear_elastic_multiplier);
    
    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[1], gs[1]);
    eval_integral (Z, this->QZ[1], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[14], X, Y, Z);
    scale_Vmatrix(this->V[14], reg_parms->lame_coefficient_2 * 
		    reg_parms->linear_elastic_multiplier);
    
    eval_integral (X, this->QX[1], this->QX[1], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[15], X, Y, Z);
    scale_Vmatrix(this->V[15], (reg_parms->linear_elastic_multiplier) *
		    (2*reg_parms->lame_coefficient_1 +
		     reg_parms->lame_coefficient_2)/2);
    
    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[1], this->QY[1], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[16], X, Y, Z);
    scale_Vmatrix(this->V[16], (reg_parms->linear_elastic_multiplier) *
		    (2*reg_parms->lame_coefficient_1 + 
		     reg_parms->lame_coefficient_2)/2);
    
    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[1], this->QZ[1], gs[2]);
    get_Vmatrix (this->V[17], X, Y, Z);
    scale_Vmatrix(this->V[17], (reg_parms->linear_elastic_multiplier) *
		    (2*reg_parms->lame_coefficient_1 +
		     reg_parms->lame_coefficient_2)/2);

    eval_integral (X, this->QX[1], this->QX[1], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[18], X, Y, Z);
    scale_Vmatrix(this->V[18], (reg_parms->linear_elastic_multiplier) *
		    (reg_parms->lame_coefficient_1)/2);
    
    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[1], this->QY[1], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[19], X, Y, Z);
    scale_Vmatrix(this->V[19], (reg_parms->linear_elastic_multiplier) *
		    (reg_parms->lame_coefficient_1)/2);
    
    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[1], this->QZ[1], gs[2]);
    get_Vmatrix (this->V[20], X, Y, Z);
    scale_Vmatrix(this->V[20], (reg_parms->linear_elastic_multiplier) *
		    (reg_parms->lame_coefficient_1)/2);

    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[21], X, Y, Z);
    scale_Vmatrix(this->V[21], reg_parms->total_displacement_penalty);

    eval_integral (X, this->QX[3], this->QX[3], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[22], X, Y, Z);
    scale_Vmatrix(this->V[22], reg_parms->third_order_penalty);

    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[3], this->QY[3], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[23], X, Y, Z);
    scale_Vmatrix(this->V[23], reg_parms->third_order_penalty);

    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[3], this->QZ[3], gs[2]);
    get_Vmatrix (this->V[24], X, Y, Z);
    scale_Vmatrix(this->V[24], reg_parms->third_order_penalty);

    eval_integral (X, this->QX[2], this->QX[2], gs[0]);
    eval_integral (Y, this->QY[1], this->QY[1], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[25], X, Y, Z);
    scale_Vmatrix(this->V[25], reg_parms->third_order_penalty);

    eval_integral (X, this->QX[2], this->QX[2], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[1], this->QZ[1], gs[2]);
    get_Vmatrix (this->V[26], X, Y, Z);
    scale_Vmatrix(this->V[26], reg_parms->third_order_penalty);

    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[2], this->QY[2], gs[1]);
    eval_integral (Z, this->QZ[1], this->QZ[1], gs[2]);
    get_Vmatrix (this->V[27], X, Y, Z);
    scale_Vmatrix(this->V[27], reg_parms->third_order_penalty);

    eval_integral (X, this->QX[1], this->QX[1], gs[0]);
    eval_integral (Y, this->QY[2], this->QY[2], gs[1]);
    eval_integral (Z, this->QZ[0], this->QZ[0], gs[2]);
    get_Vmatrix (this->V[28], X, Y, Z);
    scale_Vmatrix(this->V[28], reg_parms->third_order_penalty);

    eval_integral (X, this->QX[1], this->QX[1], gs[0]);
    eval_integral (Y, this->QY[0], this->QY[0], gs[1]);
    eval_integral (Z, this->QZ[2], this->QZ[2], gs[2]);
    get_Vmatrix (this->V[29], X, Y, Z);
    scale_Vmatrix(this->V[29], reg_parms->third_order_penalty);

    eval_integral (X, this->QX[0], this->QX[0], gs[0]);
    eval_integral (Y, this->QY[1], this->QY[1], gs[1]);
    eval_integral (Z, this->QZ[2], this->QZ[2], gs[2]);
    get_Vmatrix (this->V[30], X, Y, Z);
    scale_Vmatrix(this->V[30], reg_parms->third_order_penalty);

    eval_integral (X, this->QX[1], this->QX[1], gs[0]);
    eval_integral (Y, this->QY[1], this->QY[1], gs[1]);
    eval_integral (Z, this->QZ[1], this->QZ[1], gs[2]);
    get_Vmatrix (this->V[31], X, Y, Z);
    scale_Vmatrix(this->V[31], reg_parms->third_order_penalty);

    printf ("Regularizer initialized\n");
}
    
    
/* flavor 'c' */

#if (OPENMP_FOUND)
void
Bspline_regularize::compute_score_analytic_omp (
    Bspline_score *bspline_score, 
    const Regularization_parms* reg_parms,
    const Bspline_regularize* rst,
    const Bspline_xform* bxf)
{
    plm_long i, n;

    double S = 0.0;

    Plm_timer* timer = new Plm_timer;
    timer->start ();

    memset (rst->cond, 0, 3*64*bxf->num_knots * sizeof (double));

    // Total number of regions in grid
    n = bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2];

    bspline_score->rmetric = 0.0;

#pragma omp parallel for reduction(+:S)
    for (i=0; i<n; i++) {
        plm_long knots[64];
        double sets[3*64];

        memset (sets, 0, 3*64*sizeof (double));

        find_knots_3 (knots, i, bxf->cdims);

        S += region_smoothness_omp (sets, reg_parms, bxf ,rst->V[0], knots);
        S += region_smoothness_omp (sets, reg_parms, bxf ,rst->V[1], knots);
        S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[2], knots);
        S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[3], knots);
        S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[4], knots);
        S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[5], knots);
	S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[6], knots);
	S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[7], knots);
	S += region_smoothness_omp (sets, reg_parms, bxf, rst->V[8], knots);
	S += region_smoothness_elastic_omp (sets, reg_parms, bxf, rst->V[9], 
			rst->V[10], rst->V[11], rst->V[12], rst->V[13], 
			rst->V[14], rst->V[15], rst->V[16], rst->V[17], 
			rst->V[18], rst->V[19], rst->V[20], knots);
        S+= region_smoothness_omp (sets, reg_parms, bxf, rst->V[21], knots);
	S+= region_smoothness_omp (sets, reg_parms, bxf, rst->V[22], knots);
	S+= region_smoothness_omp (sets, reg_parms, bxf, rst->V[23], knots);
	S+= region_smoothness_omp (sets, reg_parms, bxf, rst->V[24], knots);
	S+= region_smoothness_omp (sets, reg_parms, bxf, rst->V[25], knots);
	S+= region_smoothness_omp (sets, reg_parms, bxf, rst->V[26], knots);
	S+= region_smoothness_omp (sets, reg_parms, bxf, rst->V[27], knots);
	S+= region_smoothness_omp (sets, reg_parms, bxf, rst->V[28], knots);
	S+= region_smoothness_omp (sets, reg_parms, bxf, rst->V[29], knots);
	S+= region_smoothness_omp (sets, reg_parms, bxf, rst->V[30], knots);
	S+= region_smoothness_omp (sets, reg_parms, bxf, rst->V[31], knots);
	reg_sort_sets (rst->cond, sets, knots, bxf);
    }
    
    reg_update_grad (bspline_score, rst->cond, bxf);

    bspline_score->rmetric = S;
    bspline_score->time_rmetric = timer->report ();
    delete timer;
}
#endif


/* flavor 'b' */
void
Bspline_regularize::compute_score_analytic (
    Bspline_score *bspline_score, 
    const Regularization_parms* reg_parms,
    const Bspline_regularize* rst,
    const Bspline_xform* bxf)
{
    plm_long i, n;
    plm_long knots[64];

    Plm_timer* timer = new Plm_timer;
    timer->start ();

    // Total number of regions in grid
    n = bxf->rdims[0] * bxf->rdims[1] * bxf->rdims[2];

    bspline_score->rmetric = 0.0;

    for (i=0; i<n; i++) {
        // Get the set of 64 control points for this region
        find_knots_3 (knots, i, bxf->cdims);

        region_smoothness (bspline_score, reg_parms, bxf, rst->V[0], knots);
        region_smoothness (bspline_score, reg_parms, bxf, rst->V[1], knots);
        region_smoothness (bspline_score, reg_parms, bxf, rst->V[2], knots);
        region_smoothness (bspline_score, reg_parms, bxf, rst->V[3], knots);
        region_smoothness (bspline_score, reg_parms, bxf, rst->V[4], knots);
        region_smoothness (bspline_score, reg_parms, bxf, rst->V[5], knots);
        region_smoothness (bspline_score, reg_parms, bxf, rst->V[6], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[7], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[8], knots);
	region_smoothness_elastic (bspline_score, reg_parms, bxf, rst->V[9], 
			rst->V[10], rst->V[11], rst->V[12], rst->V[13], 
			rst->V[14], rst->V[15], rst->V[16], rst->V[17], 
			rst->V[18], rst->V[19], rst->V[20], knots);
    	region_smoothness (bspline_score, reg_parms, bxf, rst->V[21], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[22], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[23], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[24], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[25], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[26], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[27], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[28], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[29], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[30], knots);
	region_smoothness (bspline_score, reg_parms, bxf, rst->V[31], knots);
    }

    bspline_score->time_rmetric = timer->report ();
    delete timer;
}
