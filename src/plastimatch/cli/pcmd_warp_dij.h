/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*  Warp one or more dij matrices based on a vector field */
#ifndef _pcmd_warp_dij_h_
#define _pcmd_warp_dij_h_

#include "plmcli_config.h"
#include "warp_parms.h"
#include "xform.h"


typedef unsigned short ushort;
typedef unsigned long ulong;

typedef struct __Pencil_Beam Pencil_Beam;
struct __Pencil_Beam {
    float energy;
    float spot_x;
    float spot_y;
    int nvox;
    ushort* vox;
};

typedef struct __Dij_Matrix Dij_Matrix;
struct __Dij_Matrix {
    float gantry_angle;
    float table_angle;
    float collimator_angle;
    float spot_spacing_dx;
    float spot_spacing_dy;
    float voxel_size_dx;
    float voxel_size_dy;
    float voxel_size_dz;
    int dose_cube_size[3];
    int num_pencil_beams;
    float absolute_dose_coefficient;
};

class Dij_header {
public:
    plm_long dose_dim[3];
    float dose_origin[3];
    float dose_spacing[3];

public:
    Dij_header ();
    void load_from_dif(const char* dif_in);
    void load_from_dij(const Dij_Matrix& dij);
    void load_from_img(const char* fixed_in);
    void load_user_dim(const plm_long dim[3]);
    void load_user_origin(const float origin[3]);
    void load_user_spacing(const float spacing[3]);  
};

void dij_parse_error (void);

void dij_write_error (void);

void ctatts_parse_error (void);

void dif_parse_error (void);

void load_dif (Dij_header* dijh, const char* dif_in);

FloatImageType::Pointer make_dose_image (
    Dij_header *dijh,
    Dij_Matrix *dij_matrix);

void set_pencil_beam_to_image (
    FloatImageType::Pointer& img,
    const Pencil_Beam *pb);

void read_pencil_beam (Pencil_Beam* pb, FILE* fp);

void write_pencil_beam (Pencil_Beam* pb, FloatImageType::Pointer img, FILE* fp);

void read_dij_header (Dij_Matrix* dij_matrix, FILE* fp);

void write_dij_header (Dij_Matrix* dij_matrix, FILE* fp);

FloatImageType::Pointer
warp_pencil_beam (
    DeformationFieldType::Pointer& vf, 
    FloatImageType::Pointer& pb_img, 
    Dij_Matrix *dij_matrix,
    Dij_header *dijh, 
    Pencil_Beam *pb);

void convert_vector_field (
    Xform::Pointer& xf,
    Warp_parms *parms);

void warp_dij_main (Warp_parms* parms);

#endif
