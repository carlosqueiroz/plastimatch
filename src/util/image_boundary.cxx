/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "itkImage.h"

#include "image_boundary.h"
#include "itk_image_load.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "volume.h"
#include "volume_boundary_behavior.h"

class Image_boundary_private {
public:
    Image_boundary_private () {
        vbb = ADAPTIVE_PADDING;
        vbt = INTERIOR_EDGE;
    }
public:
    UCharImageType::Pointer input_image;
    UCharImageType::Pointer output_image;
    Volume_boundary_behavior vbb;
    Volume_boundary_type vbt;
public:
    void run_vbt_edge ();
    void run_vbt_face ();
    void run ();
protected:
    unsigned char classify_edge (
        const Volume::Pointer& vol_in,
        const unsigned char *img_in,
        const bool zero_pad[3],
        plm_long i, plm_long j, plm_long k, plm_long v)
    {
        unsigned char value = classify_face (vol_in, img_in, zero_pad,
            i, j, k, v);
        return (value == 0) ? 0 : 1;
    }

    unsigned char classify_face (
        const Volume::Pointer& vol_in,
        const unsigned char *img_in,
        const bool zero_pad[3],
        plm_long i, plm_long j, plm_long k, plm_long v)
    {
        unsigned char value = 0;

        /* If not inside volume, then not on boundary */
        unsigned char this_vox = img_in[v];
        if (!img_in[v]) {
            return value;
        }

        /* Find boundary faces in i direction */
        if (i == 0) {
            if (zero_pad[0]) {
                value |= VBB_MASK_NEG_I;
            }
        } else {
            if (img_in[vol_in->index (i-1, j, k)] == 0) {
                value |= VBB_MASK_NEG_I;
            }
        }
        if (i == vol_in->dim[0]-1) {
            if (zero_pad[0]) {
                value |= VBB_MASK_POS_I;
            }
        } else {
            if (img_in[vol_in->index (i+1, j, k)] == 0) {
                value |= VBB_MASK_POS_I;
            }
        }

        /* Find boundary faces in j direction */
        if (j == 0) {
            if (zero_pad[1]) {
                value |= VBB_MASK_NEG_J;
            }
        } else {
            if (img_in[vol_in->index (i, j-1, k)] == 0) {
                value |= VBB_MASK_NEG_J;
            }
        }
        if (j == vol_in->dim[1]-1) {
            if (zero_pad[1]) {
                value |= VBB_MASK_POS_J;
            }
        } else {
            if (img_in[vol_in->index (i, j+1, k)] == 0) {
                value |= VBB_MASK_POS_J;
            }
        }

        /* Find boundary faces in i direction */
        if (k == 0) {
            if (zero_pad[2]) {
                value |= VBB_MASK_NEG_K;
            }
        } else {
            if (img_in[vol_in->index (i, j, k-1)] == 0) {
                value |= VBB_MASK_NEG_K;
            }
        }
        if (k == vol_in->dim[2]-1) {
            if (zero_pad[2]) {
                value |= VBB_MASK_POS_K;
            }
        } else {
            if (img_in[vol_in->index (i, j, k+1)] == 0) {
                value |= VBB_MASK_POS_K;
            }
        }
        return value;
    }
};

void 
Image_boundary_private::run_vbt_edge ()
{
    /* Convert to Plm_image type */
    Plm_image pli_in (this->input_image);
    Volume::Pointer vol_in = pli_in.get_volume_uchar ();
    unsigned char *img_in = (unsigned char*) vol_in->img;

    /* Allocate output image */
    Plm_image::Pointer pli_out = pli_in.clone ();
    Volume::Pointer vol_out = pli_out->get_volume_uchar ();
    unsigned char *img_out = (unsigned char*) vol_out->img;

    /* Figure out padding strategy for each of the three dimensions */
    bool zero_pad[3];
    for (int d = 0; d < 3; d++) {
        zero_pad[d] = (vbb == ZERO_PADDING
            || (vbb == ADAPTIVE_PADDING && vol_in->dim[d] > 1));
    }

    /* Compute the boundary */
    for (plm_long k = 0, v = 0; k < vol_in->dim[2]; k++) {
        for (plm_long j = 0; j < vol_in->dim[1]; j++) {
            for (plm_long i = 0; i < vol_in->dim[0]; i++, v++) {
                img_out[v] = classify_edge (vol_in, img_in, zero_pad, i, j, k, v);
            }
        }
    }

    /* Save the output image */
    this->output_image = pli_out->itk_uchar ();
}

void 
Image_boundary_private::run_vbt_face ()
{
    /* Convert to Plm_image type */
    Plm_image pli_in (this->input_image);
    Volume::Pointer vol_in = pli_in.get_volume_uchar ();
    unsigned char *img_in = (unsigned char*) vol_in->img;

    /* Allocate output image */
    Plm_image::Pointer pli_out = pli_in.clone ();
    Volume::Pointer vol_out = pli_out->get_volume_uchar ();
    unsigned char *img_out = (unsigned char*) vol_out->img;

    /* Figure out padding strategy for each of the three dimensions */
    bool zero_pad[3];
    for (int d = 0; d < 3; d++) {
        zero_pad[d] = (vbb == ZERO_PADDING
            || (vbb == ADAPTIVE_PADDING && vol_in->dim[d] > 1));
    }

    /* Compute the boundary */
    for (plm_long k = 0, v = 0; k < vol_in->dim[2]; k++) {
        for (plm_long j = 0; j < vol_in->dim[1]; j++) {
            for (plm_long i = 0; i < vol_in->dim[0]; i++, v++) {
                img_out[v] = classify_face (vol_in, img_in, zero_pad, i, j, k, v);
            }
        }
    }

    /* Save the output image */
    this->output_image = pli_out->itk_uchar ();
}

void 
Image_boundary_private::run ()
{
    switch (vbt) {
    case INTERIOR_EDGE:
        run_vbt_edge ();
        break;
    case INTERIOR_FACE:
    default:
        run_vbt_face ();
        break;
    }
}

Image_boundary::Image_boundary ()
{
    d_ptr = new Image_boundary_private;
}

Image_boundary::~Image_boundary ()
{
    delete d_ptr;
}

void 
Image_boundary::set_input_image (const char* image_fn)
{
    d_ptr->input_image = itk_image_load_uchar (image_fn, 0);
}

void 
Image_boundary::set_input_image (
    const UCharImageType::Pointer image)
{
    d_ptr->input_image = image;
}

void
Image_boundary::set_volume_boundary_behavior (Volume_boundary_behavior vbb)
{
    d_ptr->vbb = vbb;
}

void
Image_boundary::set_volume_boundary_type (Volume_boundary_type vbt)
{
    d_ptr->vbt = vbt;
}

void 
Image_boundary::run ()
{
    d_ptr->run ();
}

UCharImageType::Pointer
Image_boundary::get_output_image ()
{
    return d_ptr->output_image;
}

UCharImageType::Pointer
do_image_boundary (UCharImageType::Pointer image)
{
    Image_boundary ib;
    ib.set_input_image (image);
    ib.run ();
    return ib.get_output_image ();
}
