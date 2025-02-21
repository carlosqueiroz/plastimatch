/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "itk_image_load.h"
#include "mha_io.h"
#include "plm_clp.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "vf_convolve.h"
#include "xf_invert.h"
#include "xform.h"

class Vf_invert_parms {
public:
    std::string vf_in_fn;
    std::string vf_out_fn;
    std::string fixed_img_fn;
    bool have_dim;
    bool have_origin;
    bool have_spacing;
    plm_long dim[3];
    float origin[3];
    float spacing[3];
    bool old_algorithm;
    int iterations;
public:
    Vf_invert_parms () {
        vf_in_fn = "";
        vf_out_fn = "";
        fixed_img_fn = "";
        for (int d = 0; d < 3; d++) {
            dim[d] = 0;
            origin[d] = 0.f;
            spacing[d] = 1.f;
        }
        have_dim = false;
        have_origin = false;
        have_spacing = false;
        old_algorithm = false;
        iterations = 20;
    }
};

void
do_vf_invert_old (Vf_invert_parms* parms)
{
    plm_long i, j, k, v;
    int its;
    float x, y, z;
    Plm_image_header pih;
    Volume *mask, *vf_in, *vf_inv, *vf_smooth, *vf_out;
    float *img_in, *img_inv, *img_smooth, *img_out;
    unsigned char *img_mask;
    float ker[3] = { 0.3, 0.4, 0.3 };

    if (parms->fixed_img_fn != "") {
        /* if given, use the parameters from user-supplied fixed image */
        FloatImageType::Pointer fixed = itk_image_load_float (
            parms->fixed_img_fn.c_str(), 0);
        pih.set_from_itk_image (fixed);
        pih.get_origin (parms->origin);
        pih.get_spacing (parms->spacing);
        pih.get_dim (parms->dim);

        pih.set_from_gpuit (parms->dim, parms->origin, parms->spacing, 0);
    }

    /* GCS FIX: Need direction cosines */
    /* Create mask volume */
    mask = new Volume (parms->dim, parms->origin, parms->spacing, 0,
        PT_UCHAR, 1);

    /* GCS FIX: Need direction cosines */
    /* Create tmp volume */
    vf_inv = new Volume (parms->dim, parms->origin, parms->spacing, 0, 
        PT_VF_FLOAT_INTERLEAVED, 1);

    /* Load input vf */
    vf_in = read_mha (parms->vf_in_fn.c_str());
    vf_convert_to_interleaved (vf_in);

    /* Populate mask & tmp volume */
    img_mask = (unsigned char*) mask->img;
    img_in = (float*) vf_in->img;
    img_inv = (float*) vf_inv->img;
    for (z = vf_in->origin[2], k = 0, v = 0; k < vf_in->dim[2]; k++, z+=vf_in->spacing[2]) {
        for (y = vf_in->origin[1], j = 0; j < vf_in->dim[1]; j++, y+=vf_in->spacing[1]) {
            for (x = vf_in->origin[0], i = 0; i < vf_in->dim[0]; v++, i++, x+=vf_in->spacing[0]) {
                plm_long mijk[3], midx;
                float mxyz[3];
                mxyz[0] = x + img_in[3*v+0];
                mijk[0] = ROUND_INT ((mxyz[0] - vf_inv->origin[0]) / vf_inv->spacing[0]);
                mxyz[1] = y + img_in[3*v+1];
                mijk[1] = (mxyz[1] - vf_inv->origin[1]) / vf_inv->spacing[1];
                mxyz[2] = z + img_in[3*v+2];
                mijk[2] = (mxyz[2] - vf_inv->origin[2]) / vf_inv->spacing[2];

                if (mijk[0] < 0 || mijk[0] >= vf_inv->dim[0]) continue;
                if (mijk[1] < 0 || mijk[1] >= vf_inv->dim[1]) continue;
                if (mijk[2] < 0 || mijk[2] >= vf_inv->dim[2]) continue;

                midx = (mijk[2] * vf_inv->dim[1] + mijk[1]) * vf_inv->dim[0] + mijk[0];
                img_inv[3*midx+0] = -img_in[3*v+0];
                img_inv[3*midx+1] = -img_in[3*v+1];
                img_inv[3*midx+2] = -img_in[3*v+2];
                img_mask[midx] ++;
            }
        }
    }

    /* We're done with input volume now. */
    delete vf_in;

    /* GCS FIX: Need direction cosines */
    /* Create tmp & output volumes */
    vf_out = new Volume (parms->dim, parms->origin, parms->spacing, 0, 
        PT_VF_FLOAT_INTERLEAVED, 3);
    img_out = (float*) vf_out->img;
    /* GCS FIX: Need direction cosines */
    vf_smooth = new Volume (parms->dim, parms->origin, parms->spacing, 0, 
        PT_VF_FLOAT_INTERLEAVED, 3);
    img_smooth = (float*) vf_smooth->img;

    /* Iterate, pasting and smoothing */
    printf ("Paste and smooth loop\n");
    for (its = 0; its < parms->iterations; its++) {
        printf ("Iteration %d/%d\n", its, parms->iterations);
        /* Paste */
        for (v = 0, k = 0; k < vf_out->dim[2]; k++) {
            for (j = 0; j < vf_out->dim[1]; j++) {
                for (i = 0; i < vf_out->dim[0]; i++, v++) {
                    if (img_mask[v]) {
                        img_smooth[3*v+0] = img_inv[3*v+0];
                        img_smooth[3*v+1] = img_inv[3*v+1];
                        img_smooth[3*v+2] = img_inv[3*v+2];
                    } else {
                        img_smooth[3*v+0] = img_out[3*v+0];
                        img_smooth[3*v+1] = img_out[3*v+1];
                        img_smooth[3*v+2] = img_out[3*v+2];
                    }
                }
            }
        }

        /* Smooth the estimate into vf_out.  The volumes are ping-ponged. */
        printf ("Convolving\n");
        vf_convolve_x (vf_out, vf_smooth, ker, 3);
        vf_convolve_y (vf_smooth, vf_out, ker, 3);
        vf_convolve_z (vf_out, vf_smooth, ker, 3);
    }
    printf ("Done.\n");

    /* We're done with the mask & smooth image. */
    delete mask;
    delete vf_smooth;

    /* Write the output */
    write_mha (parms->vf_out_fn.c_str(), vf_out);
    delete vf_out;
}

void
do_xf_invert_new (Vf_invert_parms* parms)
{
    Xf_invert xf_invert;

    xf_invert.set_input_xf (parms->vf_in_fn.c_str());
    if (parms->fixed_img_fn != "") {
        xf_invert.set_fixed_image (parms->fixed_img_fn.c_str());
    }
    if (parms->have_dim) {
        xf_invert.set_dim (parms->dim);
    }
    if (parms->have_origin) {
        xf_invert.set_origin (parms->origin);
    }
    if (parms->have_spacing) {
        xf_invert.set_spacing (parms->spacing);
    }
#if defined (commentout)
    /* GCS FIX: direction cosines */
    if (parms->have_direction_cosines) {
        xf_invert.set_direction_cosines (parms->direction_cosines);
    }
#endif

    xf_invert.set_iterations (parms->iterations);

    /* Invert the vf */
    xf_invert.run ();

    /* Write the output */
    const Xform *xf = xf_invert.get_output ();
    xf->save (parms->vf_out_fn);

#if defined (commentout)
    write_mha (parms->vf_out_fn.c_str(), xf_invert.get_output_volume());
    plm_image_save_vol (parms->vf_out_fn.c_str(), 
        xf_invert.get_output_volume());
#endif
}

void
do_vf_invert (Vf_invert_parms* parms)
{
    if (parms->old_algorithm) {
        do_vf_invert_old (parms);
    } else {
        do_xf_invert_new (parms);
    }
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    printf ("Usage: plastimatch %s [options] input_file [input_file ...]\n", 
        argv[1]);
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Vf_invert_parms *parms, 
    dlib::Plm_clp *parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Basic options */
    parser->add_long_option ("", "input", 
	"input transform file name", 1, "");
    parser->add_long_option ("", "output", 
	"output transform file name", 1, "");
    parser->add_long_option ("", "dim", 
        "size of output vector field in voxels \"x [y z]\"", 1, "");
    parser->add_long_option ("", "origin", 
        "location of first voxel of output vector field in mm \"x y z\"", 
        1, "");
    parser->add_long_option ("", "spacing", 
        "voxel spacing of output vector field in mm \"x [y z]\"", 1, "");
    parser->add_long_option ("", "fixed", 
        "fixed image (match output vector field size to this image)", 1, "");
    parser->add_long_option ("", "old-algorithm", 
        "use the old vector field inversion algorithm", 0);
    parser->add_long_option ("", "iterations", 
        "number of iterations to run for vector field inversion (default = 20)", 1, "");

    /* Parse the command line arguments */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that an input file was given */
    if (!parser->option ("input")) {
        throw (dlib::error ("Error.  Please specify an input file "
                "using the --input options"));
    }

    /* Check that an output file was given */
    if (!parser->option ("output")) {
        throw (dlib::error ("Error.  Please specify an output file "
                "using the --output options"));
    }

    /* Check that output dimensions are known */
    /* GCS FIX: This cannot be checked here.  If input is rigid transform, 
       these parameters are not needed. */
#if defined (commentout)
    if (!parser->option ("dim") 
        || !parser->option("origin")
        || !parser->option("spacing"))
    {
        if (!parser->option ("fixed")) {
            throw (dlib::error ("Error.  Please specify either dim, origin, "
                    " and spacing -or- a fixed file"));
        }
    }
#endif

    /* Copy values into parameter struct */
    parms->vf_in_fn = parser->get_string("input");
    parms->vf_out_fn = parser->get_string("output");
    if (parser->option ("fixed")) {
        parms->fixed_img_fn = parser->get_string("fixed");
    }
    if (parser->option ("dim")) {
        parms->have_dim = true;
        parser->assign_plm_long_13 (parms->dim, "dim");
    }
    if (parser->option ("origin")) {
        parms->have_origin = true;
        parser->assign_float_13 (parms->origin, "origin");
    }
    if (parser->option ("spacing")) {
        parms->have_spacing = true;
        parser->assign_float_13 (parms->spacing, "spacing");
    }
    if (parser->option ("old-algorithm")) {
        parms->old_algorithm = true;
    }
    if (parser->option ("iterations")) {
        parser->get_value (parms->iterations, "iterations");
    }
}

void
do_command_vf_invert (int argc, char *argv[])
{
    Vf_invert_parms parms;
    
    /* Parse command line parameters */
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);

    /* Do the work */
    do_vf_invert (&parms);
}
