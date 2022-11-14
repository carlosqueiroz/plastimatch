/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"

#include "fdk.h"
#if CUDA_FOUND
#include "fdk_cuda.h"
#endif
#include "fdk_opencl.h"
#include "fdk_util.h"
#include "mha_io.h"
#include "pcmd_fdk.h"
#include "plm_clp.h"
#include "print_and_exit.h"
#include "proj_image_dir.h"
#include "string_util.h"
#include "volume.h"

static void 
set_default_parms (Fdk_parms* parms)
{
    parms->threading = THREADING_CPU_OPENMP;
    parms->image_range_requested = 0;
    parms->first_img = 0;
    parms->last_img = 119;
    parms->skip_img = 1;
    parms->dim[0] = 256;
    parms->dim[1] = 256;
    parms->dim[2] = 100;
    parms->vol_size[0] = 300.0f;
    parms->vol_size[1] = 300.0f;
    parms->vol_size[2] = 150.0f;
    parms->xy_offset[0] = 0.f;
    parms->xy_offset[1] = 0.f;
    parms->scale = 1.0f;
    parms->filter = FDK_FILTER_TYPE_RAMP;
    parms->input_dir = ".";
    parms->output_file = "output.mha";
    parms->flavor = 'c';
    parms->full_fan=1;
    parms->Full_normCBCT_name="Full_norm.mh5";
    parms->Full_radius=120;
    parms->Half_normCBCT_name="Half_norm.mh5";
    parms->Half_radius=220;
}

#if defined (commentout)
void 
fdk_parse_args (Fdk_parms* parms, int argc, char* argv[])
{
    int i, rc;
	
    if (argc < 2)
    { print_usage(); exit(1); }

    set_default_parms (parms);
    for (i = 1; i < argc; i++) {
	if (argv[i][0] != '-') break;
	if (!strcmp (argv[i], "-A")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
#if CUDA_FOUND
	    if (!strcmp(argv[i], "cuda") || !strcmp(argv[i], "CUDA")) {
		parms->threading = THREADING_CUDA;
		continue;
	    }
#endif
#if OPENCL_FOUND
	    if (!strcmp(argv[i], "opencl") || !strcmp(argv[i], "OPENCL")) {
		parms->threading = THREADING_OPENCL;
		continue;
	    }
#endif
	    /* Default */
	    parms->threading = THREADING_CPU_OPENMP;
	}
	else if (!strcmp (argv[i], "-a")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    parms->image_range_requested = 1;
	    rc = sscanf (argv[i], "%d %d %d" , 
		&parms->first_img,
		&parms->skip_img,
		&parms->last_img);
	    if (rc == 1) {
		parms->last_img = parms->first_img;
		parms->skip_img = 1;
	    } else if (rc == 2) {
		parms->last_img = parms->skip_img;
		parms->skip_img = 1;
	    } else if (rc != 3) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-f")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    if (!strcmp(argv[i], "none") || !strcmp(argv[i], "NONE")) {
		parms->filter = FDK_FILTER_TYPE_NONE;
	    }
	    else if (!strcmp(argv[i], "ramp") || !strcmp(argv[i], "RAMP")) {
		parms->filter = FDK_FILTER_TYPE_RAMP;
	    }
	    else {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-X")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    parms->flavor = argv[i][0];
	    if (parms->flavor != '0' && parms->flavor != 'a'
		&& parms->flavor != 'b' && parms->flavor != 'c'
		&& parms->flavor != 'd') {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-I")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    parms->input_dir = strdup (argv[i]);
	}
	else if (!strcmp (argv[i], "-O")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    parms->output_file = strdup (argv[i]);
	}
	else if (!strcmp (argv[i], "-r")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    unsigned int a, b, c;
	    rc = sscanf (argv[i], "%d %d %d", &a, &b, &c);
	    if (rc == 1) {
		parms->dim[0] = a;
		parms->dim[1] = a;
		parms->dim[2] = a;
	    } else if (rc == 3) {
		parms->dim[0] = a;
		parms->dim[1] = b;
		parms->dim[2] = c;
	    } else {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-s")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    rc = sscanf (argv[i], "%g" , &parms->scale);
	    if (rc != 1) {
		print_usage ();
	    }
	}
	else if (!strcmp (argv[i], "-x")) {
	    i++;
	    rc = sscanf (argv[i], "%f %f", 
		&parms->xy_offset[0],
		&parms->xy_offset[1]);
	    if (rc != 2) {
                if (i == (argc-1) || argv[i+1][0] == '-') {
                    fprintf(stderr, 
                        "option %s requires an argument\n", argv[i]);
                    exit(1);
                } else {
                    print_usage ();
                }
	    }
	}
	else if (!strcmp (argv[i], "-z")) {
	    if (i == (argc-1) || argv[i+1][0] == '-') {
		fprintf(stderr, "option %s requires an argument\n", argv[i]);
		exit(1);
	    }
	    i++;
	    rc = sscanf (argv[i], "%g %g %g", 
		&parms->vol_size[0],
		&parms->vol_size[1],
		&parms->vol_size[2]);
	    if (rc == 1) {
		parms->vol_size[1] = parms->vol_size[0];
		parms->vol_size[2] = parms->vol_size[0];
	    } else if (rc != 3) {
		print_usage ();
	    }
	}
	else {
	    print_usage ();
	}
    }
}
#endif

void
do_fdk (Fdk_parms *parms)
{

    Volume* vol;
    Proj_image_dir *proj_dir;

    /* Look for input files */
    proj_dir = new Proj_image_dir (parms->input_dir);
    if (proj_dir->num_proj_images < 1) {
        print_and_exit ("Error: couldn't find input files in directory %s\n",
            parms->input_dir.c_str());
    }

    /* Set the panel offset */
    double xy_offset[2] = { parms->xy_offset[0], parms->xy_offset[1] };
    proj_dir->set_xy_offset (xy_offset);

    /* Choose subset of input files if requested */
    if (parms->image_range_requested) {
	proj_dir->select (parms->first_img, parms->skip_img, 
	    parms->last_img);
    }

    /* Allocate memory */
    vol = my_create_volume (parms);

    printf ("Reconstructing...\n");
    switch (parms->threading) {
#if (CUDA_FOUND)
    case THREADING_CUDA:
	CUDA_reconstruct_conebeam (vol, proj_dir, parms);
	break;
#endif
#if (OPENCL_FOUND)
    case THREADING_OPENCL:
        opencl_reconstruct_conebeam (vol, proj_dir, parms);
        //OPENCL_reconstruct_conebeam_and_convert_to_hu (vol, proj_dir, &parms);
        break;
#endif
    case THREADING_CPU_SINGLE:
    case THREADING_CPU_OPENMP:
    default:
	reconstruct_conebeam (vol, proj_dir, parms);
    }

    /* Free memory */
    delete proj_dir;

    /* Prepare HU values in output volume */
    convert_to_hu (vol, parms);

    /* Do bowtie filter corrections */
    //fdk_do_bowtie (vol, &parms);

    /* Write output */
    printf ("Writing output volume(s)...\n");
    write_mha (parms->output_file.c_str(), vol);

    /* Free memory */
    delete vol;

    printf(" done.\n\n");
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch fdk [options]\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Fdk_parms* parms,
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    parser->add_long_option ("A", "threading",
	"Threading option {cpu,cuda,opencl} (default: cpu)", 1, "cpu");
    parser->add_long_option ("a", "image-range",
	"Use a sub-range of available images \"first ((skip) last)\"", 1, "");
    parser->add_long_option ("f", "filter",
	"Choice of filter {none,ramp} (default: ramp)", 1, "ramp");
    parser->add_long_option ("r", "dim",
	"The output image resolution in voxels \"num (num num)\" "
        "(default: 256 256 100", 1, "256 256 100");
    parser->add_long_option ("s", "intensity-scale",
	"Scaling factor for output image intensity", 1, "1.0");
    parser->add_long_option ("x", "detector-offset",
	"The translational offset of the detector \"x0 y0\", in pixels",
        1, "");
    parser->add_long_option ("z", "volume-size",
	"Physical size of reconstruction volume \"s1 s2 s3\", in mm "
        "(default: 300 300 150)", 1, "300 300 150");
    parser->add_long_option ("I", "input", 
	"Input file", 1, "");
    parser->add_long_option ("O", "output",
	"Prefix for output file(s)", 1, "");
    parser->add_long_option ("X", "flavor",
	"Implementation flavor {0,a,b,c,d} (default: c)", 1, "c");

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that input and output are given */
    parser->check_required ("input");
    parser->check_required ("output");

    set_default_parms (parms);
    
    /* Insert command line options into options array */
    std::string s;
    s = make_lowercase (parser->get_string("threading"));
    if (s == "cpu" || s == "openmp") {
        parms->threading = THREADING_CPU_OPENMP;
    } else if (s == "cuda" || s == "gpu") {
        parms->threading = THREADING_CUDA;
    } else if (s == "opencl") {
        parms->threading = THREADING_OPENCL;
    } else {
        throw (dlib::error ("Error.  Option \"threading\" should be one of "
                "{cpu,cuda,opencl}"));
    }

    if (parser->have_option ("image-range")) {
        parms->image_range_requested = 1;
        s = parser->get_string("image-range");
        int rc = sscanf (s.c_str(), "%d %d %d", 
		&parms->first_img,
		&parms->skip_img,
		&parms->last_img);
        if (rc == 1) {
            parms->last_img = parms->first_img;
            parms->skip_img = 1;
        } else if (rc == 2) {
            parms->last_img = parms->skip_img;
            parms->skip_img = 1;
        } else if (rc != 3) {
            throw (dlib::error ("Error.  Option \"image-range\" should "
                    "specify between one and three integers"));
        }
    }
    
    s = make_lowercase (parser->get_string("filter"));
    if (s == "none") {
        parms->filter = FDK_FILTER_TYPE_NONE;
    } else if (s == "ramp") {
        parms->filter = FDK_FILTER_TYPE_RAMP;
    } else {
        throw (dlib::error ("Error.  Option \"filter\" should be one of "
                "{none,ramp}"));
    }

    parser->assign_plm_long_13 (parms->dim, "dim");
    parms->scale = parser->get_float ("intensity-scale");
    if (parser->have_option ("detector-offset")) {
        parser->assign_float_2 (parms->xy_offset, "detector-offset");
    }
    parser->assign_float_13 (parms->vol_size, "volume-size");

    parms->input_dir = parser->get_string ("input");
    parms->output_file = parser->get_string ("output");
    parms->flavor = parser->get_string ("flavor").c_str()[0];
}

void
do_command_fdk (int argc, char *argv[])
{
    Fdk_parms parms;
    plm_clp_parse (&parms, &parse_fn, &usage_fn, argc, argv, 1);
    do_fdk (&parms);
}
