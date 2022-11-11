/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmcli_config.h"
#include "itkImageRegionIteratorWithIndex.h"

#include "itk_bbox.h"
#include "itk_image_clone.h"
#include "itk_image_save.h"
#include "itk_image_type.h"
#include "pcmd_bbox.h"
#include "plm_clp.h"
#include "plm_image.h"

class Bbox_parms {
public:
    std::string input_fn;
    std::string output_mask_fn;
    float margin;
    bool z_only;
public:
    Bbox_parms () {
        margin = 0.f;
        z_only = false;
    }
};

void
do_bbox (const Bbox_parms *parms)
{
    Plm_image pli (parms->input_fn);
    UCharImageType::Pointer img = pli.itk_uchar();

    float bbox_coordinates[6];
    int bbox_indices[6];
    itk_bbox (img, bbox_coordinates, bbox_indices);

    bbox_coordinates[0] -= parms->margin;
    bbox_coordinates[1] += parms->margin;
    bbox_coordinates[2] -= parms->margin;
    bbox_coordinates[3] += parms->margin;
    bbox_coordinates[4] -= parms->margin;
    bbox_coordinates[5] += parms->margin;
    
    printf ("coordinates: %f %f %f %f %f %f\n",
        bbox_coordinates[0], bbox_coordinates[1], bbox_coordinates[2],
        bbox_coordinates[3], bbox_coordinates[4], bbox_coordinates[5]);

    printf ("indices (no margin): %d %d %d %d %d %d\n",
        bbox_indices[0], bbox_indices[1], bbox_indices[2],
        bbox_indices[3], bbox_indices[4], bbox_indices[5]);

    if (parms->output_mask_fn != "") {
        UCharImageType::Pointer img_out = itk_image_clone_empty (img);
        itk::ImageRegionIteratorWithIndex< UCharImageType >
            it (img_out,  img_out->GetLargestPossibleRegion());
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            FloatPoint3DType point;
            UCharImageType::RegionType::IndexType idx = it.GetIndex();
            img_out->TransformIndexToPhysicalPoint (idx, point);
            if (point[2] < bbox_coordinates[2*2+0]
                || point[2] > bbox_coordinates[2*2+1])
            {
                continue;
            }
            if ((parms->z_only)
                || (point[0] > bbox_coordinates[0*2+0]
                    && point[0] < bbox_coordinates[0*2+1]
                    && point[1] > bbox_coordinates[1*2+0]
                    && point[1] < bbox_coordinates[1*2+1]))
            {
                it.Set (1);
            }
        }
        itk_image_save (img_out, parms->output_mask_fn);
    }
}

static void
usage_fn (dlib::Plm_clp* parser, int argc, char *argv[])
{
    std::cout << "Usage: plastimatch bbox [options] input-file\n";
    parser->print_options (std::cout);
    std::cout << std::endl;
}

static void
parse_fn (
    Bbox_parms* parms, 
    dlib::Plm_clp* parser, 
    int argc, 
    char* argv[]
)
{
    /* Add --help, --version */
    parser->add_default_options ();

    /* Basic options */
    parser->add_long_option ("", "output",
	"Create an output image with a bounding box mask", 1, "");
    parser->add_long_option ("", "margin",
	"Expand bounding box by margin (mm, may be negative)", 1, "0");
    parser->add_long_option ("", "z-only",
	"When creating output image, only consider z axis", 0);

    /* Parse options */
    parser->parse (argc,argv);

    /* Handle --help, --version */
    parser->check_default_options ();

    /* Check that all necessary inputs are given */
    if (parser->number_of_arguments() != 1) {
	throw (dlib::error (
                "Error.  A single input file must be given."));
    }

    /* Get input file */
    parms->input_fn = (*parser)[0];

    /* Copy remaining values into parameter struct */
    parms->margin = parser->get_float ("margin");
    if (parser->have_option ("output")) {
        parms->output_mask_fn = parser->get_string ("output");
    }
    if (parser->have_option ("z-only")) {
        parms->z_only = true;
    }
}

void
do_command_bbox (int argc, char *argv[])
{
    Bbox_parms bbox_parms;

    plm_clp_parse (&bbox_parms, &parse_fn, &usage_fn, argc, argv, 1);

    do_bbox (&bbox_parms);
}
