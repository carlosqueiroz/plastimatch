/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"

#include "cxt_extract.h"
#include "cxt_io.h"
#include "dir_list.h"
#include "file_util.h"
#include "itk_image_save.h"
#include "itk_image_type.h"
#include "itk_resample.h"
#include "logfile.h"
#include "path_util.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_warp.h"
#include "pointset.h"
#include "print_and_exit.h"
#include "rasterizer.h"
#include "rt_study.h"
#include "rt_study_metadata.h"
#include "rtss.h"
#include "rtss_contour.h"
#include "rtss_roi.h"
#include "segmentation.h"
#include "ss_list_io.h"
#include "ss_img_extract.h"
#include "ss_img_stats.h"
#include "string_util.h"
#include "warp_parms.h"
#include "xio_structures.h"

class Segmentation_private {
public:
    Plm_image::Pointer m_labelmap; /* Structure set lossy bitmap form */
    Plm_image::Pointer m_ss_img;   /* Structure set in lossless bitmap form */
    Rtss::Pointer m_rtss;          /* Structure set in polyline form */

    bool m_rtss_valid;
    bool m_ss_img_valid;

public:
    Segmentation_private () {
        m_rtss_valid = false;
        m_ss_img_valid = false;
    }
    ~Segmentation_private () {
    }
};

static std::string
compose_prefix_fn (
    const std::string& output_prefix, 
    const std::string& structure_name,
    const char* extension
)
{
    return string_format ("%s/%s.%s", 
        output_prefix.c_str(), 
        structure_name.c_str(), 
        extension);
}

Segmentation::Segmentation ()
{
    this->d_ptr = new Segmentation_private;
}

Segmentation::~Segmentation ()
{
    clear ();
    delete this->d_ptr;
}

void
Segmentation::clear ()
{
    d_ptr->m_rtss.reset();
    d_ptr->m_ss_img.reset();
    d_ptr->m_labelmap.reset();
    d_ptr->m_rtss_valid = false;
    d_ptr->m_ss_img_valid = false;
}

void
Segmentation::load (const char *ss_img, const char *ss_list)
{
    /* Load ss_img */
    if (d_ptr->m_ss_img) {
        d_ptr->m_ss_img.reset();
    }
    if (ss_img && file_exists (ss_img)) {
        d_ptr->m_ss_img = plm_image_load_native (ss_img);
    }

    /* GCS FIX: Verify it is a uchar vec */

    /* Load ss_list */
    if (d_ptr->m_rtss) {
        d_ptr->m_rtss.reset();
    }
    if (ss_list && file_exists (ss_list)) {
        lprintf ("Trying to load ss_list: %s\n", ss_list);
        d_ptr->m_rtss.reset (ss_list_load (0, ss_list));
    } else {
        d_ptr->m_rtss.reset (new Rtss);
        int num_structures = d_ptr->m_ss_img->itk_uchar_vec()->GetVectorLength() * 8;
        for (int i = 0; i < num_structures; i++) {
            d_ptr->m_rtss->add_structure ("Unknown Structure", 
                "255 255 0", i+1, i);
        }
    }

    if (d_ptr->m_rtss) {
        d_ptr->m_rtss->free_all_polylines ();
    }
    d_ptr->m_rtss_valid = false;
    d_ptr->m_ss_img_valid = true;
}

void
Segmentation::load_prefix (const std::string& prefix_dir)
{
    this->load_prefix (prefix_dir.c_str());
}

void
Segmentation::load_prefix (const char *prefix_dir)
{
    /* Clear out any existing structures */
    this->clear ();

    /* Load the list of files in the directory */
    Dir_list dl;
    dl.load (prefix_dir);

    /* Make a quick pass through the directory to find the number of 
       files.  This is used to size the ss_img. */
    int max_structures = 0;
    for (int i = 0; i < dl.num_entries; i++) {
        /* Look at filename, make sure it is an mha or nrrd file */
        const char *entry = dl.entries[i];
        if (!Segmentation::valid_extension (entry)) {
            continue;
        }
        max_structures++;
    }
    int out_vec_len = 1 + (max_structures - 1) / 8;
    if (out_vec_len < 2) out_vec_len = 2;

    /* Make a second pass that actually loads the files */
    bool first = true;
    int bit = 0;
    UCharVecImageType::Pointer ss_img;
    Plm_image_header ss_img_pih;
    for (int i = 0; i < dl.num_entries; i++) {
        /* Look at filename, make sure it is an mha or nrrd file */
        const char *entry = dl.entries[i];
        if (!Segmentation::valid_extension (entry)) {
            continue;
        }

        /* Load the file */
        std::string input_fn = string_format ("%s/%s", prefix_dir, entry);
        Plm_image img (input_fn, PLM_IMG_TYPE_ITK_UCHAR);
        Plm_image_header pih (img);

        /* Grab the structure name from the filename */
        char *structure_name = strdup (entry);
        if (extension_is (structure_name, "nii.gz")) {
            strip_extension (structure_name);
        }
        strip_extension (structure_name);
        lprintf ("Loading structure: %s\n", structure_name);

        if (first) {
            this->initialize_ss_image (pih, out_vec_len);

            ss_img = d_ptr->m_ss_img->itk_uchar_vec ();
            Plm_image_header::clone (&ss_img_pih, &pih);

            first = false;
        } else {
            ss_img_pih.print();
            if (!Plm_image_header::compare (&pih, &ss_img_pih)) {
                print_and_exit ("Image size mismatch when loading prefix_dir\n");
            }
        }

        /* Add name to ss_list */
        d_ptr->m_rtss->add_structure (
            structure_name, "", 
            d_ptr->m_rtss->num_structures + 1,
            bit);
        free (structure_name);

        /* GCS FIX: This code is replicated in ss_img_extract */
        unsigned int uchar_no = bit / 8;
        unsigned int bit_no = bit % 8;
        unsigned char bit_mask = 1 << bit_no;
        if (uchar_no > ss_img->GetVectorLength()) {
            print_and_exit ("Error.  Ss_img vector is too small.\n");
        }

        /* Set up iterators for looping through images */
        typedef itk::ImageRegionConstIterator< UCharImageType > 
            UCharIteratorType;
        typedef itk::ImageRegionIterator< UCharVecImageType > 
            UCharVecIteratorType;
        UCharImageType::Pointer uchar_img = img.itk_uchar();
        UCharIteratorType uchar_img_it (uchar_img, 
            uchar_img->GetLargestPossibleRegion());
        UCharVecIteratorType ss_img_it (ss_img, 
            ss_img->GetLargestPossibleRegion());

        /* Loop through voxels, or'ing them into ss_img */
        /* GCS FIX: This is inefficient, due to undesirable construct 
           and destruct of itk::VariableLengthVector of each pixel */
        for (
            uchar_img_it.GoToBegin(), ss_img_it.GoToBegin();
            !uchar_img_it.IsAtEnd();
            ++uchar_img_it, ++ss_img_it
        ) {
            unsigned char u = uchar_img_it.Get ();
            if (!u) continue;

            itk::VariableLengthVector<unsigned char> v 
                = ss_img_it.Get ();
            v[uchar_no] |= bit_mask;
            ss_img_it.Set (v);
        }

        /* Move to next bit */
        bit++;
    }

    if (d_ptr->m_rtss) {
        d_ptr->m_rtss->free_all_polylines ();
    }
    if (first) {
        /* We didn't get any valid images in this directory, so do nothing */
    } else {
        d_ptr->m_rtss_valid = false;
        d_ptr->m_ss_img_valid = true;
    }
}

void
Segmentation::add_structure (
    UCharImageType::Pointer itk_image, 
    const char *structure_name,
    const char *structure_color)
{
    Plm_image_header pih (itk_image);

    /* Allocate image if this is the first structure */
    if (!d_ptr->m_ss_img) {
        this->initialize_ss_image (pih, 2);
    }

    else {
        /* Make sure image size is the same */
        Plm_image_header ss_img_pih (d_ptr->m_ss_img);
        if (!Plm_image_header::compare (&pih, &ss_img_pih)) {
            print_and_exit ("Image size mismatch when adding structure\n");
        }
    }

    /* Figure out basic structure info */
    if (!structure_name) {
        structure_name = "";
    }
    if (!structure_color) {
        structure_color = "";
    }
    int bit = d_ptr->m_rtss->num_structures; /* GCS FIX: I hope this is ok */

    /* Add structure to rtss */
    d_ptr->m_rtss->add_structure (
        structure_name, structure_color,
        d_ptr->m_rtss->num_structures + 1,
        bit);

    /* Set bit within ss_img */
    this->set_structure_image (itk_image, bit);

    if (d_ptr->m_rtss) {
        d_ptr->m_rtss->free_all_polylines ();
    }
    d_ptr->m_rtss_valid = false;
    d_ptr->m_ss_img_valid = true;
}

Rtss_roi *
Segmentation::add_rtss_roi (
    const char *structure_name,
    const char *structure_color)
{
    /* Allocate rtss if first time called */
    if (!d_ptr->m_rtss_valid) {
        /* GCS FIX: In principle, I should convert existing ss_image 
           planes into rtss format first */
        d_ptr->m_rtss = Rtss::New();
        d_ptr->m_ss_img = Plm_image::Pointer();
        d_ptr->m_rtss_valid = true;
        d_ptr->m_ss_img_valid = false;
    }

    /* Figure out basic structure info */
    if (!structure_name) {
        structure_name = "";
    }
    if (!structure_color) {
        structure_color = "";
    }
    int bit = d_ptr->m_rtss->num_structures;

    /* Add structure to rtss */
    Rtss_roi *rtss_roi = d_ptr->m_rtss->add_structure (
        structure_name, structure_color,
        d_ptr->m_rtss->num_structures + 1,
        bit);
    return rtss_roi;
}

void
Segmentation::load_cxt (const std::string& input_fn, Rt_study_metadata *rsm)
{
    d_ptr->m_rtss = Rtss::New();
    cxt_load (d_ptr->m_rtss.get(), rsm, input_fn.c_str());

    d_ptr->m_rtss_valid = true;
    d_ptr->m_ss_img_valid = false;
}

void
Segmentation::load_xio (const Xio_studyset& studyset)
{
    d_ptr->m_rtss = Rtss::New();
    lprintf ("calling xio_structures_load\n");
    xio_structures_load (d_ptr->m_rtss.get(), studyset);

    d_ptr->m_rtss_valid = true;
    d_ptr->m_ss_img_valid = false;
}

bool
Segmentation::valid_extension (const char *filename)
{
    return extension_is (filename, ".mha") 
        || extension_is (filename, ".mhd")
        || extension_is (filename, ".nii")
        || extension_is (filename, ".nii.gz")
        || extension_is (filename, ".nrrd");
}

size_t
Segmentation::get_num_structures ()
{
    if (d_ptr->m_rtss) {
        return d_ptr->m_rtss->num_structures;
    }
    return 0;
}

std::string
Segmentation::get_structure_name (size_t index)
{
    if (d_ptr->m_rtss) {
        return d_ptr->m_rtss->get_structure_name (index);
    }
    return 0;
}

void 
Segmentation::set_structure_name (size_t index, const std::string& name)
{
    if (!d_ptr->m_rtss) {
        return;
    }
    d_ptr->m_rtss->set_structure_name (index, name);
}

UCharImageType::Pointer
Segmentation::get_structure_image (int index)
{
    if (!d_ptr->m_ss_img) {
        print_and_exit (
            "Error extracting unknown structure image (no ssi %d)\n", index);
    }

    if (!d_ptr->m_rtss) {
        print_and_exit (
            "Error extracting unknown structure image (no cxt %d)\n", index);
    }

    Rtss_roi *curr_structure = d_ptr->m_rtss->slist[index];
    int bit = curr_structure->bit;

    if (bit == -1) {
        print_and_exit (
            "Error extracting unknown structure image (no bit %d)\n", index);
    }
    UCharImageType::Pointer prefix_img 
        = ss_img_extract_bit (d_ptr->m_ss_img, bit);

    return prefix_img;
}

void
Segmentation::save_colormap (const std::string& colormap_fn)
{
    ss_list_save_colormap (d_ptr->m_rtss.get(), colormap_fn.c_str());
}

void
Segmentation::save_cxt (
    const Rt_study_metadata::Pointer& rsm, 
    const std::string& cxt_fn, 
    bool prune_empty
)
{
    cxt_save (d_ptr->m_rtss.get(), rsm, cxt_fn.c_str(), prune_empty);
}

void
Segmentation::save_fcsv (
    const Rtss_roi *curr_structure, 
    const std::string& fn
)
{
    Labeled_pointset pointset;

    for (size_t j = 0; j < curr_structure->num_contours; j++) {
        Rtss_contour *curr_polyline = curr_structure->pslist[j];
        for (size_t k = 0; k < curr_polyline->num_vertices; k++) {
            pointset.insert_lps ("", curr_polyline->x[k],
                curr_polyline->y[k], curr_polyline->z[k]);
        }
    }

    pointset.save_fcsv (fn);
}

void
Segmentation::save_prefix_fcsv (const std::string& output_prefix)
{
    if (!d_ptr->m_rtss) {
        print_and_exit (
            "Error: save_prefix_fcsv() tried to save a RTSS without a CXT\n");
    }

    for (size_t i = 0; i < d_ptr->m_rtss->num_structures; i++)
    {
        Rtss_roi *curr_structure = d_ptr->m_rtss->slist[i];

        std::string fn = 
            compose_prefix_fn (output_prefix, curr_structure->name, "fcsv");
        save_fcsv (curr_structure, fn);
    }
}

void
Segmentation::save_ss_image (const std::string& ss_img_fn)
{
    if (!d_ptr->m_ss_img) {
        print_and_exit (
            "Error: save_ss_image() tried to write a non-existant ss_img\n");
    }
    if (d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_GPUIT_UCHAR_VEC
        || d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_ITK_UCHAR_VEC) 
    {
        /* Image type must be uchar vector */
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
    }
    else {
        /* Image type must be uint32_t */
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
    }

    d_ptr->m_ss_img->save_image (ss_img_fn);
}

void
Segmentation::save_labelmap (const std::string& labelmap_fn)
{
    d_ptr->m_labelmap->save_image (labelmap_fn);
}

unsigned int
get_opt4d_flattened_index_from_3d_indices(itk::Index<3> index, itk::Size<3> image_size) {
    auto col = index[0];
    auto row = index[1];
    auto slice = index[2];
    auto ncols = image_size[0];
    auto nrows = image_size[1];
    auto nslices = image_size[2];
    return col + slice * ncols + row * ncols * nslices;
}

void
Segmentation::save_opt4d(const std::string &opt4d_prefix) {
    auto file_name = opt4d_prefix + ".voi";
    std::ofstream voi_file(file_name, std::ios::binary);
    if (!voi_file.is_open()) {
        throw std::runtime_error("Could not open output .voi file for writing.");
    }

    // Initialize the entire volume as air
    auto dims = d_ptr->m_rtss->m_dim;
    auto num_structures = d_ptr->m_rtss->num_structures;
    auto num_voxels = dims[0] * dims[1] * dims[2];
    std::vector<uint8_t> voi(num_voxels, 127);

    std::vector<std::vector<unsigned int>> structure_indices_list(num_structures);

    for (size_t i = 0; i < num_structures; i++) {
        // Extract an ITK image for the structure
        Rtss_roi *curr_structure = d_ptr->m_rtss->slist[i];
        int bit = curr_structure->bit;
        if (bit == -1) continue;
        UCharImageType::Pointer prefix_img
                = ss_img_extract_bit(d_ptr->m_ss_img, bit);

        // Get indices which belong to the structure
        itk::ImageRegionConstIteratorWithIndex<UCharImageType> it(prefix_img, prefix_img->GetLargestPossibleRegion());
        auto size = prefix_img->GetLargestPossibleRegion().GetSize();
        std::vector<unsigned int> indices;
        for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
            if ((unsigned int) it.Get() == 1) {
                indices.push_back(get_opt4d_flattened_index_from_3d_indices(it.GetIndex(), size));
            }
        }
        std::sort(indices.begin(), indices.end());
        structure_indices_list[i] = indices;

        // Set the voxels in the structure to the structure index
        for (const auto index: indices) {
            voi[index] = i;
        }
    }

    // Write the voxels to file. Opt4D expects each voxel to be a single byte
    for (auto i = 0; i < num_voxels; i++) {
        voi_file.write((char *) &voi[i], sizeof(char));
    }
    voi_file.close();


    // Now let's try to write the .vv file
    std::string opt4d_vv_fn = opt4d_prefix + ".vv";
    std::ofstream vv_file(opt4d_vv_fn, std::ios::binary);
    if (!vv_file.is_open()) {
        throw std::runtime_error("Could not open output file");
    }

    // Write the .vv file header
    std::string title = "vv-opt4D";
    // When written as binary, this magic number will be "ab" or "ba" depending
    // on system endianness
    short magic_number = 25185;
    char array[2];
    *((short *) array) = magic_number;
    std::string header_line_2 = "#VOIs:";

    vv_file << title << array << "\n";

    auto num_structures_string = std::to_string(num_structures);
    vv_file << header_line_2 << std::setfill('0') << std::setw(5) << num_structures_string << "\n";

    for (size_t i = 0; i < d_ptr->m_rtss->num_structures; i++) {
        vv_file << std::setfill('0') << std::setw(5)
                << std::to_string(i)
                << " "
                << d_ptr->m_rtss->get_structure_name(i)
                << "\n";
    }

    // Use run-length encoding to write the vv file
    for (const auto &indices: structure_indices_list) {
        auto num_voxels = indices.size();
        vv_file.write((char*)(&num_voxels), sizeof(int));
        if (indices.empty()) continue;

        std::vector<unsigned int>::const_iterator iter = indices.begin();
        while (iter != indices.end()) {

            unsigned int range_start = *iter;
            unsigned char range_length = 0;

            do {
                ++range_length;
                ++iter;
            } while ((iter != indices.end()) && (*iter == (range_start + range_length)) && (range_length < 0xFF));
            vv_file.write((char*)(&range_start), sizeof(range_start));
            vv_file.write((char*)(&range_length), sizeof(range_length));
        }
    }
    vv_file.close();
}



void
Segmentation::save_prefix (const std::string &output_prefix,
    const std::string& extension)
{
    if (!d_ptr->m_ss_img) {
        return;
    }

    if (!d_ptr->m_rtss) {
        printf ("WTF???\n");
    }

    for (size_t i = 0; i < d_ptr->m_rtss->num_structures; i++)
    {
        std::string fn;
        Rtss_roi *curr_structure = d_ptr->m_rtss->slist[i];
        int bit = curr_structure->bit;

        if (bit == -1) continue;
        UCharImageType::Pointer prefix_img 
            = ss_img_extract_bit (d_ptr->m_ss_img, bit);

        fn = string_format ("%s/%s.%s", 
            output_prefix.c_str(),
            curr_structure->name.c_str(),
            extension.c_str());
        itk_image_save (prefix_img, fn.c_str());
    }
}

void
Segmentation::save_prefix (const char *output_prefix)
{
    this->save_prefix (std::string (output_prefix));
}

void
Segmentation::save_ss_list (const std::string& ss_list_fn)
{
    ss_list_save (d_ptr->m_rtss.get(), ss_list_fn.c_str());
}

void
Segmentation::save_xio (
    const Rt_study_metadata::Pointer& rsm,
    Xio_ct_transform *xio_transform, 
    Xio_version xio_version, 
    const std::string &output_dir
)
{
    xio_structures_save (rsm, d_ptr->m_rtss.get(), xio_transform,
        xio_version, output_dir.c_str());
}

UInt32ImageType::Pointer
Segmentation::get_ss_img_uint32 (void)
{
    if (!d_ptr->m_ss_img) {
        print_and_exit ("Sorry, can't get_ss_img()\n");
    }
    d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
    return d_ptr->m_ss_img->m_itk_uint32;
}

UCharVecImageType::Pointer
Segmentation::get_ss_img_uchar_vec (void)
{
    if (!d_ptr->m_ss_img) {
        print_and_exit ("Sorry, can't get_ss_img()\n");
    }
    d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
    return d_ptr->m_ss_img->m_itk_uchar_vec;
}

void
Segmentation::apply_dicom_dir (const Rt_study_metadata::Pointer& rsm)
{
    if (!d_ptr->m_rtss) {
        return;
    }

    if (!rsm || !rsm->slice_list_complete()) {
        return;
    }

    d_ptr->m_rtss->apply_slice_list (rsm);
}

void
Segmentation::convert_ss_img_to_cxt (void)
{
    /* Only convert if ss_img found */
    if (!d_ptr->m_ss_img) {
        return;
    }

    /* Allocate memory for cxt */
    bool use_existing_bits;
    if (d_ptr->m_rtss) {
        use_existing_bits = true;
    }
    else {
        d_ptr->m_rtss = Rtss::New();
        use_existing_bits = false;
    }

    /* Copy geometry from ss_img to cxt */
    d_ptr->m_rtss->set_geometry (d_ptr->m_ss_img);

    if (d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_GPUIT_UCHAR_VEC
        || d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_ITK_UCHAR_VEC) 
    {
        /* Image type must be uchar vector */
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);

        /* Do extraction */
        lprintf ("Doing extraction\n");
        ::cxt_extract (d_ptr->m_rtss.get(), d_ptr->m_ss_img->m_itk_uchar_vec, 
            -1, use_existing_bits);
    }
    else {
        /* Image type must be uint32_t */
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);

        /* Do extraction -- this branch is used by plastimatch synth */
        lprintf ("Doing extraction\n");
        ::cxt_extract (d_ptr->m_rtss.get(), d_ptr->m_ss_img->m_itk_uint32, -1, 
            use_existing_bits);
    }

    d_ptr->m_rtss_valid = true;
}

void
Segmentation::convert_to_uchar_vec (void)
{
    if (!d_ptr->m_ss_img) {
        print_and_exit (
            "Error: convert_to_uchar_vec() requires an image\n");
    }
    d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
}

void
Segmentation::cxt_extract (void)
{
    if (d_ptr->m_ss_img && !d_ptr->m_rtss_valid) {
        this->convert_ss_img_to_cxt ();
    }
}

void
Segmentation::cxt_re_extract (void)
{
    d_ptr->m_rtss->free_all_polylines ();
    if (d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_GPUIT_UCHAR_VEC
        || d_ptr->m_ss_img->m_type == PLM_IMG_TYPE_ITK_UCHAR_VEC) 
    {
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_UCHAR_VEC);
        ::cxt_extract (d_ptr->m_rtss.get(), d_ptr->m_ss_img->m_itk_uchar_vec, 
            d_ptr->m_rtss->num_structures, true);
    }
    else {
        d_ptr->m_ss_img->convert (PLM_IMG_TYPE_ITK_ULONG);
        ::cxt_extract (d_ptr->m_rtss.get(), d_ptr->m_ss_img->m_itk_uint32, 
            d_ptr->m_rtss->num_structures, true);
    }

    d_ptr->m_rtss_valid = true;
}

void
Segmentation::prune_empty (void)
{
    if (d_ptr->m_rtss && d_ptr->m_rtss_valid) {
        d_ptr->m_rtss->prune_empty ();
    }
}

void
Segmentation::keyholize ()
{
    if (d_ptr->m_rtss && d_ptr->m_rtss_valid) {
        d_ptr->m_rtss->keyholize ();
    }
}

void
Segmentation::rasterize (
    Plm_image_header *pih,
    bool want_labelmap,
    bool xor_overlapping
)
{
    /* Rasterize structure sets */
    Rasterizer rasterizer;

    bool use_ss_img_vec = true;

    printf ("Rasterizing...\n");
    rasterizer.rasterize (d_ptr->m_rtss.get(), pih, false, want_labelmap, true,
        use_ss_img_vec, xor_overlapping);

    /* Convert rasterized structure sets from vol to plm_image */
    printf ("Converting...\n");
    if (want_labelmap) {
        d_ptr->m_labelmap = Plm_image::New();
        d_ptr->m_labelmap->set_volume (rasterizer.labelmap_vol);
        rasterizer.labelmap_vol = 0;
    }
    d_ptr->m_ss_img = Plm_image::New();

    if (use_ss_img_vec) {
        d_ptr->m_ss_img->set_itk (rasterizer.m_ss_img->m_itk_uchar_vec);
    }
    else {
        Volume::Pointer v = rasterizer.m_ss_img->get_volume();
        d_ptr->m_ss_img->set_volume (v);
    }
    lprintf ("Finished rasterization.\n");

    d_ptr->m_ss_img_valid = true;
}

void
Segmentation::set_geometry (const Plm_image_header *pih)
{
    if (d_ptr->m_rtss) {
        d_ptr->m_rtss->set_geometry (pih);
    }
}

void
Segmentation::find_rasterization_geometry (Plm_image_header *pih)
{
    if (d_ptr->m_rtss) {
        d_ptr->m_rtss->find_rasterization_geometry (pih);
    }
}

Segmentation::Pointer 
Segmentation::warp_nondestructive (
    const Xform::Pointer& xf, 
    Plm_image_header *pih, 
    bool use_itk) const
{
    Segmentation::Pointer rtss_warped = Segmentation::New ();

    rtss_warped->d_ptr->m_rtss = Rtss::New (
        Rtss::clone_empty (0, d_ptr->m_rtss.get()));
    rtss_warped->d_ptr->m_rtss_valid = false;

    if (d_ptr->m_labelmap) {
        printf ("Warping labelmap.\n");
        Plm_image::Pointer tmp = Plm_image::New();
        plm_warp (tmp, 0, xf, pih, d_ptr->m_labelmap, 0, false, use_itk, 0);
        rtss_warped->d_ptr->m_labelmap = tmp;
        rtss_warped->d_ptr->m_labelmap->convert (PLM_IMG_TYPE_ITK_ULONG);
    }

    if (d_ptr->m_ss_img) {
        printf ("Warping ss_img.\n");
        Plm_image::Pointer tmp = Plm_image::New();
        plm_warp (tmp, 0, xf, pih, d_ptr->m_ss_img, 0, false, use_itk, 0);
        rtss_warped->d_ptr->m_ss_img = tmp;
    }

    return rtss_warped;
}

void
Segmentation::warp (
    const Xform::Pointer& xf, 
    Plm_image_header *pih, 
    bool use_itk)
{
    if (d_ptr->m_labelmap) {
        printf ("Warping labelmap.\n");
        Plm_image::Pointer tmp = Plm_image::New();
        plm_warp (tmp, 0, xf, pih, d_ptr->m_labelmap, 0, false, use_itk, 0);
        d_ptr->m_labelmap = tmp;
        d_ptr->m_labelmap->convert (PLM_IMG_TYPE_ITK_ULONG);
    }

    if (d_ptr->m_ss_img) {
        printf ("Warping ss_img.\n");
        Plm_image::Pointer tmp = Plm_image::New();
        plm_warp (tmp, 0, xf, pih, d_ptr->m_ss_img, 0, false, use_itk, 0);
        d_ptr->m_ss_img = tmp;
    }

    /* The cxt polylines are now obsolete */
    if (d_ptr->m_rtss) {
        d_ptr->m_rtss->free_all_polylines ();
    }
    d_ptr->m_rtss_valid = false;
}

void
Segmentation::warp (
    const Xform::Pointer& xf, 
    Plm_image_header *pih, 
    Warp_parms *parms)
{
    this->warp (xf, pih, parms->use_itk);
}

bool
Segmentation::have_ss_img ()
{
    /* GCS 2021-05-04.  Hope this is ok... */
    // return (bool) d_ptr->m_ss_img;
    return d_ptr->m_ss_img_valid;
}

void
Segmentation::set_ss_img (UCharImageType::Pointer ss_img)
{
    d_ptr->m_ss_img = Plm_image::New();
    d_ptr->m_ss_img->set_itk (ss_img);

    if (d_ptr->m_rtss) {
        d_ptr->m_rtss->free_all_polylines ();
    }
    d_ptr->m_rtss_valid = false;
    d_ptr->m_ss_img_valid = true;
}

Plm_image::Pointer
Segmentation::get_ss_img ()
{
    return d_ptr->m_ss_img;
}

bool
Segmentation::have_structure_set ()
{
    /* GCS 2021-05-04.  Hope this is ok... */
    //return (bool) d_ptr->m_rtss;
    return d_ptr->m_rtss_valid;
}

Rtss::Pointer&
Segmentation::get_structure_set ()
{
    return d_ptr->m_rtss;
}

Rtss *
Segmentation::get_structure_set_raw ()
{
    return d_ptr->m_rtss.get();
}

void
Segmentation::set_structure_set (Rtss::Pointer& rtss_ss)
{
    d_ptr->m_rtss = rtss_ss;

    d_ptr->m_rtss_valid = true;
    d_ptr->m_ss_img_valid = false;
}

void
Segmentation::set_structure_set (Rtss *rtss_ss)
{
    d_ptr->m_rtss.reset (rtss_ss);

    d_ptr->m_rtss_valid = true;
    d_ptr->m_ss_img_valid = false;
}

void
Segmentation::set_structure_image (
    UCharImageType::Pointer uchar_img, 
    unsigned int bit
)
{
    /* Figure out which bit of which byte to change */
    unsigned int uchar_no = bit / 8;
    unsigned int bit_no = bit % 8;
    unsigned char bit_mask = 1 << bit_no;

    /* Expand vector length if needed */
    UCharVecImageType::Pointer ss_img = d_ptr->m_ss_img->itk_uchar_vec ();
    if ((uchar_no+1) > ss_img->GetVectorLength()) {
        lprintf ("Broadening ss_image\n");
        this->broaden_ss_image (uchar_no+1);
    }

    /* Set up iterators for looping through images */
    typedef itk::ImageRegionConstIterator< UCharImageType > 
        UCharIteratorType;
    typedef itk::ImageRegionIterator< UCharVecImageType > 
        UCharVecIteratorType;
    UCharIteratorType uchar_img_it (uchar_img, 
        uchar_img->GetLargestPossibleRegion());
    UCharVecIteratorType ss_img_it (ss_img, 
        ss_img->GetLargestPossibleRegion());

    /* Loop through voxels, or'ing them into ss_img */
    /* GCS FIX: This is inefficient, due to undesirable construct 
       and destruct of itk::VariableLengthVector of each pixel */
    for (uchar_img_it.GoToBegin(), ss_img_it.GoToBegin();
        !uchar_img_it.IsAtEnd();
        ++uchar_img_it, ++ss_img_it
    ) {
        unsigned char u = uchar_img_it.Get ();
        if (!u) continue;

        itk::VariableLengthVector<unsigned char> v 
            = ss_img_it.Get ();
        v[uchar_no] |= bit_mask;
        ss_img_it.Set (v);
    }
}

void
Segmentation::resample (float spacing[3])
{
    d_ptr->m_ss_img->set_itk (
        resample_image (d_ptr->m_ss_img->itk_uchar_vec (), spacing));
}


/* -----------------------------------------------------------------------
   Protected member functions
   ----------------------------------------------------------------------- */
void
Segmentation::initialize_ss_image (
    const Plm_image_header& pih, int vector_length)
{
    UCharVecImageType::Pointer ss_img;
    Plm_image_header ss_img_pih;

    /* Create ss_image with same resolution as first image */
    d_ptr->m_ss_img = Plm_image::New ();
    ss_img = UCharVecImageType::New ();
    itk_image_set_header (ss_img, pih);
    ss_img->SetVectorLength (vector_length);
    ss_img->Allocate ();

    /* GCS NOTE: For some reason, ss_img->FillBuffer (0) 
       doesn't do what I want. */
    itk::VariableLengthVector<unsigned char> v;
    v.SetSize (vector_length);
    v.Fill (0);
    ss_img->FillBuffer (v);

    d_ptr->m_ss_img->set_itk (ss_img);
    Plm_image_header::clone (&ss_img_pih, &pih);

    /* Create ss_list to hold strucure names */
    d_ptr->m_rtss = Rtss::New();
    d_ptr->m_rtss->set_geometry (d_ptr->m_ss_img);
}

void
Segmentation::broaden_ss_image (int new_vector_length)
{
    /* Get old image */
    UCharVecImageType::Pointer old_ss_img = d_ptr->m_ss_img->itk_uchar_vec ();
    Plm_image_header pih (old_ss_img);

    /* Create new image */
    UCharVecImageType::Pointer new_ss_img = UCharVecImageType::New ();
    itk_image_set_header (new_ss_img, pih);
    new_ss_img->SetVectorLength (new_vector_length);
    new_ss_img->Allocate ();

    /* Create "pixels" */
    itk::VariableLengthVector<unsigned char> v_old;
    itk::VariableLengthVector<unsigned char> v_new;
    int old_vector_length = old_ss_img->GetVectorLength();
    v_old.SetSize (old_vector_length);
    v_new.SetSize (new_vector_length);
    v_new.Fill (0);

    /* Loop through image */
    typedef itk::ImageRegionIterator< 
        UCharVecImageType > UCharVecIteratorType;
    UCharVecIteratorType it_old (
        old_ss_img, old_ss_img->GetLargestPossibleRegion());
    UCharVecIteratorType it_new (
        new_ss_img, new_ss_img->GetLargestPossibleRegion());
    for (it_old.GoToBegin(), it_new.GoToBegin(); 
         !it_old.IsAtEnd(); 
         ++it_old, ++it_new)
    {
        /* Copy old pixel bytes into new */
        v_old = it_old.Get();
        for (int i = 0; i < old_vector_length; i++) {
            v_new[i] = v_old[i];
        }
        it_new.Set (v_new);
    }

    /* Fixate new image */
    d_ptr->m_ss_img->set_itk (new_ss_img);
}
