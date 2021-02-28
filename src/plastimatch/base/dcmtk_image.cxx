/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <climits>
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"
#include "dcmtk/dcmimgle/dcmimage.h"

#include "dcmtk_file.h"
#include "dcmtk_metadata.h"
#include "dcmtk_module.h"
#include "dcmtk_rt_study.h"
#include "dcmtk_rt_study_p.h"
#include "dcmtk_series.h"
#include "dcmtk_slice_data.h"
#include "dcmtk_uid.h"
#include "file_util.h"
#include "logfile.h"
#include "rt_study_metadata.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_math.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "print_and_exit.h"
#include "slice_list.h"
#include "string_util.h"
#include "volume.h"

void
Dcmtk_rt_study::image_load ()
{
    /* Set up outputs */
    Plm_image::Pointer pli = Plm_image::New();
    d_ptr->img = pli;

    /* Make abbreviations */
    Dcmtk_series *ds_image = d_ptr->ds_image;
    const Dcmtk_file_list& ds_flist = ds_image->get_flist ();

    /* Create a container to hold different groups of files */
    std::list<Dcmtk_file_list> group_list;

    /* Arrange files into groups according to direction cosines */
    for (Dcmtk_file_list::const_iterator it = ds_flist.begin();
         it != ds_flist.end(); ++it)
    {
        const Dcmtk_file::Pointer& df = (*it);
        bool match_found = false;
        std::list<Dcmtk_file_list>::iterator grit;
        for (grit = group_list.begin(); grit != group_list.end(); ++grit) {
            Dcmtk_file_list& flp = *grit;
            const Dcmtk_file::Pointer& flp_df = flp.front();

            if (flp_df->get_direction_cosines() 
                == df->get_direction_cosines())
            {
                /* Add logic to append to flp */
                //printf ("Match found.  :)\n");
                match_found = true;
                flp.push_back (df);
                break;
            }
        }
        if (match_found) {
            continue;
        }
        /* Else insert new element into group_list */
        //printf ("Match not found.  :(\n");
        group_list.push_back (Dcmtk_file_list());
        group_list.back().push_back (df);
    }

    /* If multiple groups, emit a warning.  Choose the largest group. */
    Dcmtk_file_list *flist = &group_list.front ();
    if (group_list.size() > 1) {
        lprintf ("Warning, DICOM series with multiple direction cosines\n");
        std::list<Dcmtk_file_list>::iterator grit;
        for (grit = group_list.begin(); grit != group_list.end(); ++grit) {
            if ((*grit).size() > flist->size()) {
                flist = &*grit;
            }
        }
    }
    
    /* Sort group in Z direction */
    flist->sort (dcmtk_file_compare_z_position);

    /* 
     * GCS FIX:
     * Remove minimum 2 slices requirement
     * Check consistency (dim, origin) of images w/in series
     * Different image types
     * Refine slice spacing based on entire chunk size
     */

    /* Check for minimum 2 slices */
    if (flist->size() < 2) {
        return;
    }

    /* Check that every slice image in series have the same dimention */
    std::vector< std::array< size_t, 3 > > dimParams; 
    for (Dcmtk_file_list::iterator it = flist->begin(); it != flist->end(); ++it) {
      const Dcmtk_file* df = (*it).get();
      DicomImage di (df->get_dataset(), EXS_Unknown);
      Volume_header vh (df->get_volume_header());
      plm_long *dim = vh.get_dim();
    
      unsigned long length = di.getOutputDataSize(8);
      dimParams.push_back({ size_t(length), size_t(dim[0]), size_t(dim[1])});
    }
    for ( size_t i = 1; i < dimParams.size(); ++i) {
      std::array< size_t, 3 >& sliceDim = dimParams[i];
      if (sliceDim[0] != dimParams[0][0] 
        || sliceDim[1] != dimParams[0][1] 
        || sliceDim[2] != dimParams[0][2]) {
        
        print_and_exit ("Error. Wrong slice dimention, comparing to the previous ones.\n"
          "Possibly data corrupted or partially downloaded! Please, check the data!\n"
          "Corrupted slice index is %u\n"
          "Correct slice data must be (%d vs. %d x %d)\n"  
          "Corrupted slice data is (%d vs. %d x %d).\n", i, 
          dimParams[0][0], dimParams[0][1], dimParams[0][2],
          sliceDim[0], sliceDim[1], sliceDim[2]);
      }
    }

    /* Get first slice */
    const Dcmtk_file* df = (*flist->begin()).get();
    
    /* Store UIDs */
    if (d_ptr->rt_study_metadata) {
        d_ptr->rt_study_metadata->set_ct_series_uid (
            df->get_cstr (DCM_SeriesInstanceUID));
        d_ptr->rt_study_metadata->set_frame_of_reference_uid (
            df->get_cstr (DCM_FrameOfReferenceUID));
        d_ptr->rt_study_metadata->set_study_uid (
            df->get_cstr (DCM_StudyInstanceUID));
        d_ptr->rt_study_metadata->set_study_date (
            df->get_cstr (DCM_StudyDate));
        d_ptr->rt_study_metadata->set_study_time (
            df->get_cstr (DCM_StudyTime));
        d_ptr->rt_study_metadata->set_study_id (
            df->get_cstr (DCM_StudyID));

        /* Store remaining metadata */
        Metadata::Pointer& study_metadata = d_ptr->rt_study_metadata->get_study_metadata ();
        dcmtk_copy_into_metadata (study_metadata, df, DCM_PatientName);
        dcmtk_copy_into_metadata (study_metadata, df, DCM_PatientID);
        dcmtk_copy_into_metadata (study_metadata, df, DCM_PatientSex);
        dcmtk_copy_into_metadata (study_metadata, df, DCM_PatientPosition);
        dcmtk_copy_into_metadata (study_metadata, df, DCM_StudyID);

        Metadata::Pointer& image_metadata = d_ptr->rt_study_metadata->get_image_metadata ();
        dcmtk_copy_into_metadata (image_metadata, df, DCM_Modality);
        dcmtk_copy_into_metadata (image_metadata, df, DCM_InstanceCreationDate);
        dcmtk_copy_into_metadata (image_metadata, df, DCM_InstanceCreationTime);
        dcmtk_copy_into_metadata (image_metadata, df, DCM_SeriesDescription);
    }

    /* Divine image type */
    uint16_t samp_per_pix, bits_alloc, bits_stored, high_bit, pixel_rep;
    const char* phot_interp;
    bool rc = df->get_uint16 (DCM_SamplesPerPixel, &samp_per_pix);
    if (!rc) {
        return;
    }
    phot_interp = df->get_cstr (DCM_PhotometricInterpretation);
    if (!phot_interp) {
        return;
    }
    rc = df->get_uint16 (DCM_BitsAllocated, &bits_alloc);
    if (!rc) {
        return;
    }
    rc = df->get_uint16 (DCM_BitsStored, &bits_stored);
    if (!rc) {
        return;
    }
    rc = df->get_uint16 (DCM_HighBit, &high_bit);
    if (!rc) {
        return;
    }
    rc = df->get_uint16 (DCM_PixelRepresentation, &pixel_rep);
    if (!rc) {
        return;
    }

#if defined (commentout)
    lprintf ("Samp_per_pix: %d\n", (int) samp_per_pix);
    lprintf ("Phot_interp: %s\n", phot_interp);
    lprintf ("Bits_alloc: %d\n", (int) bits_alloc);
    lprintf ("Bits_stored: %d\n", (int) bits_stored);
    lprintf ("High_bit: %d\n", (int) high_bit);
    lprintf ("Pixel_rep: %d\n", (int) pixel_rep);
#endif
    
    /* Some kinds of images we don't know how to deal with.  
       Don't load these. */
    if (samp_per_pix != 1) {
        lprintf ("Sorry, couldn't load image: samp_per_pix\n");
        return;
    }
    if (strcmp (phot_interp, "MONOCHROME2")) {
        lprintf ("Sorry, couldn't load image: phot_interp\n");
        return;
    }
    if (bits_alloc != 16 && bits_alloc != 8) {
        lprintf ("Sorry, couldn't load image: bits_alloc\n");
        return;
    }
    if (bits_stored != high_bit + 1) {
        lprintf ("Sorry, couldn't load image: bits_stored/high_bit\n");
        return;
    }
    if (pixel_rep != 0 && pixel_rep != 1) {
        lprintf ("Sorry, couldn't load image: pixel_rep\n");
        return;
    }

    /* If PLM_CONFIG_VOL_LIST is enabled, the image will be loaded 
       into a PLM_IMG_TYPE_GPUIT_LIST */
#if (PLM_CONFIG_VOL_LIST)

    /* Get first slice of first chunk */
    Dcmtk_file_list::iterator it = flist->begin();
    df = it->get();

    float z_init, z_prev, z_diff, z_last;
    int slice_no = 0;
    float best_chunk_z_start = z_init = z_prev = df->get_z_position ();

    /* Get second slice to form first chunk */
    ++it; ++slice_no;
    df = (*it).get();
    z_diff = df->get_z_position() - z_prev;
    z_last = z_prev = df->get_z_position();

    /* We want to find the number and spacing for each chunk 
       within the group.  These are used to set the dim and 
       spacing of the volume. */
    int this_chunk_start = 0, best_chunk_start = 0;
    float this_chunk_diff = z_diff, best_chunk_diff = z_diff;
    int this_chunk_len = 2, best_chunk_len = 2;

    /* Loop through remaining slices */
    while (++it != flist->end())
    {
        ++slice_no;
        df = (*it).get();
        z_diff = df->get_z_position() - z_prev;
        z_last = z_prev = df->get_z_position();

        if (fabs (this_chunk_diff - z_diff) > 0.11) {
            /* Start a new chunk if difference in thickness is 
               more than 0.1 millimeter */
            this_chunk_start = slice_no - 1;
            this_chunk_len = 2;
            this_chunk_diff = z_diff;
        } else {
            /* Same thickness, increase size of this chunk */
            this_chunk_diff = ((this_chunk_len * this_chunk_diff) + z_diff)
                / (this_chunk_len + 1);
            this_chunk_len++;

            /* Check if this chunk is now the best chunk */
            if (this_chunk_len > best_chunk_len) {
                best_chunk_start = this_chunk_start;
                best_chunk_len = this_chunk_len;
                best_chunk_diff = this_chunk_diff;
                best_chunk_z_start = z_prev 
                    - (best_chunk_len-1) * best_chunk_diff;
            }
        }
    }
    
#if defined (commentout)        
    Dcmtk_file_list& flp = *grit;
    const Dcmtk_file::Pointer dfp = grit->front();
    Volume::Pointer vol = Volume::New (
        const plm_long dim[3], 
        const float origin[3], 
        const float spacing[3], 
        &dfp->get_direction_cosines(),
        vh,
        PT_FLOAT, 1);
#endif

#else /* NOT VOL_LIST */

    /* We want to find the largest chunk (most slices) with equal spacing.  
       This will be used to resample in the case of irregular spacing. */

    /* Get first slice of first chunk */
    float z_init, z_prev, z_diff, z_last;
    int slice_no = 0;
    float best_chunk_z_start = z_init = z_prev = df->get_z_position ();
    const Dcmtk_file* df_this_chunk_start = df;
    const Dcmtk_file* df_best_chunk_start = df;

    /* Get second slice to form first chunk */
    std::list<Dcmtk_file::Pointer>::const_iterator it = flist->begin();
    ++it; ++slice_no;
    df = (*it).get();
    z_diff = df->get_z_position() - z_prev;
    z_last = z_prev = df->get_z_position();
    int this_chunk_start = 0, best_chunk_start = 0;
    float this_chunk_diff = z_diff, best_chunk_diff = z_diff;
    size_t this_chunk_len = 2, best_chunk_len = 2;

    /* Loop through remaining slices */
    while (++it != flist->end())
    {
	++slice_no;
	df = (*it).get();
	z_diff = df->get_z_position() - z_prev;
	z_last = z_prev = df->get_z_position();
        // lprintf ("Slice %d, z_pos %f\n", slice_no, df->get_z_position());

	if (fabs (this_chunk_diff - z_diff) > 0.11) {
	    /* Start a new chunk if difference in thickness is 
	       more than 0.1 millimeter */
	    this_chunk_start = slice_no - 1;
	    this_chunk_len = 2;
	    this_chunk_diff = z_diff;
            df_this_chunk_start = std::prev(it)->get();
	} else {
	    /* Same thickness, increase size of this chunk */
	    this_chunk_diff = ((this_chunk_len * this_chunk_diff) + z_diff)
		/ (this_chunk_len + 1);
	    this_chunk_len++;

	    /* Check if this chunk is now the best chunk */
	    if (this_chunk_len > best_chunk_len) {
		best_chunk_start = this_chunk_start;
		best_chunk_len = this_chunk_len;
		best_chunk_diff = this_chunk_diff;
		best_chunk_z_start = z_prev 
		    - (best_chunk_len-1) * best_chunk_diff;
                df_best_chunk_start = df_this_chunk_start;
	    }
	}
    }

    /* Report information about best chunk */
    if (best_chunk_len != flist->size()) {
        lprintf ("** Warning, inequal slice spacing detected when loading DICOM.\n");
        lprintf ("Best chunck:\n  Slices %d to %d from (0 to %d)\n"
            "  Z_loc = %f %f\n" 
            "  Slice spacing = %f\n", 
            best_chunk_start, best_chunk_start + best_chunk_len - 1, slice_no, 
            best_chunk_z_start, 
            best_chunk_z_start + (best_chunk_len - 1) * best_chunk_diff, 
            best_chunk_diff);
    }
    
#if defined (commentout)
    /* Some debugging info */
    lprintf ("Slices: ");
    for (it = flist->begin(); it != flist->end(); ++it) {
        df = (*it).get();
	lprintf ("%f ", df->get_z_position());
    }
    lprintf ("\n");
#endif

    /* Create a Volume_header to hold the final, resampled, image geometry */
    Volume_header vh (df_best_chunk_start->get_volume_header());
    // vh.print();
    const float *dc = vh.get_direction_cosines();
    plm_long *dim = vh.get_dim();
    float *spacing = vh.get_spacing();
    float *origin = vh.get_origin();
    int slices_before = 
	ROUND_INT ((best_chunk_z_start - z_init) / best_chunk_diff);
    // printf ("bczs: %f, zi: %f, bcd: %f\n", best_chunk_z_start, z_init, best_chunk_diff);
    // printf ("slices_before: %d\n", slices_before);
    int slices_after = 
	ROUND_INT ((z_last - best_chunk_z_start 
		- (best_chunk_len - 1) * best_chunk_diff) / best_chunk_diff);
    // printf ("zl: %f, bczs: %f, bcl: %d, bcd: %d\n", z_last, best_chunk_z_start, best_chunk_len, best_chunk_diff);
    // printf ("slices_after: %d\n", slices_after);
    dim[2] = slices_before + best_chunk_len + slices_after;
    spacing[2] = best_chunk_diff;
    float origin_diff[3] = {
        slices_before * best_chunk_diff * dc[2],
        slices_before * best_chunk_diff * dc[5],
        slices_before * best_chunk_diff * dc[8] };
    vec3_sub2 (origin, origin_diff);

    /* Store image header */
    if (d_ptr->rt_study_metadata) {
        d_ptr->rt_study_metadata->set_image_header (Plm_image_header (vh));
    }

    /* More debugging info */
    //vh.print ();

    /* Still more debugging info */
#if defined (commentout)
    lprintf ("Resamples slices: ");
    for (plm_long i = 0; i < dim[2]; i++) {
	lprintf ("%f ", vh.get_origin()[2] + i * vh.get_spacing()[2]);
    }
    lprintf ("\n");
#endif

    pli->m_type = PLM_IMG_TYPE_GPUIT_FLOAT;
    pli->m_original_type = PLM_IMG_TYPE_GPUIT_FLOAT;
    Volume* vol = new Volume (vh, PT_FLOAT, 1);

    pli->set_volume (vol);
    float* img = (float*) vol->img;

    float origin_z_pos = origin[0] * dc[2] + origin[1] * dc[5]
        + origin[2] * dc[8];
    for (plm_long i = 0; i < dim[2]; i++) {
	/* Find the best slice, using nearest neighbor interpolation */
	std::list<Dcmtk_file::Pointer>::const_iterator best_slice_it 
            = flist->begin();
	float best_z_dist = FLT_MAX;
	float z_pos = origin_z_pos + i * spacing[2];
	for (it = flist->begin(); it != flist->end(); ++it) {
	    float this_z_dist = fabs ((*it)->get_z_position() - z_pos);
	    if (this_z_dist < best_z_dist) {
		best_z_dist = this_z_dist;
		best_slice_it = it;
	    }
	}
	/* Load the slice image data into volume */
	df = (*best_slice_it).get();

#if defined (commentout)
	lprintf ("Loading slice z=%f at location z=%f\n",
	    (*best_slice_it)->get_z_position(), z_pos);
#endif

        /* Using DicomImage makes it easy to support compressed and other 
           types of images.  But I can't figure out how to make it 
           render signed Pixel Data with Pixel Represenation of 1.
           cf. http://support.dcmtk.org/docs/mod_dcmimage.html */
//#define USE_DICOMIMAGE 1
	const uint8_t* pixel_data_8;
	const uint16_t* pixel_data_16;
	unsigned long length = 0;
        
#if defined (USE_DICOMIMAGE)
        DicomImage di (df->get_dataset(), EXS_Unknown);
        length = di.getOutputDataSize(8);
        if (((long) length) != dim[0] * dim[1]) {
            print_and_exit ("Oops.  Dicom image had wrong length "
                "(%d vs. %d x %d).\n", length, dim[0], dim[1]);
        }
        if (di.getStatus() != EIS_Normal) {
            printf ("Status is: %s\n", di.getString (di.getStatus()));
            print_and_exit ("Oops.  DicomImage status is not EIS_Normal.\n");
        }
        //const void* pixel_data = di.getInterData()->getData();
        if (bits_alloc == 8) {
            pixel_data_8 = (const uint8_t*) di.getOutputData (bits_stored);
            //pixel_data_8 = (const uint8_t*) pixel_data;
            if (!pixel_data_8) {
                print_and_exit ("Oops.  Error reading pixel data.  Punting.\n");
            }
        } else if (bits_alloc == 16) {
            pixel_data_16 = (const uint16_t*) di.getOutputData (bits_stored);
            //pixel_data_16 = (const uint16_t*) pixel_data;
            if (!pixel_data_16) {
                print_and_exit ("Oops.  Error reading pixel data.  Punting.\n");
            }
        }
#else
        df->get_dataset()->chooseRepresentation (EXS_LittleEndianExplicit, nullptr);
        rc = 0;
        if (bits_alloc == 8) {
            rc = df->get_uint8_array (DCM_PixelData, &pixel_data_8, &length);
        } else if (bits_alloc == 16) {
            rc = df->get_uint16_array (DCM_PixelData, &pixel_data_16, &length);
        }
	if (!rc) {
	    print_and_exit ("Oops.  Error reading pixel data.  Punting.\n");
	}
	if (((long) length) != dim[0] * dim[1]) {
	    print_and_exit ("Oops.  Dicom image had wrong length "
		"(%d vs. %d x %d).\n", length, dim[0], dim[1]);
	}
#endif /* USE_DICOMIMAGE */

        float rescale_slope, rescale_intercept;
        rc = df->get_ds_float (DCM_RescaleIntercept, &rescale_intercept);
        if (!rc) {
            rescale_intercept = 0;
        }
        rc = df->get_ds_float (DCM_RescaleSlope, &rescale_slope);
        if (!rc) {
            rescale_slope = 1;
        }

#if defined (commentout)
        lprintf ("S/I = %f/%f\n", rescale_slope, rescale_intercept);
#endif
        /* Apply slope and offset */
        for (plm_long j = 0; j < (plm_long) length; j++) {
            if (bits_alloc == 8 && pixel_rep == 0) {
                img[j] = rescale_slope * (uint8_t) pixel_data_8[j] 
                    + rescale_intercept;
            }
            else if (bits_alloc == 8 && pixel_rep == 1) {
                img[j] = rescale_slope * (int8_t) pixel_data_8[j] 
                    + rescale_intercept;
            }
            else if (bits_alloc == 16 && pixel_rep == 0) {
                img[j] = rescale_slope * (uint16_t) pixel_data_16[j] 
                    + rescale_intercept;
            }
            else if (bits_alloc == 16 && pixel_rep == 1) {
                img[j] = rescale_slope * (int16_t) pixel_data_16[j] 
                    + rescale_intercept;
            }
        }
	img += length;
        
	/* Store slice UID */
        if (d_ptr->rt_study_metadata) {
            d_ptr->rt_study_metadata->set_slice_uid (i, df->get_cstr (DCM_SOPInstanceUID));
        }
    }
#endif /* NOT VOL_LIST */
    if (d_ptr->rt_study_metadata) {
        d_ptr->rt_study_metadata->set_slice_list_complete ();
    }
}

static void
dcmtk_save_slice (const Rt_study_metadata::Pointer rsm, Dcmtk_slice_data *dsd)
{
    std::string tmp;
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();
    Metadata::Pointer image_metadata;
    if (rsm) {
        image_metadata = rsm->get_image_metadata ();
    }

    /* Patient, and General Study modules */
    Dcmtk_module::set_patient (dataset, image_metadata);
    Dcmtk_module::set_general_study (dataset, rsm);
    
    /* Get modality */
    std::string modality = "CT";
    if (image_metadata) {
        std::string metadata_modality = image_metadata->get_metadata (
            DCM_Modality.getGroup(), DCM_Modality.getElement());
        if (metadata_modality != "") {
            modality = metadata_modality;
        }
    }

    /* General Series module */
    Dcmtk_module::set_general_series (dataset, image_metadata,
        modality.c_str());
    dataset->putAndInsertString (DCM_SeriesInstanceUID, 
        rsm->get_ct_series_uid());
    dataset->putAndInsertString (DCM_SeriesDescription, 
        rsm->get_ct_series_description());

    /* Frame of Reference module */
    Dcmtk_module::set_frame_of_reference (dataset, rsm);
    /* XVI 4.5 requires a DCM_PositionReferenceIndicator */
    dataset->putAndInsertString (DCM_PositionReferenceIndicator, "SP");

    /* General equipment module */
    Dcmtk_module::set_general_equipment (dataset, image_metadata);

    /* General Image module */
    tmp = string_format ("%d", dsd->instance_no);
    dataset->putAndInsertString (DCM_InstanceNumber, tmp.c_str());
    if (modality == "CT") {
        dataset->putAndInsertString (DCM_ImageType, 
            "DERIVED\\SECONDARY\\AXIAL");
    } else {
        dataset->putAndInsertString (DCM_ImageType, 
            "DERIVED\\SECONDARY\\REFORMATTED");
    }

    /* Image Plane module */
    tmp = string_format ("%f\\%f", dsd->vol->spacing[1], dsd->vol->spacing[0]);
    dataset->putAndInsertString (DCM_PixelSpacing, tmp.c_str());
    dataset->putAndInsertString (DCM_ImageOrientationPatient, dsd->iop.c_str());
    dataset->putAndInsertString (DCM_ImagePositionPatient, dsd->ipp.c_str());
    dataset->putAndInsertString (DCM_SliceThickness, dsd->sthk.c_str());
    dataset->putAndInsertString (DCM_SliceLocation, dsd->sloc.c_str());

    /* Image Pixel module */
    dataset->putAndInsertString (DCM_SamplesPerPixel, "1");
    dataset->putAndInsertString (DCM_PhotometricInterpretation, "MONOCHROME2");
    dataset->putAndInsertUint16 (DCM_Rows, (Uint16) dsd->vol->dim[1]);
    dataset->putAndInsertUint16 (DCM_Columns, (Uint16) dsd->vol->dim[0]);
    dataset->putAndInsertString (DCM_BitsAllocated, "16");
    dataset->putAndInsertString (DCM_BitsStored, "16");
    dataset->putAndInsertString (DCM_HighBit, "15");
    dataset->putAndInsertString (DCM_PixelRepresentation, "1");
    /* Convert to 16-bit signed int */
    for (size_t i = 0; i < dsd->slice_size; i++) {
        float f = dsd->slice_float[i];
        dsd->slice_int16[i] = (int16_t) 
            ((f - dsd->intercept) / dsd->slope);
    }
    dataset->putAndInsertUint16Array (DCM_PixelData, 
        (Uint16*) dsd->slice_int16, dsd->slice_size);

    /* CT Image module */
    tmp = string_format ("%f", dsd->intercept);
    dataset->putAndInsertString (DCM_RescaleIntercept, tmp.c_str());
    tmp = string_format ("%f", dsd->slope);
    dataset->putAndInsertString (DCM_RescaleSlope, tmp.c_str());
    dataset->putAndInsertString (DCM_RescaleType, "HU");
    dataset->putAndInsertString (DCM_KVP, "");
    dataset->putAndInsertString (DCM_AcquisitionNumber, "");

    /* VOI LUT module */
    if (modality == "CT") {
        dcmtk_copy_from_metadata (dataset, image_metadata,
            DCM_WindowCenter, "40");
        dcmtk_copy_from_metadata (dataset, image_metadata,
            DCM_WindowWidth, "400");
    }

    /* SOP Common Module */
    if (modality == "MR") {
        dataset->putAndInsertString (
            DCM_SOPClassUID, UID_MRImageStorage);
    }
    else if (modality == "PT") {
        dataset->putAndInsertString (
            DCM_SOPClassUID, UID_NuclearMedicineImageStorage);
    }
    else if (modality == "US") {
        dataset->putAndInsertString (
            DCM_SOPClassUID, UID_UltrasoundImageStorage);
    }
    else {
        dataset->putAndInsertString (
            DCM_SOPClassUID, UID_CTImageStorage);
    }
    dataset->putAndInsertString (DCM_SOPInstanceUID, dsd->slice_uid);
    if (image_metadata->get_metadata (DCM_InstanceCreationDate) != "") {
        dataset->putAndInsertOFStringArray(DCM_InstanceCreationDate, 
            image_metadata->get_metadata(DCM_InstanceCreationDate).c_str());
    } else {
        dataset->putAndInsertOFStringArray(DCM_InstanceCreationDate, 
            rsm->get_study_date());
    }
    if (image_metadata->get_metadata (DCM_InstanceCreationTime) != "") {
        dataset->putAndInsertOFStringArray(DCM_InstanceCreationTime, 
            image_metadata->get_metadata(DCM_InstanceCreationTime).c_str());
    } else {
        dataset->putAndInsertOFStringArray(DCM_InstanceCreationTime, 
            rsm->get_study_time());
    }

    /* Write the output file */
    OFCondition status = fileformat.saveFile (dsd->fn.c_str(), 
        EXS_LittleEndianExplicit);
    if (status.bad()) {
        print_and_exit ("Error: cannot write DICOM file (%s)\n", 
            status.text());
    }
}

void
Dcmtk_rt_study::image_save (
    const char *dicom_dir)
{
    Dcmtk_slice_data dsd;
    dsd.vol = this->get_image_volume_float();
    dsd.slice_size = dsd.vol->dim[0] * dsd.vol->dim[1];
    dsd.slice_int16 = new int16_t[dsd.slice_size];
    float *dc = dsd.vol->get_direction_matrix();
    dsd.iop = string_format ("%f\\%f\\%f\\%f\\%f\\%f",
        dc[0], dc[3], dc[6], dc[1], dc[4], dc[7]);

    Plm_image_header pih (dsd.vol.get());
    d_ptr->rt_study_metadata->set_image_header (pih);

    /* Find slope / offset on a per-volume basis */
    const Rt_study_metadata::Pointer& rsm = d_ptr->rt_study_metadata;
    const Metadata::Pointer& image_metadata = rsm->get_image_metadata ();
    const std::string& meta_intercept
        = image_metadata->get_metadata (DCM_RescaleIntercept);
    const std::string& meta_slope
        = image_metadata->get_metadata (DCM_RescaleSlope);
    
    float vol_min = FLT_MAX;
    float vol_max = - FLT_MAX;
    float *img = (float*) dsd.vol->img;
    bool all_integers = true;
    for (plm_long v = 0; v < dsd.vol->npix; v++) {
        if (vol_min > img[v]) {
            vol_min = img[v];
        }
        else if (vol_max < img[v]) {
            vol_max = img[v];
        }
        if (img[v] != floorf(img[v])) {
            all_integers = false;
        }
    }
        if (vol_max - vol_min > UINT16_MAX) {
            all_integers = false;
        }
    /* Use a heuristic to determine intercept and offset.
       The heuristic is designed around the following principles:
       - prevent underflow when using low-precision DICOM string-encoded
         floating point numbers
       - map integers to integers
       - keep integer values within range
    */
    if (meta_intercept != "") {
        int rc = sscanf (meta_intercept.c_str(), "%f", &dsd.intercept);
        if (rc != 1) {
            dsd.intercept = floorf (vol_min);
        }
    } else {
        dsd.intercept = floorf (vol_min);
    }
    if (meta_slope != "") {
        int rc = sscanf (meta_slope.c_str(), "%f", &dsd.slope);
        if (rc != 1) {
            dsd.slope = 1;
        }
    }
    else if (all_integers) {
        dsd.slope = 1;
        if (vol_max - vol_min > INT16_MAX) {
            dsd.intercept = vol_max - INT16_MAX;
        }
    }
    else {
        float range = vol_max - dsd.intercept;
        if (range < 1) {
            range = 1;
        }
        dsd.slope = range / (SHRT_MAX - 100);
        std::string tmp = string_format ("%f", dsd.slope);
        sscanf (tmp.c_str(), "%f", &dsd.slope);
    }

    for (plm_long k = 0; k < dsd.vol->dim[2]; k++) {
        /* GCS FIX #2:  This is possibly correct.  Not 100% sure. */
        float z_loc = dsd.vol->origin[2] + dc[8] * k * dsd.vol->spacing[2];
        dsd.instance_no = k;
        dsd.sthk = string_format ("%f", dsd.vol->spacing[2]);
        dsd.sloc = string_format ("%f", z_loc);
        /* GCS FIX #2:  "Ditto" */
        dsd.ipp = string_format ("%f\\%f\\%f", 
            dsd.vol->origin[0] + dc[2] * k * dsd.vol->spacing[2],
            dsd.vol->origin[1] + dc[5] * k * dsd.vol->spacing[2],
            dsd.vol->origin[2] + dc[8] * k * dsd.vol->spacing[2]);
        dcmtk_uid (dsd.slice_uid, PLM_UID_PREFIX);

        dsd.slice_float = &((float*)dsd.vol->img)[k*dsd.slice_size];

        /* Format filename and prepare output directory */
        if (d_ptr->filenames_with_uid) {
            dsd.fn = string_format ("%s/image%04d_%s.dcm", dicom_dir, (int) k,
                dsd.slice_uid);
        } else {
            dsd.fn = string_format ("%s/image%04d.dcm", dicom_dir, (int) k);
        }
        make_parent_directories (dsd.fn);

        /* Fix the uid into the metadata */
        d_ptr->rt_study_metadata->set_slice_uid (k, dsd.slice_uid);

        /* Save the file to disk */
        dcmtk_save_slice (d_ptr->rt_study_metadata, &dsd);
    }
    delete[] dsd.slice_int16;
    d_ptr->rt_study_metadata->set_slice_list_complete ();
}
