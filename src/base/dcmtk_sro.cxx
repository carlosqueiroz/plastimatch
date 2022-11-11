/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_metadata.h"
#include "dcmtk_module.h"
#include "dcmtk_sro.h"
#include "dicom_util.h"
#include "file_util.h"
#include "logfile.h"
#include "plm_uid_prefix.h"
#include "print_and_exit.h"
#include "string_util.h"
#include "xform.h"

void
Dcmtk_sro::save (
    const Xform::Pointer& xf,
    const Rt_study_metadata::Pointer& rsm_fixed,
    const Rt_study_metadata::Pointer& rsm_moving,
    const std::string& dicom_dir,
    bool filenames_with_uid)
{
    /* Prepare xform */
    Xform xf_cvt;
    bool is_linear = xf->is_linear();
    if (is_linear) {
        xform_to_aff (&xf_cvt, xf.get(), 0);
    }
    else {
        xform_to_gpuit_vf (&xf_cvt, xf.get(), 0);
    }
    
    /* Prepare dcmtk */
    OFCondition ofc;
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();

    if (!rsm_fixed || !rsm_moving) {
        print_and_exit ("Sorry, anonymous spatial registration objects "
            "are not yet supported.\n");
    }
    Metadata::Pointer study_meta = rsm_fixed->get_study_metadata ();
    Metadata::Pointer sro_meta = rsm_fixed->get_sro_metadata ();

    /* Add entries for common modules */
    Dcmtk_module::set_patient (dataset, study_meta);
    Dcmtk_module::set_sop_common (dataset);

    /* General Study module */
    Dcmtk_module::set_general_study (dataset, rsm_fixed);
    dataset->putAndInsertOFStringArray (DCM_StudyDate,
        rsm_moving->get_study_date());
    dataset->putAndInsertOFStringArray (DCM_StudyTime,
        rsm_moving->get_study_time());

    dataset->putAndInsertOFStringArray (DCM_InstanceCreationDate,
        rsm_moving->get_study_date());
    dataset->putAndInsertOFStringArray (DCM_InstanceCreationTime,
        rsm_moving->get_study_time());
    dataset->putAndInsertString (DCM_StudyInstanceUID,
        rsm_fixed->get_study_uid().c_str());

    /* General Series module */
    Dcmtk_module::set_general_series (dataset, sro_meta, "REG");

    /* Spatial Registration Series module */
    /* (nothing to do) */

    /* Frame of Reference module.  The direction of the transform is opposite 
       between spatial Registration and Deforamble Spatial Registration IODs, 
       See Section 17 Part O. 
       http://dicom.nema.org/medical/Dicom/current/output/chtml/part17/chapter_O.html
       However, this implementation keeps the fixed image as 
       the Registered RCS, and the moving image be the Source RCS, 
       whether the registration is linear or deformable.
    */
    Dcmtk_module::set_frame_of_reference (dataset, rsm_fixed);

    /* General Equipment module */
    Dcmtk_module::set_general_equipment (dataset, study_meta);

    /* These do not seem to be needed,  */
#if defined (commentout)
    dataset->putAndInsertString (DCM_InstanceNumber, "");
    dataset->putAndInsertString (DCM_ContentLabel, "");
#endif

    /* Spatial Registration module or Deformable Spatial Registration 
       module */
    dataset->putAndInsertOFStringArray (DCM_ContentDate, 
        rsm_moving->get_study_date());
    dataset->putAndInsertOFStringArray (DCM_ContentTime, 
        rsm_moving->get_study_time());
    if (is_linear) {
        AffineTransformType::Pointer itk_aff;
        itk_aff = xf_cvt.get_aff();

        /* fixed image */
        DcmItem *reg_item = 0;
        dataset->findOrCreateSequenceItem (
            DCM_RegistrationSequence, reg_item, -2);
        reg_item->putAndInsertString (
            DCM_FrameOfReferenceUID, 
            rsm_fixed->get_frame_of_reference_uid().c_str());
        DcmItem *mr_item = 0;
        reg_item->findOrCreateSequenceItem (
            DCM_MatrixRegistrationSequence, mr_item, -2);
        DcmItem *rtc_item = 0;
        mr_item->findOrCreateSequenceItem (
            DCM_RegistrationTypeCodeSequence, rtc_item, -2);
        rtc_item->putAndInsertString (DCM_CodeValue, "125025");
        rtc_item->putAndInsertString (DCM_CodingSchemeDesignator, "DCM");
        rtc_item->putAndInsertString (DCM_CodeMeaning, "Visual Alignment");
        DcmItem *m_item = 0;
        mr_item->findOrCreateSequenceItem (DCM_MatrixSequence, m_item, -2);
        m_item->putAndInsertString (DCM_FrameOfReferenceTransformationMatrix,
            "1.0\\0.0\\0.0\\0.0\\0.0\\1.0\\0.0\\0.0\\"
            "0.0\\0.0\\1.0\\0.0\\0.0\\0.0\\0.0\\1.0");
        m_item->putAndInsertString (
            DCM_FrameOfReferenceTransformationMatrixType, "RIGID");

        /* moving image */
        dataset->findOrCreateSequenceItem (
            DCM_RegistrationSequence, reg_item, -2);
        reg_item->putAndInsertString (
            DCM_FrameOfReferenceUID, 
            rsm_moving->get_frame_of_reference_uid().c_str());
        reg_item->findOrCreateSequenceItem (
            DCM_MatrixRegistrationSequence, mr_item, -2);
        mr_item->findOrCreateSequenceItem (
            DCM_RegistrationTypeCodeSequence, rtc_item, -2);
        rtc_item->putAndInsertString (DCM_CodeValue, "125025");
        rtc_item->putAndInsertString (DCM_CodingSchemeDesignator, "DCM");
        rtc_item->putAndInsertString (DCM_CodeMeaning, "Visual Alignment");
        mr_item->findOrCreateSequenceItem (DCM_MatrixSequence, m_item, -2);
        std::string matrix_string;

        /* Invert the matrix, as per Section 17 Part O. */
        const AffineTransformType::MatrixType& itk_aff_mat 
            = itk_aff->GetMatrix ();
        const AffineTransformType::OutputVectorType& itk_aff_off 
            = itk_aff->GetOffset ();

        printf ("ITK_AFF_OFF\n%f %f %f\n",
            itk_aff_off[0], itk_aff_off[1], itk_aff_off[2]);
    
        /* Nb. ITK does not easily create an inverse affine transform. 
           Therefore we play with the matrices. */
        /* GCS FIX: The above comment is likely no longer true.  We 
           probably want to switch to using the ITK function. */
        vnl_matrix_fixed< double, 3, 3 > itk_aff_mat_inv =
            itk_aff_mat.GetInverse();
    
        matrix_string = string_format (
            "%f\\%f\\%f\\%f\\"
            "%f\\%f\\%f\\%f\\"
            "%f\\%f\\%f\\%f\\"
            "0.0\\0.0\\0.0\\1.0",
            itk_aff_mat_inv[0][0],
            itk_aff_mat_inv[0][1],
            itk_aff_mat_inv[0][2],
            - itk_aff_mat_inv[0][0] * itk_aff_off[0]
            - itk_aff_mat_inv[0][1] * itk_aff_off[1]
            - itk_aff_mat_inv[0][2] * itk_aff_off[2],
            itk_aff_mat_inv[1][0],
            itk_aff_mat_inv[1][1],
            itk_aff_mat_inv[1][2],
            - itk_aff_mat_inv[1][0] * itk_aff_off[0]
            - itk_aff_mat_inv[1][1] * itk_aff_off[1]
            - itk_aff_mat_inv[1][2] * itk_aff_off[2],
            itk_aff_mat_inv[2][0],
            itk_aff_mat_inv[2][1],
            itk_aff_mat_inv[2][2],
            - itk_aff_mat_inv[2][0] * itk_aff_off[0]
            - itk_aff_mat_inv[2][1] * itk_aff_off[1]
            - itk_aff_mat_inv[2][2] * itk_aff_off[2]
        );
        m_item->putAndInsertString (DCM_FrameOfReferenceTransformationMatrix,
            matrix_string.c_str());
        m_item->putAndInsertString (
            DCM_FrameOfReferenceTransformationMatrixType, "RIGID");

        printf ("SRO\n%s\n", matrix_string.c_str());

    }
    else {
        DcmItem *reg_item = 0;
        dataset->findOrCreateSequenceItem (
            DCM_DeformableRegistrationSequence, reg_item, -2);
        reg_item->putAndInsertString (
            DCM_SourceFrameOfReferenceUID, 
            rsm_moving->get_frame_of_reference_uid().c_str());

        /* For now, punt on Referenced Image Sequence */

        /* Registration Type Code Sequence (allowed to be empty) */
        DcmItem *m_item = 0;
        reg_item->findOrCreateSequenceItem (
            DCM_RegistrationTypeCodeSequence, m_item, -2);

        /* Pre Deformation Matrix Registration Sequence */
        reg_item->findOrCreateSequenceItem (
            DCM_PreDeformationMatrixRegistrationSequence, m_item, -2);
        m_item->putAndInsertString (DCM_FrameOfReferenceTransformationMatrix,
            "1.0\\0.0\\0.0\\0.0\\0.0\\1.0\\0.0\\0.0\\"
            "0.0\\0.0\\1.0\\0.0\\0.0\\0.0\\0.0\\1.0");
        m_item->putAndInsertString (
            DCM_FrameOfReferenceTransformationMatrixType, "RIGID");

        /* Post Deformation Matrix Registration Sequence */
        reg_item->findOrCreateSequenceItem (
            DCM_PostDeformationMatrixRegistrationSequence, m_item, -2);
        m_item->putAndInsertString (DCM_FrameOfReferenceTransformationMatrix,
            "1.0\\0.0\\0.0\\0.0\\0.0\\1.0\\0.0\\0.0\\"
            "0.0\\0.0\\1.0\\0.0\\0.0\\0.0\\0.0\\1.0");
        m_item->putAndInsertString (
            DCM_FrameOfReferenceTransformationMatrixType, "RIGID");

        /* Deformable Registration Grid Sequence */
        Volume::Pointer vf = xf_cvt.get_gpuit_vf();
        reg_item->findOrCreateSequenceItem (
            DCM_DeformableRegistrationGridSequence, m_item, -2);
        float *dc = vf->get_direction_matrix();
        std::string s = string_format ("%f\\%f\\%f\\%f\\%f\\%f",
            dc[0], dc[3], dc[6], dc[1], dc[4], dc[7]);
        m_item->putAndInsertString (DCM_ImageOrientationPatient, s.c_str());
        s = string_format ("%f\\%f\\%f", 
            vf->origin[0], vf->origin[1], vf->origin[2]);
        m_item->putAndInsertString (DCM_ImagePositionPatient, s.c_str());
        s = string_format ("%d\\%d\\%d", 
            vf->dim[0], vf->dim[1], vf->dim[2]);
        m_item->putAndInsertString (DCM_GridDimensions, s.c_str());
        s = string_format ("%f\\%f\\%f", 
            vf->spacing[0], vf->spacing[1], vf->spacing[2]);
        m_item->putAndInsertString (DCM_GridResolution, s.c_str());


        DcmFloatingPointSingle *fele;
        Float32 *f = (Float32*) vf->img;
        fele = new DcmFloatingPointSingle (DCM_VectorGridData);
        ofc = fele->putFloat32Array (f, 3*vf->npix);
        ofc = m_item->insert (fele);
    }

    /* Common Instance Reference module */
    /* GCS FIX: Two things to fix.  (1) Moving image may belong to different 
       study; (2) All referenced instances should be included. */
    DcmItem *rss_item = 0;
    DcmItem *ris_item = 0;
    /* moving */
    dataset->findOrCreateSequenceItem (
        DCM_ReferencedSeriesSequence, rss_item, -2);
    rss_item->findOrCreateSequenceItem (
        DCM_ReferencedInstanceSequence, ris_item, -2);
    ris_item->putAndInsertString (DCM_ReferencedSOPClassUID,
        UID_CTImageStorage);
    ris_item->putAndInsertString (DCM_ReferencedSOPInstanceUID,
        rsm_moving->get_slice_uid (0));
    rss_item->putAndInsertString (DCM_SeriesInstanceUID,
        rsm_moving->get_ct_series_uid ());
    /* fixed */
    dataset->findOrCreateSequenceItem (
        DCM_ReferencedSeriesSequence, rss_item, -2);
    rss_item->findOrCreateSequenceItem (
        DCM_ReferencedInstanceSequence, ris_item, -2);
    ris_item->putAndInsertString (DCM_ReferencedSOPClassUID,
        UID_CTImageStorage);
    ris_item->putAndInsertString (DCM_ReferencedSOPInstanceUID,
        rsm_fixed->get_slice_uid (0));
    rss_item->putAndInsertString (DCM_SeriesInstanceUID,
        rsm_fixed->get_ct_series_uid ());
    
    /* SOP Common Module */
    std::string sro_sop_instance_uid = dicom_uid (PLM_UID_PREFIX);
    if (is_linear) {
        dataset->putAndInsertString (DCM_SOPClassUID, 
            UID_SpatialRegistrationStorage);
    } else {
        dataset->putAndInsertString (DCM_SOPClassUID, 
            UID_DeformableSpatialRegistrationStorage);
    }
    dataset->putAndInsertString (DCM_SOPInstanceUID, 
        sro_sop_instance_uid.c_str());
        
    /* ----------------------------------------------------------------- *
     *  Write the output file
     * ----------------------------------------------------------------- */
    std::string sro_fn;
    const char *prefix;
    if (is_linear) {
        prefix = "sro";
    } else {
        prefix = "dro";
    }
    if (filenames_with_uid) {
        sro_fn = string_format ("%s/%s_%s.dcm", dicom_dir.c_str(),
            prefix, sro_sop_instance_uid.c_str());
    } else {
        sro_fn = string_format ("%s/%s.dcm", dicom_dir.c_str(), prefix);
    }
    make_parent_directories (sro_fn);

    lprintf ("Trying to save SRO: %s\n", sro_fn.c_str());
    ofc = fileformat.saveFile (sro_fn.c_str(), EXS_LittleEndianExplicit);
    if (ofc.bad()) {
        print_and_exit (
            "Error: cannot write DICOM Spatial Registration (%s) (%s)\n", 
            sro_fn.c_str(),
            ofc.text());
    }
}
