/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdlib.h>
#include <stdio.h>
#include "dcmtk_config.h"
#include "dcmtk/ofstd/ofstream.h"
#include "dcmtk/dcmdata/dctk.h"

#include "dcmtk_file.h"
#include "dcmtk_metadata.h"
#include "dcmtk_module.h"
#include "dcmtk_rt_study.h"
#include "dcmtk_rt_study_p.h"
#include "dcmtk_rtplan.h"
#include "dcmtk_series.h"
#include "dcmtk_util.h"
#include "file_util.h"
#include "logfile.h"
#include "metadata.h"
#include "plm_uid_prefix.h"
#include "plm_version.h"
#include "print_and_exit.h"
#include "rtplan_control_pt.h"
#include "rtplan_beam.h"
#include "string_util.h"

PLMBASE_C_API bool
dcmtk_rtplan_probe(const char *rtplan_fn)
{
    DcmFileFormat dfile;

    /* Suppress warning messages */
    OFLog::configure(OFLogger::FATAL_LOG_LEVEL);

    OFCondition ofrc = dfile.loadFile(rtplan_fn, EXS_Unknown, EGL_noChange);

    /* Restore error messages -- n.b. dcmtk doesn't have a way to
    query current setting, so I just set to default */
    OFLog::configure(OFLogger::WARN_LOG_LEVEL);

    if (ofrc.bad()) {
        return false;
    }

    const char *c;
    DcmDataset *dset = dfile.getDataset();
    ofrc = dset->findAndGetString(DCM_Modality, c);
    if (ofrc.bad() || !c) {
        return false;
    }

    if (strncmp(c, "RTPLAN", strlen("RTPLAN"))) {
        return false;
    }
    else {
        return true;
    }
}

void
Dcmtk_rt_study::rtplan_load (void)
{
    Dcmtk_series *ds_rtplan = d_ptr->ds_rtplan;
    std::string modality = ds_rtplan->get_modality();
    if (modality != "RTPLAN") {
        print_and_exit("Object modality not RTPLAN; should never happen.\n");
    }
    
    std::string class_uid = ds_rtplan->get_string (DCM_SOPClassUID);
    if (class_uid == UID_RTIonPlanStorage) {
        this->rt_ion_plan_load ();
    } else {
        lprintf ("Warning. Load of rt plan (photon) not supported.\n");
    }
}

void
Dcmtk_rt_study::rt_ion_plan_load (void)
{
    Dcmtk_series *ds_rtplan = d_ptr->ds_rtplan;    

    d_ptr->rtplan = Rtplan::New();

    lprintf("Trying to load rt plan.\n");


    /* FIX: load metadata such as patient name, etc. */
    /*const char *val2 = ds_rtplan->get_cstr(DCM_PatientName);
      const char *val3 = ds_rtplan->get_cstr(DCM_PatientID);*/

    /* Load Beam sequence */

    DcmSequenceOfItems *seq = 0;
    bool rc = ds_rtplan->get_sequence (DCM_IonBeamSequence, seq);
    if (!rc) {
        return;
    }
    unsigned long num_beams = seq->card();
    printf (">> Num beams = %lu\n", num_beams);
    for (unsigned long i = 0; i < num_beams; i++) {
        OFCondition orc;
        unsigned long count;

        long int beam_id = 0;
        Rtplan_beam *curr_beam;

        DcmItem *item = seq->getItem(i);
        orc = item->findAndGetLongInt(DCM_BeamNumber, beam_id);
        if (!orc.good()) {
            continue;
        }

        const char *beam_name;
        orc = item->findAndGetString(DCM_BeamName, beam_name);
        if (!orc.good()) {
            continue;
        }
        curr_beam = d_ptr->rtplan->add_beam (std::string(beam_name), beam_id);

        const float *vsad;
        orc = item->findAndGetFloat32Array (DCM_VirtualSourceAxisDistances,
            vsad, &count);
        if (!orc.good()) {
            lprintf (">> Error: %s\n", orc.text());
            print_and_exit ("DCM_VirtualSourceAxisDistances not found\n");
        }
        if (count != 2) {
            print_and_exit ("DCM_VirtualSourceAxisDistances"
                " should have two entries\n");
        }
        curr_beam->virtual_source_axis_distances[0] = vsad[0];
        curr_beam->virtual_source_axis_distances[1] = vsad[1];

        DcmSequenceOfItems *cp_seq = 0;
        orc = item->findAndGetSequence (DCM_IonControlPointSequence, cp_seq);
        if (!orc.good()) {
            break;
        }
        unsigned long num_cp = cp_seq->card();
        printf (">> Adding beam (%lu cp)\n", num_cp);
        for (unsigned long j = 0; j <num_cp; j++) {
            DcmItem *c_item = cp_seq->getItem(j);
            Rtplan_control_pt* curr_cp = curr_beam->add_control_pt ();

            float iso_pos[3];
            orc = dcmtk_get_ds_float3 (c_item, DCM_IsocenterPosition, iso_pos);
            if (orc.good()) {
                lprintf (">> Isocenter: %f %f %f\n", iso_pos[0], iso_pos[1],
                    iso_pos[2]);
                curr_cp->set_isocenter (iso_pos);
            }

            orc = dcmtk_get_ds_float (c_item, DCM_NominalBeamEnergy,
                &curr_cp->nominal_beam_energy);
            if (orc.good()) {
                lprintf (">> Energy: %f\n", curr_cp->nominal_beam_energy);
                curr_cp->set_isocenter (iso_pos);
            }
            
            orc = dcmtk_get_ds_float (c_item, DCM_GantryAngle,
                &curr_cp->gantry_angle);
            orc = dcmtk_get_ds_float (c_item, DCM_CumulativeMetersetWeight,
                &curr_cp->cumulative_meterset_weight);

            long int num_spots;
            orc = c_item->findAndGetLongInt (DCM_NumberOfScanSpotPositions,
                num_spots);
            if (!orc.good()) {
                print_and_exit ("DCM_NumberOfScanSpotPositions not found\n");
            }
            lprintf (">> %d spots\n", num_spots);

            const float *ss_size;
            orc = c_item->findAndGetFloat32Array (DCM_ScanningSpotSize,
                ss_size, &count);
            if (orc.good()) {
                if (count >= 2) {
                    curr_cp->set_scanning_spot_size (ss_size);
                } else {
                    print_and_exit ("DCM_ScanningSpotSize has wrong multiplicity\n");
                }
            }

            const float *spot_positions;
            const float *spot_weights;
            orc = c_item->findAndGetFloat32Array (DCM_ScanSpotPositionMap,
                spot_positions, &count);
            if (!orc.good()) {
                lprintf (">> Error: %s\n", orc.text());
                print_and_exit ("DCM_ScanSpotPositionMap not found\n");
                break;
            }
            if (count != 2 * num_spots) {
                print_and_exit ("Mismatch in spot position map size\n");
            }

            c_item->findAndGetFloat32Array (DCM_ScanSpotMetersetWeights,
                spot_weights, &count);
            if (!orc.good()) {
                print_and_exit ("DCM_ScanSpotMetersetWeights not found\n");
                break;
            }
            if (count != num_spots) {
                print_and_exit ("Mismatch in spot meterset weight size\n");
            }
            printf (">> Adding cp (%ld spots)\n", num_spots);
            curr_cp->scan_spot_position_map.assign (spot_positions, spot_positions + 2*num_spots);
            curr_cp->scan_spot_meterset_weights.assign (spot_weights, spot_weights + num_spots);
        }

        /* Rtplan_beam has an isocenter field.  We copy the isocenter 
           from the first control point into it. */
        curr_beam->set_isocenter_from_control_points();
    }
}

void
Dcmtk_rt_study::rtplan_save (const char *dicom_dir)
{
    /* Required modules (ref DICOM PS3.3 2016c)
       Patient
       General Study
       RT Series
       Frame of Reference
       General Equipment
       RT General Plan
       SOP Common
    */

    /* Prepare varibles */
    const Rt_study_metadata::Pointer& rsm = d_ptr->rt_study_metadata;
    const Metadata::Pointer& rtplan_metadata = rsm->get_rtplan_metadata ();
    std::string s; // Dummy string
    Rtplan::Pointer& rtplan = d_ptr->rtplan;

    /* Prepare dcmtk */
    OFCondition ofc;
    DcmFileFormat fileformat;
    DcmDataset *dataset = fileformat.getDataset();

    /* Add entries for common modules */
    Dcmtk_module::set_patient (dataset, rsm->get_study_metadata ());
    Dcmtk_module::set_general_study (dataset, rsm);
    Dcmtk_module::set_rt_series (dataset, rtplan_metadata, "RTPLAN");
    Dcmtk_module::set_sop_common (dataset);

    /* Frame of reference module */
    dataset->putAndInsertString (DCM_FrameOfReferenceUID, 
        rsm->get_frame_of_reference_uid().c_str());
    dataset->putAndInsertString (DCM_PositionReferenceIndicator, 
	rsm->get_position_reference_indicator());

    /* General equipment module */
    Dcmtk_module::set_general_equipment (dataset,rtplan_metadata);

    /* RT general plan module */
    dataset->putAndInsertString (DCM_RTPlanLabel,  rtplan->rt_plan_label.c_str());
    dataset->putAndInsertString (DCM_RTPlanName, rtplan->rt_plan_name.c_str());
    //dataset->putAndInsertString (DCM_RTPlanDescription, "This is only a test");
    dataset->putAndInsertString (DCM_RTPlanDate, rtplan->rt_plan_date.c_str());
    dataset->putAndInsertString (DCM_RTPlanTime, rtplan->rt_plan_time.c_str());


    if (rsm->get_rtstruct_instance_uid() == "") {
        dataset->putAndInsertString (DCM_RTPlanGeometry, "TREATMENT_DEVICE");
    } else {
        dataset->putAndInsertString (DCM_RTPlanGeometry, "PATIENT");
        DcmItem *rsss_item = 0;
        dataset->findOrCreateSequenceItem (
            DCM_ReferencedStructureSetSequence, rsss_item, -2);
        dcmtk_put (rsss_item, DCM_ReferencedSOPClassUID,
            UID_RTStructureSetStorage);
        dcmtk_put (rsss_item, DCM_ReferencedSOPInstanceUID,
            rsm->get_rtstruct_instance_uid());
    }

    if (rsm->get_dose_instance_uid() != "") {
        DcmItem *rds_item = 0;
        dataset->findOrCreateSequenceItem (
            DCM_ReferencedDoseSequence, rds_item, -2);
        dcmtk_put (rds_item, DCM_ReferencedSOPClassUID,
            UID_RTDoseStorage);
        dcmtk_put (rds_item, DCM_ReferencedSOPInstanceUID,
            rsm->get_dose_instance_uid());
    }

    /* SOP common module */
    /* GCS TODO: Figure out whether to use Plan or Ion Plan */
    // dataset->putAndInsertString (DCM_SOPClassUID, UID_RTPlanStorage);
    dataset->putAndInsertString (DCM_SOPClassUID, UID_RTIonPlanStorage);
    dcmtk_put (dataset, DCM_SOPInstanceUID, 
        d_ptr->rt_study_metadata->get_plan_instance_uid());
    dataset->putAndInsertString(DCM_InstanceCreationDate, 
        d_ptr->rt_study_metadata->get_study_date());
    dataset->putAndInsertString(DCM_InstanceCreationTime, 
        d_ptr->rt_study_metadata->get_study_time());
 
    /* RT prescription module * GCS TODO */
        
    /* RT tolerance tables */
    if (rtplan->tolerance_table_label != "") {
        DcmItem *iots_item = 0;
        dataset->findOrCreateSequenceItem (
            DCM_IonToleranceTableSequence, iots_item, -2);
        dcmtk_put (iots_item, DCM_ToleranceTableNumber, 0);
        dcmtk_put (iots_item, DCM_ToleranceTableLabel,
            rtplan->tolerance_table_label);
        dcmtk_put (iots_item, DCM_GantryAngleTolerance,
            rtplan->tolerance_gantry_angle);
        dcmtk_put (iots_item, DCM_PatientSupportAngleTolerance,
            rtplan->tolerance_patient_support_angle);
        dcmtk_put (iots_item, DCM_TableTopVerticalPositionTolerance,
            rtplan->tolerance_table_top_vertical);
        dcmtk_put (iots_item, DCM_TableTopLongitudinalPositionTolerance,
            rtplan->tolerance_table_top_longitudinal);
        dcmtk_put (iots_item, DCM_TableTopLateralPositionTolerance,
            rtplan->tolerance_table_top_lateral);
        dcmtk_put (iots_item, DCM_TableTopPitchAngleTolerance,
            rtplan->tolerance_table_top_pitch);
        dcmtk_put (iots_item, DCM_TableTopRollAngleTolerance,
            rtplan->tolerance_table_top_roll);
        dcmtk_put (iots_item, DCM_SnoutPositionTolerance,
            rtplan->tolerance_snout_position);
    }

    /* RT patient setup module */
    DcmItem *ps_item = 0;
    dataset->findOrCreateSequenceItem (
        DCM_PatientSetupSequence, ps_item, -2);
    dcmtk_put (ps_item, DCM_PatientSetupNumber, 1);
    dcmtk_put (ps_item, DCM_PatientPosition, rtplan->patient_position);
    dcmtk_put (ps_item, DCM_PatientSetupLabel, rtplan->patient_setup_label);
    DcmItem *psfds_item = 0;
    ps_item->findOrCreateSequenceItem (
        DCM_FixationDeviceSequence, psfds_item, -2);
    dcmtk_put (psfds_item, DCM_FixationDeviceType, rtplan->fixation_device_type);
    dcmtk_put (psfds_item, DCM_FixationDeviceLabel, rtplan->fixation_device_label);
    dcmtk_put (psfds_item, DCM_FixationDeviceDescription, rtplan->fixation_device_description);

    /* RT fraction scheme module */
    DcmItem *fgs_item = 0;
    dataset->findOrCreateSequenceItem (
        DCM_FractionGroupSequence, fgs_item, -2);
    dcmtk_put (fgs_item, DCM_FractionGroupNumber, 0);
    dcmtk_put (fgs_item, DCM_NumberOfFractionsPlanned,
        rtplan->number_of_fractions_planned);
    dcmtk_put (fgs_item, DCM_NumberOfBeams, rtplan->beamlist.size());
    dcmtk_put (fgs_item, DCM_FractionGroupDescription, rtplan->fraction_group_description.c_str());
    dcmtk_put (fgs_item, DCM_NumberOfFractionPatternDigitsPerDay, 
	       rtplan->number_of_fraction_pattern_digits_per_day.c_str());
    dcmtk_put (fgs_item, DCM_RepeatFractionCycleLength, rtplan->repeat_fraction_cycle_length.c_str());
    dcmtk_put (fgs_item, DCM_FractionPattern, rtplan->fraction_pattern.c_str());
    for (size_t b = 0; b < rtplan->beamlist.size(); b++) {
        DcmItem *rbs_item = 0;
        fgs_item->findOrCreateSequenceItem (
            DCM_ReferencedBeamSequence, rbs_item, -2);
        dcmtk_put (rbs_item, DCM_ReferencedBeamNumber, b+1);
        Rtplan_beam *beam = rtplan->beamlist[b];
        dcmtk_put (rbs_item, DCM_BeamMeterset, 
            beam->final_cumulative_meterset_weight);
	dcmtk_put (rbs_item, DCM_BeamDose,
	    beam->beam_dose);
    }
    dcmtk_put (fgs_item, DCM_NumberOfBrachyApplicationSetups, 0);

    /* RT ion beams module */
    for (size_t b = 0; b < rtplan->beamlist.size(); b++) {
        DcmItem *ib_item = 0;
        dataset->findOrCreateSequenceItem (
            DCM_IonBeamSequence, ib_item, -2);
        s = PLM_to_string (b+1);
        ib_item->putAndInsertString (DCM_BeamNumber, s.c_str());

        Rtplan_beam *beam = rtplan->beamlist[b];
        ib_item->putAndInsertString (DCM_BeamName, beam->name.c_str());
        ib_item->putAndInsertString (DCM_BeamDescription,
            beam->description.c_str());
        ib_item->putAndInsertString (DCM_BeamType, "STATIC");
        ib_item->putAndInsertString (DCM_RadiationType, "PROTON");
        ib_item->putAndInsertString (DCM_ScanMode, "MODULATED");
#if defined (commentout)
        ib_item->putAndInsertString (DCM_ScanMode, "MODULATED_SPEC");
        ib_item->putAndInsertString (DCM_ModulatedScanModeType,
            "STATIONARY");
#endif
        ib_item->putAndInsertString (DCM_TreatmentMachineName,
	    beam->treatment_machine_name.c_str());
        ib_item->putAndInsertString (DCM_Manufacturer,
	    beam->manufacturer.c_str());
        ib_item->putAndInsertString (DCM_InstitutionName,
	    beam->institution_name.c_str());
        ib_item->putAndInsertString (DCM_InstitutionAddress,
	    beam->institution_address.c_str());
        ib_item->putAndInsertString (DCM_InstitutionalDepartmentName,
	    beam->institutional_department_name.c_str());
        ib_item->putAndInsertString (DCM_ManufacturerModelName,
	    beam->manufacturer_model_name.c_str());
        ib_item->putAndInsertString (DCM_PrimaryDosimeterUnit, "NP");
        if (rtplan->tolerance_table_label != "") {
            dcmtk_put (ib_item, DCM_ReferencedToleranceTableNumber, 0);
        }
	if (rtplan->patient_position != "") {
            dcmtk_put (ib_item, DCM_ReferencedPatientSetupNumber, 1);
	}
        s = string_format ("%f\\%f", 
            beam->virtual_source_axis_distances[0],
            beam->virtual_source_axis_distances[1]);
        ib_item->putAndInsertString (DCM_VirtualSourceAxisDistances, s.c_str());
        ib_item->putAndInsertString (DCM_TreatmentDeliveryType,
	    beam->treatment_delivery_type.c_str());
        ib_item->putAndInsertString (DCM_NumberOfWedges, "0");
        ib_item->putAndInsertString (DCM_NumberOfCompensators, "0");
        ib_item->putAndInsertString (DCM_NumberOfBoli, "0");
        ib_item->putAndInsertString (DCM_NumberOfBlocks, "0");
        if (rtplan->snout_id != "") {
            DcmItem *snout_item = 0;
            ib_item->findOrCreateSequenceItem (
                DCM_SnoutSequence, snout_item, -2);
            snout_item->putAndInsertString (DCM_SnoutID,
                rtplan->snout_id.c_str());
        }
        if (rtplan->general_accessory_id != "") {
            DcmItem *ga_item = 0;
            ib_item->findOrCreateSequenceItem (
                DCM_GeneralAccessorySequence, ga_item, -2);
            ga_item->putAndInsertString (DCM_GeneralAccessoryNumber, "0");
            ga_item->putAndInsertString (DCM_GeneralAccessoryID,
                rtplan->general_accessory_id.c_str());
            ga_item->putAndInsertString (DCM_AccessoryCode,
                rtplan->general_accessory_code.c_str());
        }
        if (rtplan->range_shifter_id != "") {
            DcmItem *rs_item = 0;
            ib_item->findOrCreateSequenceItem (
                DCM_RangeShifterSequence, rs_item, -2);
            rs_item->putAndInsertString (DCM_RangeShifterNumber, 
                rtplan->range_shifter_number.c_str());
            rs_item->putAndInsertString (DCM_RangeShifterID,
                rtplan->range_shifter_id.c_str());
            rs_item->putAndInsertString (DCM_AccessoryCode,
                rtplan->range_shifter_code.c_str());
            rs_item->putAndInsertString (DCM_RangeShifterType,
                rtplan->range_shifter_type.c_str());
        }
        if (rtplan->range_modulator_id != "") {
            DcmItem *rm_item = 0;
            ib_item->findOrCreateSequenceItem (
                DCM_RangeModulatorSequence, rm_item, -2);
            rm_item->putAndInsertString (DCM_RangeModulatorNumber, "0");
            rm_item->putAndInsertString (DCM_RangeModulatorID,
                rtplan->range_modulator_id.c_str());
            rm_item->putAndInsertString (DCM_AccessoryCode,
                rtplan->range_modulator_code.c_str());
        }
        ib_item->putAndInsertString (DCM_NumberOfRangeShifters, 
	    rtplan->number_of_range_shifters.c_str());
        ib_item->putAndInsertString (DCM_NumberOfLateralSpreadingDevices,"0"); 
        ib_item->putAndInsertString (DCM_NumberOfRangeModulators, "0");
        ib_item->putAndInsertString (DCM_PatientSupportType, "TABLE");
        ib_item->putAndInsertString (DCM_PatientSupportID, 
            rtplan->patient_support_id.c_str());
        ib_item->putAndInsertString (DCM_PatientSupportAccessoryCode, 
            rtplan->patient_support_accessory_code.c_str());
        dcmtk_put (ib_item, DCM_FinalCumulativeMetersetWeight,
            beam->final_cumulative_meterset_weight);

        dcmtk_put (ib_item, DCM_NumberOfControlPoints, beam->cplist.size());
        for (size_t c = 0; c < beam->cplist.size(); c++) {
            DcmItem *cp_item = 0;
            ib_item->findOrCreateSequenceItem (
                DCM_IonControlPointSequence, cp_item, -2);
            s = PLM_to_string (c);
            cp_item->putAndInsertString (DCM_ControlPointIndex, s.c_str());

            Rtplan_control_pt *cp = beam->cplist[c];
            s = PLM_to_string (cp->cumulative_meterset_weight);
            cp_item->putAndInsertString (DCM_CumulativeMetersetWeight,
                s.c_str());
	    if (c == 0) {
                s = PLM_to_string (beam->snout_position);
                cp_item->putAndInsertString (DCM_SnoutPosition, s.c_str());
                s = PLM_to_string (beam->gantry_angle);
                cp_item->putAndInsertString (DCM_GantryAngle, s.c_str());
                cp_item->putAndInsertString (DCM_GantryRotationDirection,
                    beam->gantry_rotation_direction.c_str());
                s = PLM_to_string (beam->gantry_pitch_angle);
	        cp_item->putAndInsertString (DCM_GantryPitchAngle, s.c_str());
	        cp_item->putAndInsertString (DCM_GantryPitchRotationDirection, 
	            beam-> gantry_pitch_rotation_direction.c_str());		
                s = PLM_to_string (beam->beam_limiting_device_angle);
	        cp_item->putAndInsertString (DCM_BeamLimitingDeviceAngle, s.c_str());
	        cp_item->putAndInsertString (DCM_BeamLimitingDeviceRotationDirection, 
	            beam-> beam_limiting_device_rotation_direction.c_str());
                s = PLM_to_string (beam->patient_support_angle);
	        cp_item->putAndInsertString (DCM_PatientSupportAngle, s.c_str());
	        cp_item->putAndInsertString (DCM_PatientSupportRotationDirection, 
	            beam-> patient_support_rotation_direction.c_str());
                s = PLM_to_string (beam->table_top_vertical_position);
	        cp_item->putAndInsertString (DCM_TableTopVerticalPosition, s.c_str());
                s = PLM_to_string (beam->table_top_longitudinal_position);
	        cp_item->putAndInsertString (DCM_TableTopLongitudinalPosition, s.c_str());
                s = PLM_to_string (beam->table_top_lateral_position);
	        cp_item->putAndInsertString (DCM_TableTopLateralPosition, s.c_str());
                s = PLM_to_string (beam->table_top_pitch_angle);
	        cp_item->putAndInsertString (DCM_TableTopPitchAngle, s.c_str());
	        cp_item->putAndInsertString (DCM_TableTopPitchRotationDirection, 
	            beam-> table_top_pitch_rotation_direction.c_str());
                s = PLM_to_string (beam->table_top_roll_angle);
	        cp_item->putAndInsertString (DCM_TableTopRollAngle, s.c_str());
	        cp_item->putAndInsertString (DCM_TableTopRollRotationDirection, 
	            beam-> table_top_roll_rotation_direction.c_str());
		s = string_format ("%f\\%f\\%f", 
		    beam->isocenter_position[0],
		    beam->isocenter_position[1],
		    beam->isocenter_position[2]);
	        cp_item->putAndInsertString (DCM_IsocenterPosition, s.c_str());
	    }
            s = PLM_to_string (cp->nominal_beam_energy);
            cp_item->putAndInsertString (DCM_NominalBeamEnergy, s.c_str());

            s = PLM_to_string (cp->number_of_paintings);
            cp_item->putAndInsertString (DCM_NumberOfPaintings,
                s.c_str());
	    cp_item->putAndInsertString (DCM_ScanSpotTuneID, 
                cp->scan_spot_tune_id.c_str());
            s = string_format ("%f\\%f", cp->scanning_spot_size[0],
                cp->scanning_spot_size[1]);
            cp_item->putAndInsertString (DCM_ScanningSpotSize,
                s.c_str());

            /* Dcmtk has no putAndInsertFloat32Array, so we must 
               use more primitive methods */
            size_t num_spots = cp->scan_spot_position_map.size() / 2;
            lprintf (">> Writing %d spots\n", num_spots);
	    s = PLM_to_string (num_spots);
            cp_item->putAndInsertString (DCM_NumberOfScanSpotPositions,
                s.c_str());
            if (num_spots != cp->scan_spot_meterset_weights.size()) {
                lprintf ("Warning, scan spot positions (%d) and weights (%d)"
                    " are mismatched.\n", 
                    (int) cp->scan_spot_position_map.size(),
                    (int) cp->scan_spot_meterset_weights.size());
                if (cp->scan_spot_meterset_weights.size() < num_spots) {
                    num_spots = cp->scan_spot_meterset_weights.size();
                }
            }
            DcmFloatingPointSingle *fele;
            Float32 *f;
            fele = new DcmFloatingPointSingle (DCM_ScanSpotPositionMap);
#if __cplusplus >= 201103L
            f = cp->scan_spot_position_map.data ();
#else
            f = &cp->scan_spot_position_map[0];
#endif
            ofc = fele->putFloat32Array (f, 2*num_spots);
            ofc = cp_item->insert (fele);
            fele = new DcmFloatingPointSingle (DCM_ScanSpotMetersetWeights);
#if __cplusplus >= 201103L
            f = cp->scan_spot_meterset_weights.data ();
#else
            f = &cp->scan_spot_meterset_weights[0];
#endif
            ofc = fele->putFloat32Array (f, num_spots);
            ofc = cp_item->insert (fele);
        }
    }
    
    /* ----------------------------------------------------------------- */
    /*     Write the output file                                         */
    /* ----------------------------------------------------------------- */
    std::string filename;
    if (d_ptr->filenames_with_uid) {
        filename = string_format ("%s/rtplan_%s.dcm", dicom_dir,
            d_ptr->rt_study_metadata->get_dose_series_uid());
    } else {
        filename = string_format ("%s/rtplan.dcm", dicom_dir);
    }
    make_parent_directories (filename);

    ofc = fileformat.saveFile (filename.c_str(), EXS_LittleEndianExplicit);
    if (ofc.bad()) {
        print_and_exit ("Error: cannot write DICOM RTPLAN (%s)\n", 
            ofc.text());
    }
}
