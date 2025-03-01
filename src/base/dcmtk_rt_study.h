/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_rt_study_h_
#define _dcmtk_rt_study_h_

#include "plmbase_config.h"
#include <list>

#include "itk_image.h"
#include "plm_image.h"
#include "plm_int.h"
#include "rt_study_metadata.h"
#include "rtss.h"
#include "rtplan.h"
#include "volume.h"

class Dcmtk_rt_study_private;
class Dcmtk_slice_data;

class PLMBASE_API Dcmtk_rt_study {
public:
    Dcmtk_rt_study_private *d_ptr;
public:
    Dcmtk_rt_study ();
    Dcmtk_rt_study (const char* dicom_path);
    ~Dcmtk_rt_study ();
public:
    void load (const char *dicom_path);
    void save (const char *dicom_path);
public:
    const char* get_ct_series_uid () const;
    const char* get_dose_instance_uid () const;
    const char* get_dose_series_uid () const;
    const char* get_frame_of_reference_uid () const;
    const char* get_plan_instance_uid () const;
    const char* get_rtss_instance_uid () const;
    const char* get_rtss_series_uid () const;
    const char* get_study_date () const;
    const char* get_study_time () const;
    const char* get_study_description () const;
    const char* get_study_uid () const;
    std::vector<Dcmtk_slice_data>* get_slice_data();

public:
    Plm_image::Pointer& get_image ();
    Volume::Pointer get_image_volume_float ();
    Volume *get_volume ();
    void set_image (const Plm_image::Pointer& image);

    Rtss::Pointer& get_rtss ();
    void set_rtss (const Rtss::Pointer& rtss);

    Rtplan::Pointer& get_rtplan();
    void set_rtplan (const Rtplan::Pointer& rtplan);

    Plm_image::Pointer& get_dose ();
    void set_dose (const Plm_image::Pointer& dose);

    void set_rt_study_metadata (
        const Rt_study_metadata::Pointer& rt_study_metadata);

    void set_filenames_with_uid (bool filenames_with_uid);

public:
    void image_save (const char *dicom_path);
    void dose_save (const char *dicom_path);
    void rtss_save (const char *dicom_path);
    void rtplan_save (const char *dicom_path);

    void load_directory (void);
    void image_load ();
    void rtss_load ();
    void rtdose_load ();
    void rtplan_load();

    void insert_file (const char* fn);
    void insert_directory (const char* fn);
    void sort_all (void);

    void debug (void) const;

protected:
    void rt_ion_plan_load ();
};

#endif
