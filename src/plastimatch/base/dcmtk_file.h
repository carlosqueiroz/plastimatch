/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _dcmtk_file_h_
#define _dcmtk_file_h_

#include "plmbase_config.h"
#include "sys/plm_int.h"
#include "volume_header.h"

class DcmDataset;
class DcmElement;
class DcmSequenceOfItems;
class DcmTagKey;

class Dcmtk_file_private;

class Dcmtk_file
{
public:
    Dcmtk_file ();
    Dcmtk_file (const char *fn);
    ~Dcmtk_file ();

public:
    Dcmtk_file_private *d_ptr;

public:
    void debug () const;
    DcmDataset* get_dataset (void) const;
    const char* get_cstr (const DcmTagKey& tag_key) const;
    bool get_uint8 (const DcmTagKey& tag_key, uint8_t* val) const;
    bool get_uint16 (const DcmTagKey& tag_key, uint16_t* val) const;
    bool get_float (const DcmTagKey& tag_key, float* val) const;
    bool get_ds_float (const DcmTagKey& tag_key, float* val) const;
    bool get_uint16_array (const DcmTagKey& tag_key, 
	const uint16_t** val, unsigned long* count) const;
    bool get_int16_array (const DcmTagKey& tag_key, 
	const int16_t** val, unsigned long* count) const;
    bool get_uint32_array (const DcmTagKey& tag_key, 
	const uint32_t** val, unsigned long* count) const;
    bool get_int32_array (const DcmTagKey& tag_key, 
	const int32_t** val, unsigned long* count) const;
    bool get_element (const DcmTagKey& tag_key, DcmElement* val) const;
    bool get_sequence (const DcmTagKey& tag_key, 
        DcmSequenceOfItems*& seq) const;
    const Volume_header* get_volume_header () const;
    plm_long get_z_position () const;

    void load_header (const char *fn);
};


PLMBASE_C_API void dcmtk_series_test (char *dicom_dir);
PLMBASE_C_API bool dcmtk_file_compare_z_position (
        const Dcmtk_file* f1,
        const Dcmtk_file* f2
);


#endif
