/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtss_structure_h_
#define _rtss_structure_h_

#include "plmbase_config.h"
#include "pstring.h"
#include "plmbase_config.h"

#define CXT_BUFLEN 2048

class Plm_image;
class Plm_image_header;

class Rtss_polyline {
public:
    int slice_no;           /* Can be "-1" */
    Pstring ct_slice_uid;
    int num_vertices;
    float* x;
    float* y;
    float* z;
public:
    PLMBASE_API Rtss_polyline ();
    PLMBASE_API ~Rtss_polyline ();
};

class Rtss_structure {
public:
    Pstring name;
    Pstring color;
    int id;                    /* Used for import/export (must be >= 1) */
    int bit;                   /* Used for ss-img (-1 for no bit) */
    size_t num_contours;
    Rtss_polyline** pslist;
public:
    PLMBASE_API Rtss_structure ();
    PLMBASE_API ~Rtss_structure ();

    PLMBASE_API void clear ();
    PLMBASE_API Rtss_polyline* add_polyline ();
    PLMBASE_API void set_color (const char* color_string);
    PLMBASE_API void get_dcm_color_string (Pstring *dcm_color) const;
    PLMBASE_API void structure_rgb (int *r, int *g, int *b) const;

    static void adjust_name (Pstring *name_out, const Pstring *name_in);
};

#if defined (commentout)
PLMBASE_API Cxt_structure_list*
cxt_create (void);
PLMBASE_API void cxt_init (Cxt_structure_list* structures);
PLMBASE_API Cxt_structure* cxt_add_structure (
    Cxt_structure_list *cxt, 
    const Pstring& structure_name, 
    const Pstring& color, 
    int structure_id);
PLMBASE_API Cxt_structure* cxt_find_structure_by_id (Cxt_structure_list* structures, int structure_id);
PLMBASE_API void cxt_debug (Cxt_structure_list* structures);
PLMBASE_API void cxt_adjust_structure_names (Cxt_structure_list* structures);
PLMBASE_API void cxt_free (Cxt_structure_list* structures);
PLMBASE_API void cxt_destroy (Cxt_structure_list* structures);
PLMBASE_API void cxt_prune_empty (Cxt_structure_list* structures);
PLMBASE_API Cxt_structure_list* cxt_clone_empty (
    Cxt_structure_list* cxt_out, 
    Cxt_structure_list* cxt_in
);
PLMBASE_API void cxt_apply_geometry (Cxt_structure_list* structures);
PLMBASE_API void cxt_set_geometry_from_plm_image_header (
    Cxt_structure_list* cxt, 
    Plm_image_header *pih
);
PLMBASE_API void cxt_set_geometry_from_plm_image (
    Cxt_structure_list* structures,
    Plm_image *pli
);
#endif

#if defined (commentout)
PLMBASE_API Cxt_polyline* cxt_add_polyline (Cxt_structure* structure);
PLMBASE_API void cxt_structure_rgb (const Cxt_structure *structure, int *r, int *g, int *b);
PLMBASE_API void cxt_adjust_name (Pstring *name_out, const Pstring *name_in);
#endif


#endif
