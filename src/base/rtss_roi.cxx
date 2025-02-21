/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "plm_math.h"
#include "rtss_contour.h"
#include "rtss_roi.h"
#include "string_util.h"

Rtss_roi::Rtss_roi ()
{
    this->id = -1;
    this->bit = 0;
    this->num_contours = 0;
    this->pslist = 0;
    this->rsp_value = -1.0;
}

Rtss_roi::~Rtss_roi ()
{
    this->clear ();
}

void
Rtss_roi::clear ()
{
    for (size_t i = 0; i < this->num_contours; i++) {
        if (this->pslist[i] != nullptr) {
            delete this->pslist[i];
        }
    }
    free (this->pslist);

    this->name = "";
    this->color = "";
    this->id = -1;
    this->bit = 0;
    this->num_contours = 0;
    this->pslist = 0;
    this->rsp_value = -1.0;
}

Rtss_contour*
Rtss_roi::add_polyline ()
{
    Rtss_contour* new_polyline;

    this->num_contours++;
    this->pslist = (Rtss_contour**) realloc (this->pslist, 
	    this->num_contours * sizeof(Rtss_contour*));

    new_polyline 
	= this->pslist[this->num_contours - 1]
	= new Rtss_contour;
    return new_polyline;
}

Rtss_contour*
Rtss_roi::add_polyline (size_t num_vertices)
{
    Rtss_contour* rtss_contour = this->add_polyline();
    rtss_contour->num_vertices = num_vertices;
    rtss_contour->slice_no = -1;
    rtss_contour->ct_slice_uid = "";
    rtss_contour->x = (float*) malloc (num_vertices * sizeof(float));
    rtss_contour->y = (float*) malloc (num_vertices * sizeof(float));
    rtss_contour->z = (float*) malloc (num_vertices * sizeof(float));
    return rtss_contour;
}

std::string
Rtss_roi::adjust_name (const std::string& name_in)
{
    /* GE Adv sim doesn't like names with strange punctuation. */
    /* 3D Slicer color table doesn't allow spaces */
    std::string name_out = name_in;
    for (size_t i = 0; i < name_in.length(); i++) {
	if (isalnum (name_in[i])) {
	    name_out[i] = name_in[i];
	} else {
	    name_out[i] = '_';
	}
    }
    return name_out;
}

void
Rtss_roi::set_color (const char* color_string)
{
    int r = 255, g = 0, b = 0;
    if (color_string) {
        if (3 == sscanf (color_string, "%d %d %d", &r, &g, &b)) {
            /* Parsed OK */
        }
        else if (3 == sscanf (color_string, "%d\\%d\\%d", &r, &g, &b)) {
            /* Parsed OK */
        } else {
            r = 255;
            g = 0;
            b = 0;
        }
    }

    this->color = string_format ("%d %d %d", r, g, b);
}

std::string
Rtss_roi::get_dcm_color_string () const
{
    int r, g, b;
    this->get_rgb (&r, &g, &b);
    return string_format ("%d\\%d\\%d", r, g, b);
}

void
Rtss_roi::get_rgb (int *r, int *g, int *b) const
{
    *r = 255;
    *g = 0;
    *b = 0;
    if (this->color.empty()) {
	return;
    }

    /* Ignore return code -- unparsed values will remain unassigned */
    sscanf (this->color.c_str(), "%d %d %d", r, g, b);
}
