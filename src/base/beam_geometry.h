/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _beam_geometry_h_
#define _beam_geometry_h_

#include "plmbase_config.h"
#include "plm_int.h"
#include "smart_pointer.h"

enum Beam_geometry_type {
    BEAM_GEOMETRY_UNDEFINED,
    BEAM_GEOMETRY_LEGACY,
    BEAM_GEOMETRY_IEC
};

class PLMBASE_API Beam_geometry {
public:
    SMART_POINTER_SUPPORT (Beam_geometry);
public:
    Beam_geometry ();
    ~Beam_geometry ();
public:
    Beam_geometry_type beam_geometry_type;

    /* Machine quantities */
    float source_aperture_distance;
    float source_axis_distance;
    float source_detector_distance;
    float aperture_size[2];
    float detector_size[2];

    /* Plan quantities */
    float gantry_angle;
    float couch_angle;
    float isocenter[3];

    /* Computational quantities */
    plm_long computation_dim[2];
    float computation_offset[2];
    float computation_spacing[2];

    /* Derived quantities */
    float source_position[3];
};

#endif
