/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _volume_boundary_type_h_
#define _volume_boundary_type_h_

#include "plmutil_config.h"
#include <string>

/*! \brief This enum is used to control the output type for 
  volume bounary detection.  If INTERIOR_EDGE, the 
  boundary will be the set of voxels within the image region that 
  lie on the border of the region.  If INTERIOR_FACE,
  the output image has dimension one larger than the input image, 
  and each voxel is a bitmask of size seven, which correspond to a 
  boundary at the voxel corner where all three dimensions have 
  negative index, the boundary at three edges where two dimensions 
  have negative index, and the boundary at three faces where 
  the face has negative index.
  Note however, that corner and edges on the volume boundary 
  are not guaranteed to be correct, since these are not used 
  by the downstream distance map code. */
enum Volume_boundary_type {
    INTERIOR_EDGE,
    INTERIOR_FACE
};

const unsigned char VBB_MASK_NEG_I = 0x01;
const unsigned char VBB_MASK_NEG_J = 0x02;
const unsigned char VBB_MASK_NEG_K = 0x04;
const unsigned char VBB_MASK_POS_I = 0x08;
const unsigned char VBB_MASK_POS_J = 0x10;
const unsigned char VBB_MASK_POS_K = 0x20;

PLMUTIL_API Volume_boundary_type volume_boundary_type_parse (const std::string& string);
PLMUTIL_API Volume_boundary_type volume_boundary_type_parse (const char* string);

#endif
