/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _image_stats_h_
#define _image_stats_h_

#include "plmbase_config.h"
#include "plm_int.h"

class Image_stats
{
public:
    template<class T> Image_stats (const T& t);
public:
    double min_val;
    double max_val;
    double avg_val;
    plm_long num_vox;
    plm_long num_non_zero;
public:
    void print ();
};

template<class T> PLMBASE_API void image_stats_print (const T& t);

#endif
