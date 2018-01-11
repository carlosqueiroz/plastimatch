/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmbase_config.h"
#include "image_stats.h"
#include "itk_image_stats.h"
#include "logfile.h"
#include "plm_image.h"
#include "volume.h"
#include "volume_stats.h"

/* -----------------------------------------------------------------------
   Statistics like min, max, etc.
   ----------------------------------------------------------------------- */
template <class T>
Image_stats::Image_stats (const T& t)
{
    itk_image_stats (t, this);
}

#if defined (commentout)
template <>
Image_stats::Image_stats (const Volume*& vol)
{
    volume_stats (vol, this);
}
#endif

template <>
Image_stats::Image_stats (const Volume::Pointer& vol)
{
    volume_stats (vol.get(), this);
}

void
Image_stats::print ()
{
    lprintf (" MIN %g AVG %g MAX %g NONZERO: (%d / %d)\n",
        min_val, avg_val, max_val, num_non_zero, num_vox);
}

template<class T> void
image_stats_print (const T& t)
{
    Image_stats image_stats (t);
    image_stats.print ();
}

#if defined (commentout)
template<> void
image_stats_print (const Volume::Pointer& t)
{
    image_stats_print (t);
}
#endif

template<> PLMBASE_API void
image_stats_print (const Plm_image::Pointer& t)
{
    switch (t->m_type) {
    case PLM_IMG_TYPE_GPUIT_UCHAR:
    case PLM_IMG_TYPE_GPUIT_UINT16:
    case PLM_IMG_TYPE_GPUIT_SHORT:
    case PLM_IMG_TYPE_GPUIT_UINT32:
    case PLM_IMG_TYPE_GPUIT_INT32:
    case PLM_IMG_TYPE_GPUIT_FLOAT:
        image_stats_print (t->get_volume());
        break;
    case PLM_IMG_TYPE_ITK_UCHAR:
        image_stats_print (t->itk_uchar());
        break;
    case PLM_IMG_TYPE_ITK_SHORT:
        image_stats_print (t->itk_short());
        break;
    case PLM_IMG_TYPE_ITK_USHORT:
        image_stats_print (t->itk_ushort());
        break;
    case PLM_IMG_TYPE_ITK_FLOAT:
        image_stats_print (t->itk_float());
        break;
    default:
        lprintf ("Error, cannot compute image_stats_print on type %s\n",
            plm_image_type_string (t->m_type));
        break;
    }
}

template PLMBASE_API void image_stats_print (const Volume::Pointer&);
//template PLMBASE_API void image_stats_print (const Volume*&);
