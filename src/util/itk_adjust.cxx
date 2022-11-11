/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmutil_config.h"
#include <limits>
#include "itkImageRegionIterator.h"

#include "float_pair_list.h"
#include "itk_adjust.h"
#include "itk_image_clone.h"
#include "plm_math.h"
#include "print_and_exit.h"
#include "pwlut.h"

FloatImageType::Pointer
itk_adjust (FloatImageType::Pointer image_in, const Float_pair_list& al)
{
    FloatImageType::Pointer image_out = itk_image_clone (image_in);

    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;
    FloatImageType::RegionType rg = image_out->GetLargestPossibleRegion ();
    FloatIteratorType it (image_out, rg);

    Pwlut pwlut;
    pwlut.set_lut (al);

    for (it.GoToBegin(); !it.IsAtEnd(); ++it) {
        it.Set (pwlut.lookup (it.Get()));
    }
    return image_out;
}

FloatImageType::Pointer
itk_adjust (FloatImageType::Pointer image_in, const std::string& adj_string)
{
    Float_pair_list al = parse_float_pairs (adj_string);

    if (al.empty()) {
        print_and_exit ("Error: couldn't parse adjust string: %s\n",
            adj_string.c_str());
    }

    return itk_adjust (image_in, al);
}

FloatImageType::Pointer
itk_auto_adjust (FloatImageType::Pointer image_in)
{
    typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;

    FloatImageType::RegionType rg = image_in->GetLargestPossibleRegion ();
    FloatIteratorType it (image_in, rg);
    Float_pair_list::const_iterator ait;

    /* GCS: This is just something for spark, works for CT image differencing
       -- make a better method later */
    Float_pair_list al;
    al.push_back (std::make_pair (-std::numeric_limits<float>::max(), 0.0));
    al.push_back (std::make_pair (-200.0,0));
    al.push_back (std::make_pair (0.0,127.5));
    al.push_back (std::make_pair (+200.0,255));
    al.push_back (std::make_pair (std::numeric_limits<float>::max(), 0.0));

    return itk_adjust (image_in, al);
}
