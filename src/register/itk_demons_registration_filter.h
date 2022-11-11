/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef ITK_DEMONS_REGISTRATION_FILTER_H
#define ITK_DEMONS_REGISTRATION_FILTER_H

#include "plmregister_config.h"
#include "itk_demons_types.h"
#include "itk_image_type.h"

class Stage_parms;

struct itk_demons_registration_filter
{
protected:
    PDEDeformableRegistrationFilterType::Pointer m_demons_filter;

public:
    virtual void update_specific_parameters (const Stage_parms* parms)=0;
    virtual ~itk_demons_registration_filter(){};

    PDEDeformableRegistrationFilterType::Pointer get_demons_filter_impl()
    {
        return this->m_demons_filter;
    }
};

#endif // ITK_DEMONS_REGISTRATION_FILTER_H
