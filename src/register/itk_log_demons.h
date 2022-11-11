/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _itk_log_demons_h_
#define _itk_log_demons_h_

#include "plmregister_config.h"
#include "itk_demons_types.h"
#include "itk_demons_registration_filter.h"

class itk_log_domain_demons_filter: public itk_demons_registration_filter
{
    typedef LogDomainDemonsFilterType::DemonsRegistrationFunctionType LogDomainDemonsFunctionType;
    typedef LogDomainDemonsFunctionType::GradientType GradientType;

public:
    itk_log_domain_demons_filter();
    ~itk_log_domain_demons_filter();
    void update_specific_parameters(const Stage_parms* stage);
};

#endif
