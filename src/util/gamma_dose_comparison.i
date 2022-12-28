%module plmutil
%include <std_shared_ptr.i>
%include <std_string.i>
%shared_ptr(Plm_image)
%{
#include "gamma_dose_comparison.h"
%}
%import(module="plmbase") "plm_image.i"
%include "plmutil_config.h"
%include "gamma_dose_comparison.h"
