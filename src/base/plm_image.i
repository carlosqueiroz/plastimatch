%module plmbase
%include <std_shared_ptr.i>
%include <std_string.i>
%shared_ptr(Plm_image)
%{
#include "plm_image.h"
%}
%include "plmbase_config.h"
%include "smart_pointer.h"
%include "plm_image.h"
