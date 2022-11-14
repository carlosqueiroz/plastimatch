##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_clp)

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMCLP_LIBRARY_SRC
  plm_clp.cxx plm_clp.h
  )

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
#plm_add_library (
#  plmclp
#  "${PLMCLP_LIBRARY_SRC}" 
#  ""
#  "")
plm_add_library (
  plmclp
  "${PLMCLP_LIBRARY_SRC}"
  ""
  ""
  "${PLASTIMATCH_INCLUDE_DIRECTORIES}"
  ""
)
