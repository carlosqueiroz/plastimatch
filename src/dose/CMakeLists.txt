##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_dose)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmdose_config.h.in
    ${PLM_BINARY_DIR}/plmdose_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMDOSE_LIBRARY_SRC
  beam_calc.cxx beam_calc.h
  beam_map.cxx beam_map.h
  bragg_curve.cxx bragg_curve.h
  compute_dose.cxx compute_dose.h
  dose_volume_functions.cxx dose_volume_functions.h
  particle_type.cxx particle_type.h
  rt_beam_model.cxx rt_beam_model.h
  rt_depth_dose.cxx rt_depth_dose.h
  rt_dij.cxx rt_dij.h
  rt_dose_timing.h
  rt_lut.cxx rt_lut.h
  rt_spot_map.cxx rt_spot_map.h
  rt_mebs.cxx rt_mebs.h
  rt_parms.cxx rt_parms.h
  plan_calc.cxx plan_calc.h
  rt_sigma.cxx rt_sigma.h
  wed_parms.cxx wed_parms.h
  )

set (PLMDOSE_LIBRARY_DEPENDENCIES 
  plmbase
  plmutil
  plmsys
  )

set (PLMDOSE_LIBRARY_DEPENDENCIES 
  ${PLMDOSE_LIBRARY_DEPENDENCIES} 
  specfun)

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmdose
  "${PLMDOSE_LIBRARY_SRC}" 
  "${PLMDOSE_LIBRARY_DEPENDENCIES}"
  ""
  "${PLASTIMATCH_INCLUDE_DIRECTORIES}"
  "")
