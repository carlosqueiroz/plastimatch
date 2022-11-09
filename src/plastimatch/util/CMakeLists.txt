##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_util)

configure_file (
    ${CMAKE_CURRENT_SOURCE_DIR}/plmutil_config.h.in
    ${PLM_BINARY_DIR}/plmutil_config.h
)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories (AFTER ${DEVILLARD_INCLUDE_DIR})

if (CUDA_FOUND)
  add_subdirectory (cuda)
  include_directories (BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
endif ()

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (PLMUTIL_LIBRARY_SRC
  bspline_correspond.cxx
  dice_statistics.cxx dice_statistics.h 
  dicom_sro_save.cxx
  distance_map.cxx
  diff.cxx
  dvh.cxx dvh.h 
  gamma_dose_comparison.cxx gamma_dose_comparison.h
  geometry_chooser.cxx geometry_chooser.h 
  hausdorff_distance.cxx hausdorff_distance.h 
  image_boundary.cxx image_boundary.h
  image_center.cxx image_center.h
  itk_adjust.cxx itk_adjust.h
  itk_crop.cxx
  itk_distance_map.cxx
  itk_gabor.cxx
  itk_histogram_matching.cxx itk_histogram_matching.h
  itk_intersect.cxx itk_intersect.h
  itk_local_intensity_correction.cxx itk_local_intensity_correction.h
  itk_mask.cxx
  itk_scale.cxx
  itk_threshold.cxx
  itk_union.cxx itk_union.h
  landmark_diff.cxx
  proj_image_filter.cxx
  plm_series.cxx
  plm_study.cxx
  rt_study_warp.cxx rt_study_warp.h
  sift.cxx
  simplify_points.cxx 
  ss_img_stats.cxx 
  synthetic_mha.cxx
  synthetic_vf.cxx
  threshbox.cxx
  volume_adjust.cxx volume_adjust.h
  volume_boundary_type.cxx volume_boundary_type.h
  xf_invert.cxx xf_invert.h
  zcrop.cxx zcrop.h
  )
if (FFTW_FOUND)
  set (PLMUTIL_LIBRARY_SRC
    ${PLMUTIL_LIBRARY_SRC}
    gabor.cxx gabor.h 
    ramp_filter.cxx ramp_filter.h
   )
endif ()

set (PLMUTIL_LIBRARY_HEADERS
  "${PLM_BINARY_DIR}/plmutil_config.h"
  )
foreach (ARG ${PLMUTIL_LIBRARY_SRC})
  string (REGEX MATCH ".*\\.h$" TMP "${ARG}")
  if (TMP)
    list (APPEND PLMUTIL_LIBRARY_HEADERS "${TMP}")
  endif ()
endforeach ()

##-----------------------------------------------------------------------------
##  LIBRARY DEPENDENCIES
##-----------------------------------------------------------------------------
set (PLMUTIL_LIBRARY_DEPENDENCIES 
  ${FFTW_LIBRARIES}
  devillard
  plmbase
  plmsys
  )

if (CUDA_FOUND)
  set (PLMUTIL_LIBRARY_DEPENDENCIES 
    ${PLMUTIL_LIBRARY_DEPENDENCIES}
    plmcuda
    )
  set (PLMUTIL_LIBRARY_DEPENDENCIES 
    ${PLMUTIL_LIBRARY_DEPENDENCIES}
    plmutilcuda
    )
endif ()

##-----------------------------------------------------------------------------
##  SPECIAL BUILD RULES: OpenMP & SSE2
##-----------------------------------------------------------------------------
if (OPENMP_FOUND)
  set (PLMUTIL_LIBRARY_LDFLAGS "${OPENMP_LDFLAGS}")
  set_source_files_properties (dice_statistics.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (image_center.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (vf_invert.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
  set_source_files_properties (distance_map.cxx
    PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
endif ()

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
plm_add_library (
  plmutil
  "${PLMUTIL_LIBRARY_SRC}" 
  "${PLMUTIL_LIBRARY_DEPENDENCIES}"
  "${PLMUTIL_LIBRARY_LDFLAGS}"
  "${PLASTIMATCH_INCLUDE_DIRECTORIES}"
  "${PLMUTIL_LIBRARY_HEADERS}")

if (SWIG_FOUND)
  include (UseSWIG)
  message ("Oh yea, give me a SWIG.")
  
  set (PLMUTIL_SWIG_LIBRARY_SRC
    gamma_dose_comparison.i)
  set_property (SOURCE ${PLMUTIL_SWIG_LIBRARY_SRC}
    PROPERTY CPLUSPLUS ON)
  swig_add_library (plmutil-cs
    TYPE USE_BUILD_SHARED_LIBS
    LANGUAGE csharp
    # Do I need to set OUTPUT_DIR or OUTFILE_DIR ?
    SOURCES ${PLMUTIL_SWIG_LIBRARY_SRC})
endif ()


# because plmutilcuda is dynamically loaded (not linked)
# CMake needs to be told explicitly that plmutil
# depends on it, so we tell it explicitly here
if (CUDA_FOUND)
  add_dependencies (plmutil plmutilcuda)
endif ()
