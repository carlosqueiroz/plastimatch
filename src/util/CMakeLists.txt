##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_util)

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

set (PLMUTIL_LIBRARY_HEADERS "")
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
  set (PLMUTIL_SWIG_LIBRARY_SRC
    gamma_dose_comparison.i)
  set_property (SOURCE ${PLMUTIL_SWIG_LIBRARY_SRC}
    PROPERTY CPLUSPLUS ON)
  set (CSHARP_OUTPUT_DIR "${PLM_BINARY_DIR}/csharp")
  set (CSHARP_OUTFILE_DIR "${PLM_BINARY_DIR}/csharp-cxx")
  set (PYTHON_OUTPUT_DIR "${PLM_BINARY_DIR}/python")
  set (PYTHON_OUTFILE_DIR "${PLM_BINARY_DIR}/python-cxx")
  if (PLM_CONFIG_ENABLE_CSHARP)
    swig_add_library (plmutil-cs
      TYPE SHARED
      LANGUAGE csharp
      OUTPUT_DIR "${CSHARP_OUTPUT_DIR}"
      OUTFILE_DIR "${CSHARP_OUTFILE_DIR}"
      SOURCES ${PLMUTIL_SWIG_LIBRARY_SRC})
    set_property (TARGET plmutil-cs PROPERTY SWIG_INCLUDE_DIRECTORIES 
      "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/sys>;$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/base>")
    target_link_libraries (plmutil-cs PUBLIC plmutil)
    if (TARGET DCMTK::dcmdata)
      target_link_libraries (plmutil-cs PUBLIC DCMTK::dcmdata)
    else ()
      target_link_libraries (plmutil-cs PUBLIC dcmdata)
    endif ()
    set_target_properties (plmutil-cs PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CSHARP_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${CSHARP_OUTPUT_DIR}"
      ARCHIVE DESTINATION "${PLM_INSTALL_LIB_DIR}"
      PUBLIC_HEADER DESTINATION "${PLM_INSTALL_INCLUDE_DIR}"
      )
  endif ()
  if (PLM_CONFIG_ENABLE_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    swig_add_library (plmutil_py
      TYPE SHARED
      LANGUAGE python
      OUTPUT_DIR "${PYTHON_OUTPUT_DIR}"
      OUTFILE_DIR "${PYTHON_OUTFILE_DIR}"
      SOURCES ${PLMUTIL_SWIG_LIBRARY_SRC})
    set_property (TARGET plmutil_py PROPERTY SWIG_INCLUDE_DIRECTORIES
      "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/sys>;$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src/base>")
    target_include_directories (plmutil_py PUBLIC ${Python3_INCLUDE_DIRS})
    target_link_libraries (plmutil_py PUBLIC plmutil)
    if (TARGET DCMTK::dcmdata)
      target_link_libraries (plmutil_py PUBLIC DCMTK::dcmdata)
    else ()
      target_link_libraries (plmutil_py PUBLIC dcmdata)
    endif ()
    set_target_properties (plmutil_py PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${PYTHON_OUTPUT_DIR}"
      LIBRARY_OUTPUT_DIRECTORY "${PYTHON_OUTPUT_DIR}"
      ARCHIVE DESTINATION "${PLM_INSTALL_LIB_DIR}"
      PUBLIC_HEADER DESTINATION "${PLM_INSTALL_INCLUDE_DIR}"
      )
  endif ()
endif ()


# because plmutilcuda is dynamically loaded (not linked)
# CMake needs to be told explicitly that plmutil
# depends on it, so we tell it explicitly here
if (CUDA_FOUND)
  add_dependencies (plmutil plmutilcuda)
endif ()
