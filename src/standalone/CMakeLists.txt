##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_standalone)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories (BEFORE ${CMAKE_CURRENT_BINARY_DIR})
include_directories (AFTER ${INIH_INCLUDE_DIR})

set (XVI_ARCHIVE_LIBRARIES 
    ${PLASTIMATCH_LIBS}
    inih
    )

##-----------------------------------------------------------------------------
##  SOURCE FILES
##-----------------------------------------------------------------------------
set (BRAGG_CURVE_SRC
  bragg_curve_main.cxx 
  bragg_curve_opts.cxx
  )
set (SOBP_SRC
  sobp_main.cxx 
  )
set (CHECK_GRAD_SRC
  check_grad.cxx
  )
set (COMPUTE_DISTANCE_SRC
  compute_distance.cxx
  closest_point.cxx 
  read_obj.cxx
  )
set (CUDA_PROBE_SRC
  cuda_probe_main.cxx
  )
set (DEMONS_SRC
  demons_main.cxx 
  demons_opts.cxx
  )
set (DICOM_INFO_SRC
  dicom_info.cxx
  )
set (DICOM_UID_SRC
  dicom_uid_main.cxx
  )
set (DLIB_TRAIN_SRC 
  dlib_train.cxx
  )
set (FDK_SRC
  fdk_main.cxx 
  fdk_opts.cxx
)
set (HND_TO_PFM_SRC
  hnd_to_pfm.cxx 
  )
set (LANDMARK_DIFF_SRC
  landmark_diff_main.cxx
  )
set (MERGE_VFS_SRC
  merge_vector_fields.cxx
  )
set (MEX_DRR_SRC
  mex_drr.c
  )
set (MHA_TO_RAW_SRC
  mha_to_raw.c
  )
set (OPENCL_PROBE_SRC
  opencl_probe_main.c
  opencl_probe.cxx
  )
if (OpenCL_FOUND)
  plm_add_opencl_file (OPENCL_PROBE_SRC opencl_probe.cl)
endif ()
set (RAW_TO_MHA_SRC
  raw_to_mha.c
  )
set (VISCOUS_SRC
  viscous.cxx
  )
set (WED_SRC
  wed_main.cxx
  )
set (XVI_ARCHIVE_SRC
    xvi_archive.cxx
    )
set (GAMMA_THRESH_SRC
  Gamma_Thresh.cxx
  )
set (FLIP_IMAGE_SRC
  flip_image.cxx
  )
set (DECOMP_VECTOR_SRC
  decomp_vector.cxx
  )
set (FIND_CENTER_SRC
  find_center.cxx
  )
set (SHUFFLE_MHA_SRC
  shuffle_mha_main.cxx
  )
set (VF_INVERT_SRC
  vf_invert_main.cxx
  )
set (XF_TO_EMPIREFMT_SRC
    xf_to_empirefmt.cxx
    )
set (CMD_PROMPT_LAUNCHER_SRC
    cmd_prompt_launcher.cxx
    )
if (QT4_FOUND)
  # set(nki2mha_converter_SRC
  #   nki2mha_converter_main.cpp
  #   nki2mha_converter.cpp nki2mha_converter.h)
  # QT4_WRAP_CPP(nki2mha_converter_SRC
  #   nki2mha_converter.h)
  # QT4_WRAP_UI(nki2mha_converter_SRC
  #   nki2mha_converter.ui)
  # QT4_ADD_RESOURCES (nki2mha_converter_SRC
  #   nki2mha_converter.qrc)

  set(GAMMA_GUI_BASE_SRC
    gamma_gui_main.cpp	
    gamma_gui.cpp
    gamma_gui.h
    qyklabel.h
    qyklabel.cpp
    DlgGammaView.h
    DlgGammaView.cxx
    YK16GrayImage.cxx
    YK16GrayImage.h	
    qt_util.h
    qt_util.cxx
    qcustomplot.h
    qcustomplot.cpp
    yk_config.h)

  QT4_WRAP_CPP(GAMMA_GUI_MOC
    DlgGammaView.h
    qyklabel.h
    qcustomplot.h
    gamma_gui.h)
	
  QT4_WRAP_UI(GAMMA_GUI_UI
    gamma_gui.ui
    DlgGammaView.ui)

  set(GAMMA_GUI_SRC
    ${GAMMA_GUI_BASE_SRC}
    ${GAMMA_GUI_MOC}
    ${GAMMA_GUI_UI}
    )

# Don't compile the files which include generated headers until
# generated headers are built.
set_source_files_properties (${GAMMA_GUI_BASE_SRC}
    PROPERTIES OBJECT_DEPENDS "${GAMMA_GUI_UI}")
    
set(REGISTER_GUI_SRC
  register_gui_main.cpp
  register_gui.cpp
  register_gui.h
  qyklabel.h
  qyklabel.cpp
  YK16GrayImage.cxx
  YK16GrayImage.h
  qt_util.h
  qt_util.cxx
  qcustomplot.h
  qcustomplot.cpp
  yk_config.h
  YKThreadRegi.h
  YKThreadRegi.cpp
  )

QT4_WRAP_CPP(REGISTER_GUI_SRC	
  qyklabel.h
  qcustomplot.h
  register_gui.h
  YKThreadRegi.h)
QT4_WRAP_UI(REGISTER_GUI_SRC
  register_gui.ui
  )

set(BEAMDATA_GEN_GUI_SRC	
	beamdata_gen_gui.h
	beamdata_gen_gui.cpp
	beamdata_gen_gui_main.cpp
	qyklabel.h
	qyklabel.cpp
	YK16GrayImage.cxx
	YK16GrayImage.h
	qt_util.h
	qt_util.cxx
	qcustomplot.h
	qcustomplot.cpp
	yk_config.h
	BeamDataRFA.cxx
	BeamDataRFA.h
	)

QT4_WRAP_CPP(BEAMDATA_GEN_GUI_SRC	
	beamdata_gen_gui.h
	qyklabel.h
	qcustomplot.h
	beamdata_gen_gui.h
	)
QT4_WRAP_UI(BEAMDATA_GEN_GUI_SRC
	beamdata_gen_gui.ui
	)

endif()

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
if (ITK_FOUND)
  plm_add_executable_v3 (compute_distance "${COMPUTE_DISTANCE_SRC}" 
    "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
  plm_add_executable_v3 (dicom_info "${DICOM_INFO_SRC}"
    "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
  plm_add_executable_v3 (landmark_diff "${LANDMARK_DIFF_SRC}" 
    "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
  plm_add_executable_v3 (merge_vfs  "${MERGE_VFS_SRC}"
    "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
  plm_add_executable_v3 (shuffle_mha "${SHUFFLE_MHA_SRC}"
    "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
  if (PLM_BUILD_VISCOUS)
    plm_add_executable_v3 (viscous "${VISCOUS_SRC}"
      "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
  endif ()
#  plm_add_executable_v3 (vf_invert "${VF_INVERT_SRC}"
#    "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
  plm_add_executable_v3 (wed "${WED_SRC}" 
    "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
  if (PLM_DCM_USE_DCMTK)
    plm_add_executable_v3 (xvi_archive "${XVI_ARCHIVE_SRC}" 
      "" "${XVI_ARCHIVE_LIBRARIES}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
  endif ()
endif ()

if (QT4_FOUND)
  set (QT4_INCLUDE_DIRS ${QT_QTGUI_INCLUDE_DIR}
    ${QT_INCLUDE_DIR} ${QT_QTCORE_INCLUDE_DIR})
  set (QT4_LIBRARIES ${QT_QTGUI_LIBRARY} ${QT_QTCORE_LIBRARY})
  # plm_add_executable_v3 (nki2mha_converter "${nki2mha_converter_SRC}"
  #   "${QT4_INCLUDE_DIRS}" "${PLASTIMATCH_LIBS};${QT4_LIBRARIES}"
  #   "${PLASTIMATCH_LDFLAGS}" ${INSTALL_IF_NOT_DEBIAN})
  if (DCMTK_FOUND)
    plm_add_executable_v3 (gamma_gui "${GAMMA_GUI_SRC}"
      "${QT4_INCLUDE_DIRS}" "${PLASTIMATCH_LIBS};${QT4_LIBRARIES}"
      "${PLASTIMATCH_LDFLAGS}" ${INSTALL_IF_NOT_DEBIAN})
    plm_add_executable_v3 (beamdata_gen_gui "${BEAMDATA_GEN_GUI_SRC}" 
      "${QT4_INCLUDE_DIRS}" "${PLASTIMATCH_LIBS};${QT4_LIBRARIES}"
      "${PLASTIMATCH_LDFLAGS}" ${INSTALL_IF_NOT_DEBIAN})
  endif ()
  plm_add_executable_v3 (register_gui "${REGISTER_GUI_SRC}" 
    "${QT4_INCLUDE_DIRS}" "${PLASTIMATCH_LIBS};${QT4_LIBRARIES}"
    "${PLASTIMATCH_LDFLAGS}" ${INSTALL_IF_NOT_DEBIAN})
endif()

plm_add_executable_v3 (check_grad "${CHECK_GRAD_SRC}" 
  "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
plm_add_executable_v3 (demons "${DEMONS_SRC}" 
  "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
plm_add_executable_v3 (dlib_train "${DLIB_TRAIN_SRC}" 
  "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
plm_add_executable_v3 (hnd_to_pfm "${HND_TO_PFM_SRC}" 
  "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
plm_add_executable_v3 (bragg_curve "${BRAGG_CURVE_SRC}" 
  "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
plm_add_executable_v3 (sobp "${SOBP_SRC}" 
  "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})

# plm_add_executable_v3 (fdk "${FDK_SRC}" 
#   "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_ALWAYS})
if (CUDA_FOUND)
  plm_add_executable_v3 (cuda_probe "${CUDA_PROBE_SRC}" 
    "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_IF_NOT_DEBIAN})
endif ()

# if (MATLAB_FOUND)
#   mex_target (mex_drr "${MEX_DRR_SRC}"
#   "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}")
# endif ()

if (OpenCL_FOUND)
  plm_add_executable_v3 (opencl_probe "${OPENCL_PROBE_SRC}"
    "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_IF_NOT_DEBIAN})
endif ()

# dicom_uid requires dcmtk
if (DCMTK_FOUND)
  plm_add_executable_v3 (dicom_uid "${DICOM_UID_SRC}" 
    "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_IF_NOT_DEBIAN})
endif ()

