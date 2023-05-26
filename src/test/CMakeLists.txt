#-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src_plastimatch_test)

include_directories (BEFORE ${CMAKE_CURRENT_SOURCE_DIR})
include_directories (BEFORE ${PLM_BINARY_DIR})

if (CUDA_FOUND)
  add_subdirectory (cuda)
  include_directories (BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/cuda")
endif ()

##-----------------------------------------------------------------------------
##  CMAKE FLAGS
##-----------------------------------------------------------------------------
option (PLM_CONFIG_BUILD_TEST_PROGRAMS
    "Build extra test program executables" OFF)

##-----------------------------------------------------------------------------
##  BUILD TARGETS
##-----------------------------------------------------------------------------
# Test executable -- plastimatch api
if (PLM_CONFIG_BUILD_TEST_PROGRAMS)
    plm_add_executable_v3 (api_test api_test.cxx 
      "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})

    # Test executable -- c++ properties
    plm_add_executable_v3 (cpp_bitfield_test cpp_bitfield_test.cxx
      "" "" "" ${INSTALL_NEVER})
    plm_add_executable_v3 (cpp_eps_test cpp_eps_test.cxx
      "" "" "" ${INSTALL_NEVER})
    plm_add_executable_v3 (cpp_floating_point_test cpp_floating_point_test.cxx
      "" "" "" ${INSTALL_NEVER})
    plm_add_executable_v3 (cpp_limits_test cpp_limits_test.cxx
      "" "" "" ${INSTALL_NEVER})
    plm_add_executable_v3 (cpp_overflow_test cpp_overflow_test.cxx
      "" "" "" ${INSTALL_NEVER})
    plm_add_executable_v3 (cpp_template_test cpp_template_test.cxx
      "" "" "" ${INSTALL_NEVER})
    plm_add_executable_v3 (cpp_sizeof_test cpp_sizeof_test.cxx
      "" "" "" ${INSTALL_NEVER})

    # Test executable -- dcmtk
    if (ITK_FOUND AND DCMTK_FOUND)
      plm_add_executable_v3 (dcmtk_test dcmtk_test.cxx
	"" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
      plm_add_executable_v3 (rtplan_test rtplan_test.cxx 
	"" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
    endif ()

    # Test executable -- dlib
    plm_add_executable_v3 (dlib_test dlib_test.cxx
      "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
    plm_add_executable_v3 (dlib_thread_test dlib_thread_test.cxx
      "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})

    # Test executable -- gabor_test
    plm_add_executable_v3 (gabor_test gabor_test.cxx 
      "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})

    # Test executables -- itk
    if (ITK_FOUND)
      plm_add_executable_v3 (itk_test itk_test.cxx
	"" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
      plm_add_executable_v3 (itk_test_directions itk_test_directions.cxx
	"" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
      plm_add_executable_v3 (itk_test_threads itk_thread_test.cxx
	"" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
      plm_add_executable_v3 (itk_registration_v4 itk_registration_v4.cxx
	"" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
    endif ()

    # Test executable -- libyaml
    if (yaml_FOUND)
      plm_add_executable_v3 (libyaml_test libyaml_test.cxx
	"" "${yaml_LIBRARIES}" "" ${INSTALL_NEVER})
    endif ()

    # Test executable -- matlab
    if (MATLAB_FOUND)
	file (WRITE "${PLM_BINARY_DIR}/compile_mex_test.m"
	    "mex \"${CMAKE_SOURCE_DIR}/mex_test.c\";exit;\n")
	file (WRITE "${PLM_BINARY_DIR}/mex_test.cmake"
	    "EXECUTE_PROCESS (COMMAND ${MATLAB_EXE} -nosplash -nodesktop -nojvm
            -r compile_mex_test
            RESULT_VARIABLE RESULT
            OUTPUT_VARIABLE STDOUT
            ERROR_VARIABLE STDERR)\n")
	add_custom_command (
	    OUTPUT "${PLM_BINARY_DIR}/mex_test${MATLAB_MEXEXT}"
	    COMMAND ${CMAKE_COMMAND} -P "${PLM_BINARY_DIR}/mex_test.cmake"
	    DEPENDS "${CMAKE_SOURCE_DIR}/mex_test.c")
	add_custom_target (mat_mex_test
	    DEPENDS "${PLM_BINARY_DIR}/mex_test${MATLAB_MEXEXT}")
	target_link_libraries (mex_test ${MATLAB_LIBRARIES})
    endif ()

    # Test executable -- octave
    if (OCTAVE_FOUND)
	add_custom_command (
	    OUTPUT "${PLM_BINARY_DIR}/mex_test.mex"
	    COMMAND ${OCTAVE_MKOCTFILE} --mex 
	    -o "${PLM_BINARY_DIR}/mex_test.mex" 
	    "${CMAKE_SOURCE_DIR}/mex_test.c"
	    DEPENDS "${CMAKE_SOURCE_DIR}/mex_test.c")
	add_custom_target (oct_mex_test
	    DEPENDS "${PLM_BINARY_DIR}/mex_test.mex")
    endif ()

    # Test executable -- nlopt
    if (NLOPT_FOUND)
      set (NLOPT_TEST_LIBS ${GPUIT_LIBRARIES} ${nlopt_LIBRARIES})
      plm_add_executable_v3 (nlopt_test nlopt_test.c
	"" "${NLOPT_TEST_LIBS}" "${OPENMP_LDFLAGS}" ${INSTALL_NEVER})
      set_target_properties (nlopt_test PROPERTIES LINKER_LANGUAGE CXX)
    endif ()

    # Test executable -- nm
    plm_add_executable_v3 (nelder_mead_test nelder_mead_test.cxx
      "" "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}" ${INSTALL_NEVER})

    # Test executable -- opencl
    if (OPENCL_FOUND)
      set (OPENCL_TEST_SRC
	opencl_test.cxx opencl_test.h opencl_test.cl
	"${PLM_BINARY_DIR}/opencl_test.cl"
	)
      set (OPENCL_TEST_LIBS ${OPENCL_LIBRARIES} ${PLASTIMATCH_LIBS})
      plm_add_executable_v3 (opencl_test "${OPENCL_TEST_SRC}" 
	"" "${OPENCL_TEST_LIBS}" "" ${INSTALL_NEVER})

      # I don't yet know how to bundle the .cl file within the executable.
      # Therefore, copy the .cl into binary directory.
      add_custom_command (
	OUTPUT "${PLM_BINARY_DIR}/opencl_test.cl" 
	COMMAND ${CMAKE_COMMAND} "-E" "copy" 
	"${CMAKE_CURRENT_SOURCE_DIR}/opencl_test.cl" 
	"${PLM_BINARY_DIR}/opencl_test.cl" 
	DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/opencl_test.cl")
    endif ()

    # Test executable -- openmp
    if (OPENMP_FOUND)
      set_source_files_properties (openmp_test.cxx
	PROPERTIES COMPILE_FLAGS ${OPENMP_FLAGS})
      plm_add_executable_v3 (openmp_test openmp_test.cxx
	"" "${PLASTIMATCH_LIBS}" "${OPENMP_LDFLAGS}" ${INSTALL_NEVER})
    endif ()

    # Test executable -- plm
    plm_add_executable_v3 (plm_restart_test plm_restart_test.cxx
      "" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})

    # Test executable -- qt
    if (Qt5_FOUND)
      set (QT5_TEST_LIBRARIES Qt5::Network Qt5::Widgets Qt5::Gui Qt5::Core)
      plm_add_executable_v3 (qt5_test qt_test.cxx
	"" "${QT5_TEST_LIBRARIES}" "" ${INSTALL_NEVER})
    endif ()
    if (QT4_FOUND)
      set (QT4_TEST_INCLUDES ${PLASTIMATCH_INCLUDES} ${QT_QTGUI_INCLUDE_DIR}
	${QT_INCLUDE_DIR} ${QT_QTCORE_INCLUDE_DIR})
      set (QT4_TEST_LIBRARIES ${QT_QTGUI_LIBRARY} ${QT_QTCORE_LIBRARY})
      plm_add_executable_v3 (qt4_test qt_test.cxx
	"${QT4_TEST_INCLUDES}" "${QT4_TEST_LIBRARIES}" "" ${INSTALL_NEVER})
    endif ()

    # Test executable -- ransac
    plm_add_executable_v3 (ransac_test ransac_test.cxx
      "" "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" ${INSTALL_NEVER})

    # Test executable -- rapidjson
    if (RAPIDJSON_FOUND)
      plm_add_executable_v3 (rapidjson_test rapidjson_test.cxx
	"" "${PLASTIMATCH_LIBS}" "${PLASTIMATCH_LDFLAGS}" ${INSTALL_NEVER})
    endif ()

    # Test executable -- smart pointers I
    ## find_package (TR1)
    plm_add_executable_v3 (smartp_test_i smartp_test_i.cxx
      "" "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" ${INSTALL_NEVER})

    # Test executable -- smart pointers II
    plm_add_executable_v3 (smartp_test_ii smartp_test_ii.cxx
      "" "${PLASTIMATCH_LIBS}" "${GPUIT_EXE_LDFLAGS}" ${INSTALL_NEVER})
endif ()
