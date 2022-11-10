##-----------------------------------------------------------------------------
##  Welcome to the Plastimatch CMakeLists.txt file
##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (src)

##-----------------------------------------------------------------------------
##  Add subdirectories
##-----------------------------------------------------------------------------
if (PLM_CONFIG_DEBIAN_BUILD)
  set (PLM_CONFIG_ENABLE_FATM OFF)
endif ()

if (PLM_CONFIG_ENABLE_FATM)
  add_subdirectory (fatm)
endif ()

sb_set (PLM_USING_SUPERBUILD OFF)
if (PLM_CONFIG_ENABLE_PLASTIMATCH)
  if (NOT ITK_FOUND OR NOT DCMTK_FOUND)
    if (PLM_CONFIG_ENABLE_SUPERBUILD)
      set (proj plastimatch)
      set (PLM_USING_SUPERBUILD ON)
      if(CMAKE_EXTRA_GENERATOR)
	set(gen "${CMAKE_EXTRA_GENERATOR} - ${CMAKE_GENERATOR}")
      else()
	set(gen "${CMAKE_GENERATOR}")
      endif()

      set (sb_option_list "")
      foreach (_var ${sb_cmake_vars})
	list (APPEND sb_option_list "-D${_var}=${${_var}}")
      endforeach ()
      if (opt4D_FOUND)
	list (APPEND sb_option_list "-Dopt4D_DIR=${opt4D_DIR}")
      endif ()

      set (sb_dependencies devillard inih lbfgs nkidecompress specfun)
      if (NOT ITK_FOUND)
	list (APPEND sb_dependencies ITK)
      endif ()
      if (NOT DCMTK_FOUND AND PLM_CONFIG_ENABLE_DCMTK)
	list (APPEND sb_dependencies DCMTK)
      endif ()
      if (NOT SQLite3_FOUND)
	list (APPEND sb_dependencies sqlite3)
      endif ()

      ExternalProject_Add (${proj}
	DOWNLOAD_COMMAND ""
	INSTALL_COMMAND ""
	SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/plastimatch"
	BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/plastimatch"
	CMAKE_GENERATOR ${gen}
	CMAKE_ARGS
	-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
	-DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
	-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
	-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
	-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
	-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
	-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
	-DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
	-DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}
	-DCMAKE_OSX_SYSROOT:PATH=${CMAKE_OSX_SYSROOT}
	-DITK_DIR:PATH=${ITK_DIR}
	-DDCMTK_DIR:PATH=${DCMTK_DIR}
	-DPLASTIMATCH_VERSION_STRING:STRING=${PLASTIMATCH_VERSION_STRING}
	${sb_option_list}
	DEPENDS
	${sb_dependencies}
	)
    else ()
      message (STATUS "Plastimatch will not be built (ITK or DCMTK not found; superbuild disabled)")
    endif ()
  else ()
    add_subdirectory (plastimatch)
  endif ()
endif ()
