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
plm_add_static_library (
  plmclp
  "${PLMCLP_LIBRARY_SRC}"
  ""
  ""
  ""
)
target_include_directories (plmclp 
	PRIVATE 
	"${DLIB_INCLUDE_DIRS}"
	"${PLM_SOURCE_DIR}/src/sys"
	"${PLM_BINARY_DIR}"
	)

