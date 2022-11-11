##-----------------------------------------------------------------------------
##  See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
##-----------------------------------------------------------------------------
project (CUDA_tex_test)

if (CUDA_FOUND)
  cuda_compile (CUDA_TEX_TEST_WRAPPERS 
    tex_kernels.cu
    tex_stubs.cu)
  set (CUDA_TEX_TEST_SRC
    ${CUDA_TEX_TEST_WRAPPERS}
    tex_kernels.h
    tex_stubs.h
    textures.c
    )
  plm_add_executable_v3 (cuda_tex_test "${CUDA_TEX_TEST_SRC}" 
    "" "${CUDA_LIBRARIES}" "" ${INSTALL_NEVER})
endif ()
