#-----------------------------------------------------------------------------
# Baixa e compila o DCMTK (versão 3.6.9) sem verificação de checksum
#-----------------------------------------------------------------------------

set(proj DCMTK)
set(dcmtk_url "https://dicom.offis.de/download/dcmtk/dcmtk361/dcmtk-3.6.1_20170228.tar.gz")

ExternalProject_Add(${proj}
  DOWNLOAD_DIR ${proj}-download
  URL ${dcmtk_url}
  # Linha de checksum removida
  SOURCE_DIR ${proj}
  BINARY_DIR ${proj}-build
  CMAKE_GENERATOR ${gen}
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
    # -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
    -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
    -DBUILD_APPS:BOOL=OFF
    # Forçando biblioteca estática; para usar biblioteca compartilhada, altere para ON
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DDCMTK_OVERWRITE_WIN32_COMPILER_FLAGS:BOOL=OFF
  INSTALL_COMMAND ""
)

# Define a variável DCMTK_DIR para apontar para o diretório de build do DCMTK
set(DCMTK_DIR ${CMAKE_BINARY_DIR}/${proj}-build)