.. _building_plastimatch:

Building plastimatch from source
================================

Downloading the code
--------------------
To download from git, use the following command::

  git clone https://gitlab.com/plastimatch/plastimatch.git

If you are using Windows, you will need a git client.
We recommend cygwin (http://cygwin.com) for command-line users, 
and TortoiseGit (http://tortoisegit.org) for graphical users.
Starting with Windows 10, you can use the Ubuntu bash
shell for git client.

If you have already downloaded a previous version, 
you can update to the latest version by executing the following command 
from within the plastimatch source directory::

  git pull

Build dependencies
------------------

Debian install
^^^^^^^^^^^^^^
On debian, all the needed dependencies are already included.
The following command will install all the needed prerequisites.::

   sudo apt-get install g++ make git cmake-curses-gui \
     libblas-dev liblapack-dev libsqlite3-dev \
     libdcmtk-dev libdlib-dev libfftw3-dev \
     libinsighttoolkit5-dev \
     libpng-dev libtiff-dev uuid-dev zlib1g-dev 

Cmake (required)
^^^^^^^^^^^^^^^^
Plastimatch uses cmake, so you must download and install cmake 
before you can build plastimatch.  Download from here:

  http://cmake.org/

Cmake 3.1.3 or higher is required.

C/C++ Compiler (required)
^^^^^^^^^^^^^^^^^^^^^^^^^
You will need a C/C++ compiler.  If you are running 
Windows, we recommend Microsoft Visual Studio.
We use the free Visual Studio Community 2017 for development.
You can download it from here:

  https://visualstudio.microsoft.com/

You may also use the MinGW compiler.

On OSX, you need the Xcode package, and you must also install the 
command line tools.  
If you wish to use g++ instead of clang, do something like 
the following:

  CC=/usr/bin/gcc CXX=/usr/bin/g++ ccmake /path/to/plastimatch/

ITK (required)
^^^^^^^^^^^^^^
ITK is required.  Get ITK from here:

  http://itk.org/

We currently support ITK 4.12.2 and greater.  Older versions might work,
but are not tested.

When you build ITK, the following settings are recommended or required::

  CMAKE_BUILD_TYPE                          Release
  BUILD_EXAMPLES                            OFF
  BUILD_SHARED_LIBS                         (EITHER)
  BUILD_TESTING                             OFF
  Module_ITKReview                          ON


DCMTK (optional)
^^^^^^^^^^^^^^^^
DCMTK is needed for DICOM-RT support.
Version 3.6.2 or higher are recommended.  On linux, feel free to 
use the dcmtk that comes from your package manager (that's what I do).

There are special considerations to building dcmtk:

#. PNG, TIFF, and ZLIB are not required
#. On linux x86_64 platforms, you need to add -fPIC to 
   CMAKE_CXX_FLAGS and CMAKE_C_FLAGS, or eqivalently,
   set DCMTK_FORCE_FPIC_ON_UNIX to ON
#. On windows, you need to set DCMTK_OVERWRITE_WIN32_COMPILER_FLAGS to OFF
#. When you run cmake on plastimatch, set DCMTK_DIR to the build directory

CUDA (optional)
^^^^^^^^^^^^^^^
CUDA is needed if you want GPU acceleration of the DRR, FDK, and B-Spline 
registration codes.  
You need to install the driver and toolkit, but the SDK is not needed.

Please note that CUDA is constantly evolving in order to provide new
high performance computing features. 
The following table will help you with selecting the
correct CUDA version to install/upgrade::

  CUDA <= 2.X           Not supported
  CUDA >= 3.X           Supported
  CUDA >= 5.0           Supported, Required for Kepler

Download CUDA from here:

  http://developer.nvidia.com/object/cuda_archive.html

FFTW (optional)
^^^^^^^^^^^^^^^
The FFTW library is used to implement the ramp filter for FDK 
cone-beam reconstruction.  So if you are not using the FDK code, 
you don't need this.  We recommend the most current version of FFTW 3.

  http://www.fftw.org/

On windows, the precompiled DLLs work fine.  
However, you do need to create the import libraries.  
See this page for details:

  http://www.fftw.org/install/windows.html  

WxWidgets (optional)
^^^^^^^^^^^^^^^^^^^^
WxWidgets is needed if you want to build "Mondoshot", the dicom screen 
capture program.  Download WxWidgets from here:

  http://wxwidgets.org

Compiling plastimatch (Windows)
-------------------------------
Before compiling plastimatch, compile or install the desired 
prerequisites.  At a minimum, you must compile required 
packages such as ITK.  Be sure to build ITK and plastimatch 
using the same build type (e.g. both as Debug, or both as Release).

Run CMake as follows:

#. Select source directory and binary directory
#. Click configure
#. Select makefile format (e.g. MS VC 2005)
#. <CMake configures>
#. Set the ITK directory (sometimes it might be found automatically)
#. Set directories for optional components (such as slicer)
#. Click configure
#. <CMake configures>
#. Click OK
#. <CMake generates>

Then build in Visual Studio as follows:

#. Navigate to your binary directory
#. Open the project file plastimatch.sln into MSVC.  
#. Change the build type (e.g. release, debug) to match ITK (and other 
   dependencies.  You probably want release.
#. Click "Build Solution".  Let the project build.

Special instructions for running cmake with MSYS/gcc on Windows
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There is a trick to building with MSYS/gcc.  
The trick is that you need to run the win32 cmake from 
the MSYS command line instead of the GUI.  For example, here is 
the command that I use::

   $ mkdir /c/gcs6/build/plastimatch-mingw
   $ cd /c/gcs6/build/plastimatch-mingw
   $ /c/Program\ Files/CMake\ 2.8/bin/cmake \
       -DITK_DIR=/c/gcs6/build/itk-mingw \
       -G"MSYS Makefiles" \
       /c/gcs6/projects/plastimatch

Then, edit CMakeCache.txt to set your options.  Re-run cmake 
to create the MSYS Makefile, and then run make to build.

Compiling plastimatch (Unix)
----------------------------

Build plastimatch as follows:

#. mkdir /path/to/build/files; cd /path/to/build/files
#. ccmake /path/to/source/files
#. Type "c" to configure
#. <CMake configures>
#. Set the ITK directory (it may be found automatically)
#. Set directories for other optional components (if necessary)
#. Type "c" to configure
#. <CMake configures>
#. Type "g" to generate
#. <CMake generates>
#. Type "make"

Users with multicore systems can speed up the process of compiling
plastimatch considerably by invoking make with the -j option.  For
example, a user with a dual-core system would type:

   make -j 2

whereas a user with an eight core system would type:

   make -j 8

You can probably get even better performance by increasing the 
the number of processes (specified by the -j option) 
beyond the number of cores.  One rule of thumb is to 
use approximately 1.5 times the number of available CPUs (see 
`[1] <http://developers.sun.com/solaris/articles/parallel_make.html#3>`_,
`[2] <http://stackoverflow.com/questions/414714/compiling-with-g-using-multiple-cores>`_).
