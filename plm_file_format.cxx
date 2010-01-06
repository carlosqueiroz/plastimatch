/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>
#include <itksys/SystemTools.hxx>
#include <itkImageIOBase.h>

#include "gdcm_rtss.h"
#include "itk_image.h"
#include "plm_file_format.h"

Plm_file_format
plm_file_format_deduce (char* path)
{
    std::string ext;
    
    if (itksys::SystemTools::FileIsDirectory (path)) {
	/* GCS TODO:  Distinguish between xio, dicom, rtog directories */
	return PLM_FILE_FMT_DICOM_DIR;
    }

    ext = itksys::SystemTools::GetFilenameLastExtension (std::string (path));

    if (!itksys::SystemTools::Strucmp (ext.c_str(), ".txt")) {
	/* Probe for pointset */
	int rc;
	const int MAX_LINE = 2048;
	char line[MAX_LINE];
	float f[4];
	FILE* fp = fopen (path, "rb");
	if (!fp) return PLM_FILE_FMT_NO_FILE;

	fgets (line, MAX_LINE, fp);
	fclose (fp);

	rc = sscanf (line, "%g %g %g %g", &f[0], &f[1], &f[2], &f[3]);
	if (rc == 3) {
	    return PLM_FILE_FMT_POINTSET;
	}

	/* Not sure, assume image */
	return PLM_FILE_FMT_IMG;
    }

    if (!itksys::SystemTools::Strucmp (ext.c_str(), ".cxt")) {
	return PLM_FILE_FMT_CXT;
    }

    if (!itksys::SystemTools::Strucmp (ext.c_str(), ".dij")) {
	return PLM_FILE_FMT_DIJ;
    }

    if (!itksys::SystemTools::Strucmp (ext.c_str(), ".pfm")) {
	return PLM_FILE_FMT_PROJ_IMG;
    }
    if (!itksys::SystemTools::Strucmp (ext.c_str(), ".hnd")) {
	return PLM_FILE_FMT_PROJ_IMG;
    }


    /* Maybe vector field? */
    itk::ImageIOBase::IOPixelType pixelType;
    itk::ImageIOBase::IOComponentType componentType;
    itk__GetImageType (std::string (path), pixelType, componentType);
    if (pixelType == itk::ImageIOBase::VECTOR) {
	return PLM_FILE_FMT_VF;
    }

    /* Maybe dicom rtss? */
    if (gdcm_rtss_probe (path)) {
	return PLM_FILE_FMT_DICOM_RTSS;
    }

    return PLM_FILE_FMT_IMG;
}

char*
plm_file_format_string (Plm_file_format file_type)
{
    switch (file_type) {
    case PLM_FILE_FMT_NO_FILE:
	return "No such file";
	break;
    case PLM_FILE_FMT_UNKNOWN:
	return "Unknown";
	break;
    case PLM_FILE_FMT_IMG:
	return "Image";
	break;
    case PLM_FILE_FMT_DIJ:
	return "Dij matrix";
	break;
    case PLM_FILE_FMT_POINTSET:
	return "Pointset";
	break;
    case PLM_FILE_FMT_CXT:
	return "Cxt file";
	break;
    case PLM_FILE_FMT_DICOM_DIR:
	return "Dicom directory";
	break;
    case PLM_FILE_FMT_XIO_DIR:
	return "XiO directory";
	break;
    case PLM_FILE_FMT_RTOG_DIR:
	return "RTOG directory";
	break;
    case PLM_FILE_FMT_PROJ_IMG:
	return "Projection image";
	break;
    default:
	return "Unknown/default";
	break;
    }
}

Plm_file_format 
plm_file_format_parse (const char* string)
{
    if (!strcmp (string, "mha")) {
	return PLM_FILE_FMT_IMG;
    }
    else if (!strcmp (string, "vf")) {
	return PLM_FILE_FMT_VF;
    }
    else if (!strcmp (string, "dij")) {
	return PLM_FILE_FMT_DIJ;
    }
    else if (!strcmp (string, "pointset")) {
	return PLM_FILE_FMT_POINTSET;
    }
    else if (!strcmp (string, "cxt")) {
	return PLM_FILE_FMT_CXT;
    }
    else if (!strcmp (string, "dicom") || !strcmp (string, "dicom-dir")) {
	return PLM_FILE_FMT_DICOM_DIR;
    }
    else if (!strcmp (string, "xio")) {
	return PLM_FILE_FMT_XIO_DIR;
    }
    else if (!strcmp (string, "rtog")) {
	return PLM_FILE_FMT_RTOG_DIR;
    }
    else if (!strcmp (string, "proj")) {
	return PLM_FILE_FMT_PROJ_IMG;
    }
    else if (!strcmp (string, "rtss") || !strcmp (string, "dicom-rtss")) {
	return PLM_FILE_FMT_DICOM_RTSS;
    }
    else {
	return PLM_FILE_FMT_UNKNOWN;
    }
}

