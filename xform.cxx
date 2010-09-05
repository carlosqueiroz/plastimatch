/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "itkArray.h"
#include "itkResampleImageFilter.h"
#include "itkBSplineResampleImageFunction.h"
#include "itkTransformFileWriter.h"
#include "itkTransformFileReader.h"
#include "bspline.h"
#include "math_util.h"
#include "plm_image.h"
#include "plm_image_header.h"
#include "plm_parms.h"
#include "print_and_exit.h"
#include "mha_io.h"
#include "resample_mha.h"
#include "volume.h"
#include "xform.h"

static void init_itk_bsp_default (Xform *xf);
static void
itk_bsp_set_grid (Xform *xf,
    const BsplineTransformType::OriginType bsp_origin,
    const BsplineTransformType::SpacingType bsp_spacing,
    const BsplineTransformType::RegionType bsp_region,
    const BsplineTransformType::DirectionType bsp_direction);
static void
itk_bsp_set_grid_img (Xform *xf,
    const Plm_image_header* pih,
    float* grid_spac);
static void load_gpuit_bsp (Xform *xf, const char* fn);
static void itk_xform_load (Xform *xf, const char* fn);

/* -----------------------------------------------------------------------
   Utility functions
   ----------------------------------------------------------------------- */
static int
strcmp_alt (const char* s1, const char* s2)
{
  return strncmp (s1, s2, strlen(s2));
}

static int
get_parms (FILE* fp, itk::Array<double>* parms, int num_parms)
{
    float f;
    int r, s;

    s = 0;
    while ((r = fscanf (fp, "%f",&f))) {
	(*parms)[s++] = (double) f;
	if (s == num_parms) break;
	if (!fp) break;
    }
    return s;
}

/* -----------------------------------------------------------------------
   Load/save functions
   ----------------------------------------------------------------------- */
void
xform_load (Xform *xf, const char* fn)
{
    char buf[1024];
    FILE* fp;

    fp = fopen (fn, "r");
    if (!fp) {
	print_and_exit ("Error: xf_in file %s not found\n", fn);
    }
    if (!fgets(buf,1024,fp)) {
	print_and_exit ("Error reading from xf_in file.\n");
    }

    if (strcmp_alt (buf,"ObjectType = MGH_XFORM_TRANSLATION") == 0) {
	TranslationTransformType::Pointer trn = TranslationTransformType::New();
	TranslationTransformType::ParametersType xfp(12);
	int num_parms;

	num_parms = get_parms (fp, &xfp, 3);
	if (num_parms != 3) {
	    print_and_exit ("Wrong number of parameters in xf_in file.\n");
	} else {
	    trn->SetParameters(xfp);
#if defined (commentout)
	    std::cout << "Initial translation parms = " << trn << std::endl;
#endif
	}
	xf->set_trn (trn);
	fclose (fp);
    } else if (strcmp_alt(buf,"ObjectType = MGH_XFORM_VERSOR")==0) {
	VersorTransformType::Pointer vrs = VersorTransformType::New();
	VersorTransformType::ParametersType xfp(6);
	int num_parms;

	num_parms = get_parms (fp, &xfp, 6);
	if (num_parms != 6) {
	    print_and_exit ("Wrong number of parameters in xf_in file.\n");
	} else {
	    vrs->SetParameters(xfp);
#if defined (commentout)
	    std::cout << "Initial versor parms = " << vrs << std::endl;
#endif
	}
	xf->set_vrs (vrs);
	fclose (fp);
    } else if (strcmp_alt(buf,"ObjectType = MGH_XFORM_AFFINE")==0) {
	AffineTransformType::Pointer aff = AffineTransformType::New();
	AffineTransformType::ParametersType xfp(12);
	int num_parms;

	num_parms = get_parms (fp, &xfp, 12);
	if (num_parms != 12) {
	    print_and_exit ("Wrong number of parameters in xf_in file.\n");
	} else {
	    aff->SetParameters(xfp);
#if defined (commentout)
	    std::cout << "Initial affine parms = " << aff << std::endl;
#endif
	}
	xf->set_aff (aff);
	fclose (fp);
    } else if (strcmp_alt (buf, "#Insight Transform File V1.0") == 0) {
	fclose(fp);
	itk_xform_load (xf, fn);
	
	//float f;
	//int p, s, s1, rc;
	//int num_parms = 12;
	//AffineTransformType::Pointer aff = AffineTransformType::New();
	//AffineTransformType::ParametersType xfp(12);

	///* Skip 2 lines */
	//fgets (buf, 1024, fp);
	//fgets (buf, 1024, fp);  /* GCS FIX: need to test if the file is actually affine! */

	///* Read line with parameters */
	//fgets (buf, 1024, fp);

	///* Find beginning of parameters */
	//rc = sscanf (buf, "Parameters: %n%f", &s, &f);
	//if (rc != 1) {
	//    print_and_exit ("Error parsing ITK-format xform file.\n");
	//}

	//p = 0;
	//while ((rc = sscanf (&buf[s], "%f%n", &f, &s1)) == 1) {
	//    xfp[p++] = (double) f;
	//    if (p == num_parms) break;
	//    s += s1;
	//}

	//if (p != 12) {
	//    print_and_exit ("Wrong number of parameters in ITK xform file.\n");
	//} else {
	//    aff->SetParameters(xfp);

	//#if defined (commentout)
	//	    std::cout << "Initial affine parms = " << aff << std::endl;
	//#endif
	//	}
	//	xf->set_aff (aff);
	//	fclose (fp);
    } else if (strcmp_alt (buf, "ObjectType = MGH_XFORM_BSPLINE") == 0) {
	int s[3];
	float p[3];
	BsplineTransformType::RegionType::SizeType bsp_size;
	BsplineTransformType::RegionType bsp_region;
	BsplineTransformType::SpacingType bsp_spacing;
	BsplineTransformType::OriginType bsp_origin;
	BsplineTransformType::DirectionType bsp_direction;

	/* Initialize direction cosines to identity */
	bsp_direction[0][0] = bsp_direction[1][1] = bsp_direction[2][2] = 1.0;

	/* Create the bspline structure */
	init_itk_bsp_default (xf);

	/* Skip 2 lines */
	fgets(buf,1024,fp);
	fgets(buf,1024,fp);

	/* Load bulk transform, if it exists */
	fgets(buf,1024,fp);
	if (!strncmp ("BulkTransform", buf, strlen("BulkTransform"))) {
	    TranslationTransformType::Pointer trn = TranslationTransformType::New();
	    VersorTransformType::Pointer vrs = VersorTransformType::New();
	    AffineTransformType::Pointer aff = AffineTransformType::New();
	    itk::Array <double> xfp(12);
	    float f;
	    int n, num_parm = 0;
	    char *p = buf + strlen("BulkTransform = ");
	    while (sscanf (p, " %g%n", &f, &n) > 0) {
		if (num_parm>=12) {
		    print_and_exit ("Error loading bulk transform\n");
		}
		xfp[num_parm] = f;
		p += n;
		num_parm++;
	    }
	    if (num_parm == 12) {
		aff->SetParameters(xfp);
#if defined (commentout)
		std::cout << "Bulk affine = " << aff;
#endif
		xf->get_bsp()->SetBulkTransform (aff);
	    } else if (num_parm == 6) {
		vrs->SetParameters(xfp);
#if defined (commentout)
		std::cout << "Bulk versor = " << vrs;
#endif
		xf->get_bsp()->SetBulkTransform (vrs);
	    } else if (num_parm == 3) {
		trn->SetParameters(xfp);
#if defined (commentout)
		std::cout << "Bulk translation = " << trn;
#endif
		xf->get_bsp()->SetBulkTransform (trn);
	    } else {
		print_and_exit ("Error loading bulk transform\n");
	    }
	    fgets(buf,1024,fp);
	}

	/* Load origin, spacing, size */
	if (3 != sscanf(buf,"Offset = %g %g %g",&p[0],&p[1],&p[2])) {
	    print_and_exit ("Unexpected line in xform_in file.\n");
	}
	bsp_origin[0] = p[0]; bsp_origin[1] = p[1]; bsp_origin[2] = p[2];

	fgets(buf,1024,fp);
	if (3 != sscanf(buf,"ElementSpacing = %g %g %g",&p[0],&p[1],&p[2])) {
	    print_and_exit ("Unexpected line in xform_in file.\n");
	}
	bsp_spacing[0] = p[0]; bsp_spacing[1] = p[1]; bsp_spacing[2] = p[2];

	fgets(buf,1024,fp);
	if (3 != sscanf(buf,"DimSize = %d %d %d",&s[0],&s[1],&s[2])) {
	    print_and_exit ("Unexpected line in xform_in file.\n");
	}
	bsp_size[0] = s[0]; bsp_size[1] = s[1]; bsp_size[2] = s[2];

#if defined (commentout)
	std::cout << "Offset = " << origin << std::endl;
	std::cout << "Spacing = " << spacing << std::endl;
	std::cout << "Size = " << size << std::endl;
#endif

	fgets(buf,1024,fp);
	if (strcmp_alt (buf, "ElementDataFile = LOCAL")) {
	    print_and_exit ("Error: bspline xf_in failed sanity check\n");
	}

	/* Set the BSpline grid to specified parameters */
	bsp_region.SetSize (bsp_size);
	itk_bsp_set_grid (xf, bsp_origin, bsp_spacing, bsp_region, bsp_direction);

	/* Read bspline coefficients from file */
	const unsigned int num_parms = xf->get_bsp()->GetNumberOfParameters();
	BsplineTransformType::ParametersType bsp_coeff;
	bsp_coeff.SetSize (num_parms);
	for (unsigned int i = 0; i < num_parms; i++) {
	    float d;
	    if (!fgets(buf,1024,fp)) {
		print_and_exit ("Missing bspline coefficient from xform_in file.\n");
	    }
	    if (1 != sscanf(buf,"%g",&d)) {
		print_and_exit ("Bad bspline parm in xform_in file.\n");
	    }
	    bsp_coeff[i] = d;
	}
	fclose (fp);

	/* Copy into bsp structure */
	xf->get_bsp()->SetParametersByValue (bsp_coeff);

    } else if (strcmp_alt(buf,"MGH_GPUIT_BSP <experimental>")==0) {
	fclose (fp);
	load_gpuit_bsp (xf, fn);
    } else {
	/* Close the file and try again, it is probably a vector field */
	fclose (fp);
	DeformationFieldType::Pointer vf = DeformationFieldType::New();
	vf = itk_image_load_float_field (fn);
	if (!vf) {
	    print_and_exit ("Unexpected file format for xf_in file.\n");
	}
	xf->set_itk_vf (vf);
    }
}

static void
load_gpuit_bsp (Xform *xf, const char* fn)
{
    Bspline_xform* bxf;

    bxf = read_bxf (fn);
    if (!bxf) {
	print_and_exit ("Error loading bxf format file: %s\n", fn);
    }

    /* Tell Xform that it is gpuit_bsp */
    xf->set_gpuit_bsp (bxf);
}

template< class T >
static void
itk_xform_save (T transform, char *filename)
{
    typedef itk::TransformFileWriter TransformWriterType;
    TransformWriterType::Pointer outputTransformWriter;
	
    outputTransformWriter= TransformWriterType::New();
    outputTransformWriter->SetFileName( filename );
    outputTransformWriter->SetInput( transform );
    try
    {
	outputTransformWriter->Update();
    }
    catch (itk::ExceptionObject &err)
    {
	std::cerr << err << std::endl;
	print_and_exit ("Error writing file: %s\n", filename);
    }
}

static void
itk_xform_load (Xform *xf, const char* fn)
{
    /* Load from file to into reader */
    itk::TransformFileReader::Pointer transfReader;
    transfReader = itk::TransformFileReader::New();
    transfReader->SetFileName(fn);
    try {
	transfReader->Update();
    }
    catch( itk::ExceptionObject & excp ) {
	print_and_exit ("Error reading ITK transform file: %s\n", fn);
    }
 
    /* Confirm that there is only one xform in the file */
    typedef itk::TransformFileReader::TransformListType* TransformListType;
    TransformListType transfList = transfReader->GetTransformList();
    std::cout << "Number of transforms = " << transfList->size() << std::endl;
    if (transfList->size() != 1) {
	print_and_exit ("Error. ITK transform file has multiple "
	    "transforms: %s\n", fn);
    }

    /* Deduce transform type, and copy into Xform structure */
    itk::TransformFileReader::TransformListType::const_iterator 
	itTrasf = transfList->begin();
    if (!strcmp((*itTrasf)->GetNameOfClass(), "TranslationTransform")) {
	TranslationTransformType::Pointer itk_xf
	    = TranslationTransformType::New();
	itk_xf = static_cast<TranslationTransformType*>(
	    (*itTrasf).GetPointer());
	xf->set_trn (itk_xf);
    }
    else if (!strcmp((*itTrasf)->GetNameOfClass(), "VersorRigid3DTransform"))
    {
	VersorTransformType::Pointer itk_xf
	    = VersorTransformType::New();
	VersorTransformType::InputPointType cor;
	cor.Fill(12);
	itk_xf->SetCenter(cor);
	itk_xf = static_cast<VersorTransformType*>(
	    (*itTrasf).GetPointer());
	xf->set_vrs (itk_xf);
    }
    else if (!strcmp((*itTrasf)->GetNameOfClass(), "QuaternionRigidTransform"))
    {
	QuaternionTransformType::Pointer quatTransf 
	    = QuaternionTransformType::New();
	QuaternionTransformType::InputPointType cor;
	cor.Fill(12);
	quatTransf->SetCenter(cor);
	quatTransf = static_cast<QuaternionTransformType*>(
	    (*itTrasf).GetPointer());
	//quatTransf->Print(std::cout);
	xf->set_quat (quatTransf);
    }
    else if (!strcmp((*itTrasf)->GetNameOfClass(), "AffineTransform"))
    {
	AffineTransformType::Pointer affineTransf
	    = AffineTransformType::New();
	AffineTransformType::InputPointType cor;
	cor.Fill(12);
	affineTransf->SetCenter(cor);
	affineTransf = static_cast<AffineTransformType*>(
	    (*itTrasf).GetPointer());
	//affineTransf->Print(std::cout);
	xf->set_aff (affineTransf);
    }
}

#if defined (commentout)
void
xform_save_translation (TranslationTransformType::Pointer transform, char* filename)
{
    FILE* fp = fopen (filename,"w");
    if (!fp) {
	printf ("Error: Couldn't open file %s for write\n", filename);
	return;
    }

    fprintf (fp,"ObjectType = MGH_XFORM_TRANSLATION\n");
    for (unsigned int i = 0; i < transform->GetNumberOfParameters(); i++) {
	fprintf (fp, "%g\n", transform->GetParameters()[i]);
    }
    fclose (fp);
}

void
xform_save_versor (VersorTransformType::Pointer transform, char* filename)
{
    FILE* fp = fopen (filename,"w");
    if (!fp) {
	printf ("Error: Couldn't open file %s for write\n", filename);
	return;
    }

    fprintf (fp,"ObjectType = MGH_XFORM_VERSOR\n");
    for (unsigned int i = 0; i < transform->GetNumberOfParameters(); i++) {
	fprintf (fp, "%g\n", transform->GetParameters()[i]);
    }
    fclose (fp);
}

void
xform_save_affine (AffineTransformType::Pointer transform, char* filename)
{
    FILE* fp = fopen (filename,"w");
    if (!fp) {
	printf ("Error: Couldn't open file %s for write\n", filename);
	return;
    }

    fprintf (fp,"ObjectType = MGH_XFORM_AFFINE\n");
    for (unsigned int i = 0; i < transform->GetNumberOfParameters(); i++) {
	fprintf (fp, "%g\n", transform->GetParameters()[i]);
    }
    fclose (fp);
}
#endif

void
xform_save_itk_bsp (BsplineTransformType::Pointer transform, char* filename)
{
    FILE* fp = fopen (filename,"w");
    if (!fp) {
	printf ("Error: Couldn't open file %s for write\n", filename);
	return;
    }

    fprintf (fp,
	     "ObjectType = MGH_XFORM_BSPLINE\n"
	     "NDims = 3\n"
	     "BinaryData = False\n");
    if (transform->GetBulkTransform()) {
	if ((!strcmp("TranslationTransform", transform->GetBulkTransform()->GetNameOfClass())) ||
	    (!strcmp("AffineTransform", transform->GetBulkTransform()->GetNameOfClass())) ||
	    (!strcmp("VersorTransform", transform->GetBulkTransform()->GetNameOfClass())) ||
	    (!strcmp("VersorRigid3DTransform", transform->GetBulkTransform()->GetNameOfClass()))) {
	    fprintf (fp, "BulkTransform =");
	    for (unsigned int i = 0; i < transform->GetBulkTransform()->GetNumberOfParameters(); i++) {
		fprintf (fp, " %g", transform->GetBulkTransform()->GetParameters()[i]);
	    }
	    fprintf (fp, "\n");
	} else if (strcmp("IdentityTransform", transform->GetBulkTransform()->GetNameOfClass())) {
	    printf("Warning!!! BulkTransform exists. Type=%s\n", transform->GetBulkTransform()->GetNameOfClass());
	    printf(" # of parameters=%d\n", transform->GetBulkTransform()->GetNumberOfParameters());
	    printf(" The code currently does not know how to handle this type and will not write the parameters out!\n");
	}
    }

    fprintf (fp,
	     "Offset = %f %f %f\n"
	     "ElementSpacing = %f %f %f\n"
	     "DimSize = %lu %lu %lu\n"
	     "ElementDataFile = LOCAL\n",
	     transform->GetGridOrigin()[0],
	     transform->GetGridOrigin()[1],
	     transform->GetGridOrigin()[2],
	     transform->GetGridSpacing()[0],
	     transform->GetGridSpacing()[1],
	     transform->GetGridSpacing()[2],
	     transform->GetGridRegion().GetSize()[0],
	     transform->GetGridRegion().GetSize()[1],
	     transform->GetGridRegion().GetSize()[2]
	     );
    for (unsigned int i = 0; i < transform->GetNumberOfParameters(); i++) {
	fprintf (fp, "%g\n", transform->GetParameters()[i]);
    }
    fclose (fp);
}

void
xform_save (Xform *xf, char* fn)
{
    switch (xf->m_type) {
    case XFORM_ITK_TRANSLATION:
	itk_xform_save (xf->get_trn(), fn);
	break;
    case XFORM_ITK_VERSOR:
	itk_xform_save (xf->get_vrs(), fn);
	break;
    case XFORM_ITK_QUATERNION:
	itk_xform_save (xf->get_quat(), fn);
	break;
    case XFORM_ITK_AFFINE:
	itk_xform_save (xf->get_aff(), fn);
	break;
    case XFORM_ITK_BSPLINE:
	xform_save_itk_bsp (xf->get_bsp(), fn);
	break;
    case XFORM_ITK_VECTOR_FIELD:
	itk_image_save (xf->get_itk_vf(), fn);
	break;
    case XFORM_GPUIT_BSPLINE:
	write_bxf (fn, xf->get_gpuit_bsp());
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	write_mha (fn, xf->get_gpuit_vf());
	break;
    case XFORM_NONE:
	print_and_exit ("Error trying to save null transform\n");
	break;
    default:
	print_and_exit ("Unhandled case trying to save transform\n");
	break;
    }
}

#if defined (GCS_REARRANGING_STUFF)
void
init_versor_moments (RegistrationType::Pointer registration,
		       VersorTransformType* versor)
{
    typedef itk::CenteredTransformInitializer < VersorTransformType,
	FloatImageType, FloatImageType > TransformInitializerType;
    TransformInitializerType::Pointer initializer =
	TransformInitializerType::New();

    initializer->SetTransform(versor);
    initializer->SetFixedImage(registration->GetFixedImage());
    initializer->SetMovingImage(registration->GetMovingImage());

    initializer->GeometryOn();

    printf ("Calling Initialize Transform\n");
    initializer->InitializeTransform();

    std::cout << "Transform is " << registration->GetTransform()->GetParameters() << std::endl;
}
#endif /* GCS_REARRANGING_STUFF */

/* -----------------------------------------------------------------------
   Transform points
   ----------------------------------------------------------------------- */
void
xform_transform_point (FloatPointType* point_out, Xform* xf_in, FloatPointType point_in)
{
    DeformationFieldType::Pointer vf = xf_in->get_itk_vf ();
    DeformationFieldType::IndexType idx;

    bool isInside = vf->TransformPhysicalPointToIndex (point_in, idx);
    if (isInside) {
	DeformationFieldType::PixelType pixelValue = vf->GetPixel (idx);
	printf ("pi [%g %g %g]\n", point_in[0], point_in[1], point_in[2]);
	printf ("idx [%ld %ld %ld]\n", idx[0], idx[1], idx[2]);
	printf ("vf [%g %g %g]\n", pixelValue[0], pixelValue[1], pixelValue[2]);
	for (int d = 0; d < 3; d++) {
	    (*point_out)[d] = point_in[d] + pixelValue[d];
	}
	printf ("po [%g %g %g]\n", (*point_out)[0], (*point_out)[1], (*point_out)[2]);
    } else {
	(*point_out) = point_in;
    }
}

/* -----------------------------------------------------------------------
   Defaults
   ----------------------------------------------------------------------- */
static void
init_translation_default (Xform *xf_out)
{
    TranslationTransformType::Pointer trn = TranslationTransformType::New();
    xf_out->set_trn (trn);
}

static void
init_versor_default (Xform *xf_out)
{
    VersorTransformType::Pointer vrs = VersorTransformType::New();
    xf_out->set_vrs (vrs);
}

static void
init_quaternion_default (Xform *xf_out)
{
    QuaternionTransformType::Pointer quat = QuaternionTransformType::New();
    xf_out->set_quat (quat);
}

static void
init_affine_default (Xform *xf_out)
{
    AffineTransformType::Pointer aff = AffineTransformType::New();
    xf_out->set_aff (aff);
}

static void
init_itk_bsp_default (Xform *xf)
{
    BsplineTransformType::Pointer bsp = BsplineTransformType::New();
    xf->set_itk_bsp (bsp);
}

/* -----------------------------------------------------------------------
   Conversions for trn, vrs, aff
   ----------------------------------------------------------------------- */
void
xform_trn_to_vrs (Xform *xf_out, Xform* xf_in)
{
    init_versor_default (xf_out);
    xf_out->get_vrs()->SetOffset(xf_in->get_trn()->GetOffset());
}

void
xform_trn_to_aff (Xform *xf_out, Xform* xf_in)
{
    init_affine_default (xf_out);
    xf_out->get_aff()->SetOffset(xf_in->get_trn()->GetOffset());
}

void
xform_vrs_to_quat (Xform *xf_out, Xform* xf_in)
{
    init_quaternion_default (xf_out);
    xf_out->get_quat()->SetMatrix(xf_in->get_vrs()->GetRotationMatrix());
    xf_out->get_quat()->SetOffset(xf_in->get_vrs()->GetOffset());
}

void
xform_vrs_to_aff (Xform *xf_out, Xform* xf_in)
{
    init_affine_default (xf_out);
    xf_out->get_aff()->SetMatrix(xf_in->get_vrs()->GetRotationMatrix());
    xf_out->get_aff()->SetOffset(xf_in->get_vrs()->GetOffset());
}

/* -----------------------------------------------------------------------
   Conversion to itk_bsp
   ----------------------------------------------------------------------- */
/* Initialize using bspline spacing */
static void
itk_bsp_set_grid (Xform *xf,
	    const BsplineTransformType::OriginType bsp_origin,
	    const BsplineTransformType::SpacingType bsp_spacing,
	    const BsplineTransformType::RegionType bsp_region,
	    const BsplineTransformType::DirectionType bsp_direction)
{
    printf ("Setting bsp_spacing\n");
    std::cout << bsp_spacing << std::endl;
    printf ("Setting bsp_origin\n");
    std::cout << bsp_origin << std::endl;
    printf ("Setting bsp_region\n");
    std::cout << bsp_region;
    printf ("Setting bsp_direction = ");
    for (int d1 = 0; d1 < 3; d1++) {
	for (int d2 = 0; d2 < 3; d2++) {
	    printf ("%g ", bsp_direction[d1][d2]);
	}
    }
    printf ("\n");
#if defined (commentout)
#endif

    /* Set grid specifications to bsp struct */
    xf->get_bsp()->SetGridSpacing (bsp_spacing);
    xf->get_bsp()->SetGridOrigin (bsp_origin);
    xf->get_bsp()->SetGridRegion (bsp_region);
    xf->get_bsp()->SetIdentity ();

    /* GCS FIX: Assume direction cosines orthogonal */
    xf->get_bsp()->SetGridDirection (bsp_direction);

    /* SetGridRegion automatically initializes internal coefficients to zero */
}

/* Initialize using image spacing */
static void
bsp_grid_from_img_grid (
    BsplineTransformType::OriginType& bsp_origin,    /* Output */
    BsplineTransformType::SpacingType& bsp_spacing,  /* Output */
    BsplineTransformType::RegionType& bsp_region,    /* Output */
    const Plm_image_header* pih,		     /* Input */
    float* grid_spac)				     /* Input */
{
    BsplineTransformType::RegionType::SizeType bsp_size;

    /* Convert image specifications to grid specifications */
    for (int d=0; d<3; d++) {
	float img_ext = (pih->Size(d) - 1) * pih->m_spacing[d];
#if defined (commentout)
	printf ("img_ext[%d] %g <- (%d - 1) * %g\n", 
	    d, img_ext, pih->Size(d), pih->m_spacing[d]);
#endif
	bsp_origin[d] = pih->m_origin[d] - grid_spac[d];
	bsp_spacing[d] = grid_spac[d];
	bsp_size[d] = 4 + (int) floor (img_ext / grid_spac[d]);
    }
    bsp_region.SetSize (bsp_size);
}

/* Initialize using image spacing */
static void
itk_bsp_set_grid_img (
    Xform *xf,
    const Plm_image_header* pih,
    float* grid_spac)
{
    BsplineTransformType::OriginType bsp_origin;
    BsplineTransformType::SpacingType bsp_spacing;
    BsplineTransformType::RegionType bsp_region;
    BsplineTransformType::DirectionType bsp_direction;

    /* Compute bspline grid specifications */
    bsp_direction = pih->m_direction;
    bsp_grid_from_img_grid (bsp_origin, bsp_spacing, bsp_region, 
	pih, grid_spac);

    /* Set grid specifications into xf structure */
    itk_bsp_set_grid (xf, bsp_origin, bsp_spacing, bsp_region, bsp_direction);
}

static void
xform_trn_to_itk_bsp_bulk (Xform *xf_out, Xform* xf_in,
			    const Plm_image_header* pih,
			    float* grid_spac)
{
    init_itk_bsp_default (xf_out);
    itk_bsp_set_grid_img (xf_out, pih, grid_spac);
    xf_out->get_bsp()->SetBulkTransform (xf_in->get_trn());
}

static void
xform_vrs_to_itk_bsp_bulk (Xform *xf_out, Xform* xf_in,
			    const Plm_image_header* pih,
			    float* grid_spac)
{
    init_itk_bsp_default (xf_out);
    itk_bsp_set_grid_img (xf_out, pih, grid_spac);
    xf_out->get_bsp()->SetBulkTransform (xf_in->get_vrs());
}
static void
xform_quat_to_itk_bsp_bulk (Xform *xf_out, Xform* xf_in,
			    const Plm_image_header* pih,
			    float* grid_spac)
{
    init_itk_bsp_default (xf_out);
    itk_bsp_set_grid_img (xf_out, pih, grid_spac);
    xf_out->get_bsp()->SetBulkTransform (xf_in->get_quat());
}
static void
xform_aff_to_itk_bsp_bulk (Xform *xf_out, Xform* xf_in,
			    const Plm_image_header* pih,
			    float* grid_spac)
{
    init_itk_bsp_default (xf_out);
    itk_bsp_set_grid_img (xf_out, pih, grid_spac);
    xf_out->get_bsp()->SetBulkTransform (xf_in->get_aff());
}

/* Convert xf to vector field to bspline */
static void
xform_any_to_itk_bsp_nobulk (
    Xform *xf_out, 
    Xform* xf_in,
    const Plm_image_header* pih,
    float* grid_spac)
{
    int d;
    Xform xf_tmp;
    Plm_image_header pih_bsp;

    /* Set bsp grid parameters in xf_out */
    init_itk_bsp_default (xf_out);
    itk_bsp_set_grid_img (xf_out, pih, grid_spac);
    BsplineTransformType::Pointer bsp_out = xf_out->get_bsp();

    /* Create temporary array for output coefficients */
    const unsigned int num_parms = bsp_out->GetNumberOfParameters();
    BsplineTransformType::ParametersType bsp_coeff;
    bsp_coeff.SetSize (num_parms);

    /* Compute bspline grid specifications */
    BsplineTransformType::OriginType bsp_origin;
    BsplineTransformType::SpacingType bsp_spacing;
    BsplineTransformType::RegionType bsp_region;
    BsplineTransformType::DirectionType bsp_direction;
    bsp_grid_from_img_grid (bsp_origin, bsp_spacing, 
	bsp_region, pih, grid_spac);

    /* GCS FIX: above should set m_direction */
    bsp_direction[0][0] = bsp_direction[1][1] = bsp_direction[2][2] = 1.0;

    /* Make a vector field at bspline grid spacing */
    pih_bsp.m_origin = bsp_origin;
    pih_bsp.m_spacing = bsp_spacing;
    pih_bsp.m_region = bsp_region;
    pih_bsp.m_direction = bsp_direction;
    xform_to_itk_vf (&xf_tmp, xf_in, &pih_bsp);

    /* Vector field is interleaved.  We need planar for decomposition. */
    FloatImageType::Pointer img = FloatImageType::New();
    img->SetOrigin (pih_bsp.m_origin);
    img->SetSpacing (pih_bsp.m_spacing);
    img->SetRegions (pih_bsp.m_region);
    img->SetDirection (pih_bsp.m_direction);
    img->Allocate ();

    /* Loop through planes */
    unsigned int counter = 0;
    for (d = 0; d < 3; d++) {
	/* Copy a single VF plane into img */
	typedef itk::ImageRegionIterator< FloatImageType > FloatIteratorType;
	typedef itk::ImageRegionIterator< DeformationFieldType > VFIteratorType;
	FloatIteratorType img_it (img, pih_bsp.m_region);
	VFIteratorType vf_it (xf_tmp.get_itk_vf(), pih_bsp.m_region);
	for (img_it.GoToBegin(), vf_it.GoToBegin(); 
	     !img_it.IsAtEnd(); 
	     ++img_it, ++vf_it) 
	{
	    img_it.Set(vf_it.Get()[d]);
	}

	/* Decompose into bpline coefficient image */
	typedef itk::BSplineDecompositionImageFilter <FloatImageType, 
	    DoubleImageType> DecompositionType;
	DecompositionType::Pointer decomposition = DecompositionType::New();
	decomposition->SetSplineOrder (SplineOrder);
	decomposition->SetInput (img);
	decomposition->Update ();

	/* Copy the coefficients into a temporary parameter array */
	typedef BsplineTransformType::ImageType ParametersImageType;
	ParametersImageType::Pointer newCoefficients 
	    = decomposition->GetOutput();
	typedef itk::ImageRegionIterator<ParametersImageType> Iterator;
	Iterator co_it (newCoefficients, bsp_out->GetGridRegion());
	co_it.GoToBegin();
	while (!co_it.IsAtEnd()) {
	    bsp_coeff[counter++] = co_it.Get();
	    ++co_it;
	}
    }

    /* Finally fixate coefficients into recently created bsp structure */
    bsp_out->SetParametersByValue (bsp_coeff);
}

/* GCS Jun 3, 2008.  When going from a lower image resolution to a 
    higher image resolution, the origin pixel moves outside of the 
    valid region defined by the bspline grid.  To fix this, we re-form 
    the b-spline with extra coefficients such that all pixels fall 
    within the valid region. */
void
itk_bsp_extend_to_region (Xform* xf,		    
	const Plm_image_header* pih,
	const ImageRegionType* roi)
{
    int d, old_idx;
    unsigned long i, j, k;
    int extend_needed = 0;
    BsplineTransformType::Pointer bsp = xf->get_bsp();
    BsplineTransformType::OriginType bsp_origin = bsp->GetGridOrigin();
    BsplineTransformType::RegionType bsp_region = bsp->GetGridRegion();
    BsplineTransformType::RegionType::SizeType bsp_size = bsp->GetGridRegion().GetSize();
    int eb[3], ea[3];  /* # of control points to "extend before" and "extend after" existing grid */

    /* Figure out if we need to extend the bspline grid.  If so, compute ea & eb, 
       as well as new values of bsp_region and bsp_origin. */
    for (d = 0; d < 3; d++) {
	float old_roi_origin = bsp->GetGridOrigin()[d] + bsp->GetGridSpacing()[d];
	float old_roi_corner = old_roi_origin + (bsp->GetGridRegion().GetSize()[d] - 3) * bsp->GetGridSpacing()[d];
	float new_roi_origin = pih->m_origin[d] + roi->GetIndex()[d] * pih->m_spacing[d];
	float new_roi_corner = new_roi_origin + (roi->GetSize()[d] - 1) * pih->m_spacing[d];
#if defined (commentout)
	printf ("extend? [%d]: (%g,%g) -> (%g,%g)\n", d,
		old_roi_origin, old_roi_corner, new_roi_origin, new_roi_corner);
	printf ("img_siz = %d, img_spac = %g\n", img_region.GetSize()[d], img_spacing[d]);
#endif
	ea[d] = eb[d] = 0;
	if (old_roi_origin > new_roi_origin) {
	    float diff = old_roi_origin - new_roi_origin;
	    eb[d] = (int) ceil (diff / bsp->GetGridSpacing()[d]);
	    bsp_origin[d] -= eb[d] * bsp->GetGridSpacing()[d];
	    bsp_size[d] += eb[d];
	    extend_needed = 1;
	}
	if (old_roi_corner < new_roi_corner) {
	    float diff = new_roi_origin - old_roi_origin;
	    ea[d] = (int) ceil (diff / bsp->GetGridSpacing()[d]);
	    bsp_size[d] += ea[d];
	    extend_needed = 1;
	}
    }
#if defined (commentout)
    printf ("ea = %d %d %d, eb = %d %d %d\n", ea[0], ea[1], ea[2], eb[0], eb[1], eb[2]);
#endif
    if (extend_needed) {

	/* Allocate new parameter array */
	BsplineTransformType::Pointer bsp_new = BsplineTransformType::New();
	BsplineTransformType::RegionType old_region = bsp->GetGridRegion();
	bsp_region.SetSize (bsp_size);
	bsp_new->SetGridOrigin (bsp_origin);
	bsp_new->SetGridRegion (bsp_region);
	bsp_new->SetGridSpacing (bsp->GetGridSpacing());

#if defined (commentout)
	std::cout << "OLD BSpline Region = "
		  << bsp->GetGridRegion();
	std::cout << "NEW BSpline Region = "
		  << bsp_new->GetGridRegion();
	std::cout << "OLD BSpline Grid Origin = "
		  << bsp->GetGridOrigin()
		  << std::endl;
	std::cout << "NEW BSpline Grid Origin = "
		  << bsp_new->GetGridOrigin()
		  << std::endl;
	std::cout << "BSpline Grid Spacing = "
		  << bsp_new->GetGridSpacing()
		  << std::endl;
#endif

	/* Copy current parameters in... */
	const unsigned int num_parms = bsp_new->GetNumberOfParameters();
	BsplineTransformType::ParametersType bsp_coeff;
	bsp_coeff.SetSize (num_parms);
	for (old_idx = 0, d = 0; d < 3; d++) {
	    for (k = 0; k < old_region.GetSize()[2]; k++) {
		for (j = 0; j < old_region.GetSize()[1]; j++) {
		    for (i = 0; i < old_region.GetSize()[0]; i++, old_idx++) {
			int new_idx;
			new_idx = ((((d * bsp_size[2]) + k + eb[2]) * bsp_size[1] + (j + eb[1])) * bsp_size[0]) + (i + eb[0]);
			/* GCS FIX - use GetParameters() */
//			printf ("[%4d <- %4d] %g\n", new_idx, old_idx, bsp_parms_old[old_idx]);
			bsp_coeff[new_idx] = bsp->GetParameters()[old_idx];
		    }
		}
	    }
	}

	/* Copy coefficients into recently created bsp struct */
	bsp_new->SetParametersByValue (bsp_coeff);

	/* Fixate xf with new bsp */
	xf->set_itk_bsp (bsp_new);
    }
}

static void
xform_itk_bsp_to_itk_bsp (Xform *xf_out, Xform* xf_in,
		      const Plm_image_header* pih,
		      float* grid_spac)
{
    BsplineTransformType::Pointer bsp_old = xf_in->get_bsp();

    init_itk_bsp_default (xf_out);
    itk_bsp_set_grid_img (xf_out, pih, grid_spac);

    /* Need to copy the bulk transform */
    BsplineTransformType::Pointer bsp_out = xf_out->get_bsp();
    bsp_out->SetBulkTransform (bsp_old->GetBulkTransform());

    /* Create temporary array for output coefficients */
    const unsigned int num_parms = xf_out->get_bsp()->GetNumberOfParameters();
    BsplineTransformType::ParametersType bsp_coeff;
    bsp_coeff.SetSize (num_parms);

    /* GCS May 12, 2008.  I feel like the below algorithm suggested 
	by ITK is wrong.  If BSplineResampleImageFunction interpolates the 
	coefficient image, the resulting resampled B-spline will be smoother 
	than the original.  But maybe the algorithm is OK, just a problem with 
	the lack of proper ITK documentation.  Need to test this.

       What this code does is this:
         1) Resample coefficient image using B-Spline interpolator
	 2) Pass resampled image to decomposition filter
	 3) Copy decomposition filter output into new coefficient image

       This is the original comment from the ITK code:
	 Now we need to initialize the BSpline coefficients of the higher 
	 resolution transform. This is done by first computing the actual 
	 deformation field at the higher resolution from the lower 
	 resolution BSpline coefficients. Then a BSpline decomposition 
	 is done to obtain the BSpline coefficient of the higher 
	 resolution transform. */
    unsigned int counter = 0;
    for (unsigned int k = 0; k < Dimension; k++) {
	typedef BsplineTransformType::ImageType ParametersImageType;
	typedef itk::ResampleImageFilter<ParametersImageType, ParametersImageType> ResamplerType;
	ResamplerType::Pointer resampler = ResamplerType::New();

	typedef itk::BSplineResampleImageFunction<ParametersImageType, double> FunctionType;
	FunctionType::Pointer fptr = FunctionType::New();

	typedef itk::IdentityTransform<double, Dimension> IdentityTransformType;
	IdentityTransformType::Pointer identity = IdentityTransformType::New();

	resampler->SetInput (bsp_old->GetCoefficientImage()[k]);
	resampler->SetInterpolator (fptr);
	resampler->SetTransform (identity);
	resampler->SetSize (bsp_out->GetGridRegion().GetSize());
	resampler->SetOutputSpacing (bsp_out->GetGridSpacing());
	resampler->SetOutputOrigin (bsp_out->GetGridOrigin());
	resampler->SetOutputDirection (bsp_out->GetGridDirection());

	typedef itk::BSplineDecompositionImageFilter<ParametersImageType, ParametersImageType> DecompositionType;
	DecompositionType::Pointer decomposition = DecompositionType::New();

	decomposition->SetSplineOrder (SplineOrder);
	decomposition->SetInput (resampler->GetOutput());
	decomposition->Update();

	ParametersImageType::Pointer newCoefficients = decomposition->GetOutput();

	// copy the coefficients into a temporary parameter array
	typedef itk::ImageRegionIterator<ParametersImageType> Iterator;
	Iterator it (newCoefficients, bsp_out->GetGridRegion());
	while (!it.IsAtEnd()) {
	    bsp_coeff[counter++] = it.Get();
	    ++it;
	}
    }

    /* Finally fixate coefficients into recently created bsp structure */
    bsp_out->SetParametersByValue (bsp_coeff);
}

/* Compute itk_bsp grid specifications from gpuit specifications */
static void
gpuit_bsp_grid_to_itk_bsp_grid (
	BsplineTransformType::OriginType& bsp_origin,	    /* Output */
	BsplineTransformType::SpacingType& bsp_spacing,	    /* Output */
	BsplineTransformType::RegionType& bsp_region,  /* Output */
	Bspline_xform* bxf)			    /* Input */
{
    BsplineTransformType::SizeType bsp_size;

    for (int d = 0; d < 3; d++) {
	bsp_size[d] = bxf->cdims[d];
	bsp_origin[d] = bxf->img_origin[d] + bxf->roi_offset[d] * bxf->img_spacing[d] - bxf->grid_spac[d];
	bsp_spacing[d] = bxf->grid_spac[d];
    }
    bsp_region.SetSize (bsp_size);
}

static void
gpuit_bsp_to_itk_bsp_raw (Xform *xf_out, Xform* xf_in, 
			  const Plm_image_header* pih)
{
    typedef BsplineTransformType::ImageType ParametersImageType;
    typedef itk::ImageRegionIterator<ParametersImageType> Iterator;
    Bspline_xform* bxf = xf_in->get_gpuit_bsp();

    BsplineTransformType::OriginType bsp_origin;
    BsplineTransformType::SpacingType bsp_spacing;
    BsplineTransformType::RegionType bsp_region;
    BsplineTransformType::DirectionType bsp_direction = pih->m_direction;

    /* Convert bspline grid geometry from gpuit to itk */
    gpuit_bsp_grid_to_itk_bsp_grid (bsp_origin, bsp_spacing, bsp_region, bxf);

    /* Create itk bspline structure */
    init_itk_bsp_default (xf_out);
    itk_bsp_set_grid (xf_out, bsp_origin, bsp_spacing, bsp_region, bsp_direction);

    /* RMK: bulk transform is Identity (not supported by GPUIT) */

    /* Create temporary array for output coefficients */
    const unsigned int num_parms = xf_out->get_bsp()->GetNumberOfParameters();
    BsplineTransformType::ParametersType bsp_coeff;
    bsp_coeff.SetSize (num_parms);

    /* Copy from GPUIT coefficient array to ITK coefficient array */
    int k = 0;
    for (int d = 0; d < 3; d++) {
	for (int i = 0; i < bxf->num_knots; i++) {
	    bsp_coeff[k] = bxf->coeff[3*i+d];
	    k++;
	}
    }

    /* Fixate coefficients into bsp structure */
    xf_out->get_bsp()->SetParametersByValue (bsp_coeff);
}

/* If grid_spac is null, then don't resample */
void
xform_gpuit_bsp_to_itk_bsp (Xform *xf_out, Xform* xf_in,
		      const Plm_image_header* pih,
		      const ImageRegionType* roi, /* Not yet used */
		      float* grid_spac)
{
    typedef BsplineTransformType::ImageType ParametersImageType;
    typedef itk::ImageRegionIterator<ParametersImageType> Iterator;
//    Bspline_xform* bxf = xf_in->get_gpuit_bsp();
    Xform xf_tmp;
    OriginType img_origin_old;
    SpacingType img_spacing_old;
    ImageRegionType img_region_old;

    printf ("Input pih\n");
    printf ("Origin\n");
    std::cout << pih->m_origin << std::endl;
    printf ("Spacing\n");
    std::cout << pih->m_spacing << std::endl;
    printf ("Region\n");
    std::cout << pih->m_region;
//    printf ("grid_spac = %g %g %g\n", grid_spac[0], grid_spac[1], grid_spac[2]);

    if (grid_spac) {
	/* Convert to itk data structure */
	printf ("Running: gpuit_bsp_to_itk_bsp_raw\n");
	gpuit_bsp_to_itk_bsp_raw (&xf_tmp, xf_in, pih);

	/* Then, resample the xform to the desired grid spacing */
	printf ("Running: xform_itk_bsp_to_itk_bsp\n");
	xform_itk_bsp_to_itk_bsp (xf_out, &xf_tmp, pih, grid_spac);
    } else {
	/* Convert to itk data structure only */
	printf ("Running: gpuit_bsp_to_itk_bsp_raw\n");
	gpuit_bsp_to_itk_bsp_raw (xf_out, xf_in, pih);
    }

    printf ("Completed: xform_itk_bsp_to_itk_bsp\n");
}

/* -----------------------------------------------------------------------
   Conversion to itk_vf
   ----------------------------------------------------------------------- */
static DeformationFieldType::Pointer
xform_itk_any_to_itk_vf (itk::Transform<double,3,3>* xf,
		     const Plm_image_header* pih)
{
    DeformationFieldType::Pointer itk_vf = DeformationFieldType::New();

    itk_vf->SetOrigin (pih->m_origin);
    itk_vf->SetSpacing (pih->m_spacing);
    itk_vf->SetRegions (pih->m_region);
    //std::cout << pih->m_direction;
    itk_vf->SetDirection (pih->m_direction);
    itk_vf->Allocate ();

    typedef itk::ImageRegionIteratorWithIndex< DeformationFieldType > FieldIterator;
    FieldIterator fi (itk_vf, itk_vf->GetLargestPossibleRegion());

    fi.GoToBegin();

    DoublePointType fixed_point;
    DoublePointType moving_point;
    DeformationFieldType::IndexType index;

    FloatVectorType displacement;

    while (!fi.IsAtEnd()) {
	index = fi.GetIndex();
	itk_vf->TransformIndexToPhysicalPoint (index, fixed_point);
	moving_point = xf->TransformPoint (fixed_point);
	for (int r = 0; r < 3; r++) {
	    displacement[r] = moving_point[r] - fixed_point[r];
	}
	fi.Set (displacement);
	++fi;
    }
    return itk_vf;
}

/* ITK bsp is different from itk_any, because additional control points might be 
    needed */
static DeformationFieldType::Pointer 
xform_itk_bsp_to_itk_vf (Xform* xf_in, const Plm_image_header* pih)
{
    int d;
    Xform xf_tmp;
    float grid_spac[3];

    /* Deep copy of itk_bsp */
    for (d = 0; d < 3; d++) {
	grid_spac[d] = xf_in->get_bsp()->GetGridSpacing()[d];
    }
    xform_itk_bsp_to_itk_bsp (&xf_tmp, xf_in, pih, grid_spac);

    /* Extend bsp control point grid */
    itk_bsp_extend_to_region (&xf_tmp, pih, &pih->m_region);

    /* Convert extended bsp to vf */
    return xform_itk_any_to_itk_vf (xf_tmp.get_bsp(), pih);
}

static DeformationFieldType::Pointer 
xform_itk_vf_to_itk_vf (DeformationFieldType::Pointer vf, Plm_image_header* pih)
{
    vf = vector_resample_image (vf, pih);
    return vf;
}

/* Here what we're going to do is use GPUIT library to interpolate the 
    B-Spline at its native resolution, then convert gpuit_vf -> itk_vf. 

    GCS: Aug 6, 2008.  The above idea doesn't work, because the native 
    resolution might not encompass the image.  Here is what we will do:
    1) Convert to ITK B-Spline
    2) Extend ITK B-Spline to encompass image
    3) Render vf.
    */
static DeformationFieldType::Pointer
xform_gpuit_bsp_to_itk_vf (Xform* xf_in, Plm_image_header* pih)
{
    DeformationFieldType::Pointer itk_vf;

    Xform xf_tmp;
    OriginType img_origin;
    SpacingType img_spacing;
    ImageRegionType img_region;

    /* Copy from GPUIT coefficient array to ITK coefficient array */
    printf ("gpuit_bsp_to_itk_bsp_raw\n");
    gpuit_bsp_to_itk_bsp_raw (&xf_tmp, xf_in, pih);

    /* Resize itk array to span image */
    printf ("itk_bsp_extend_to_region\n");
    itk_bsp_extend_to_region (&xf_tmp, pih, &pih->m_region);

    /* Render to vector field */
    printf ("xform_itk_any_to_itk_vf\n");
    itk_vf = xform_itk_any_to_itk_vf (xf_tmp.get_bsp(), pih);

    return itk_vf;
}

/* 1) convert gpuit -> itk at the native resolution, 
   2) convert itk -> itk to change resolution.  */
DeformationFieldType::Pointer 
xform_gpuit_vf_to_itk_vf (
    Volume* vf,            /* Input */
    Plm_image_header* pih    /* Input, can be null */
)
{
    int i;
    DeformationFieldType::SizeType sz;
    DeformationFieldType::IndexType st;
    DeformationFieldType::RegionType rg;
    DeformationFieldType::PointType og;
    DeformationFieldType::SpacingType sp;
    DeformationFieldType::Pointer itk_vf = DeformationFieldType::New();
    FloatVectorType displacement;

    /* Copy header & allocate data for itk */
    for (i = 0; i < 3; i++) {
	st[i] = 0;
	sz[i] = vf->dim[i];
	sp[i] = vf->pix_spacing[i];
	og[i] = vf->offset[i];
    }
    rg.SetSize (sz);
    rg.SetIndex (st);

    itk_vf->SetRegions (rg);
    itk_vf->SetOrigin (og);
    itk_vf->SetSpacing (sp);
    itk_vf->Allocate();

    /* Copy data into itk */
    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator fi (itk_vf, itk_vf->GetLargestPossibleRegion());

    if (vf->pix_type == PT_VF_FLOAT_INTERLEAVED) {
	float* img = (float*) vf->img;
	int i = 0;
	for (fi.GoToBegin(); !fi.IsAtEnd(); ++fi) {
	    for (int r = 0; r < 3; r++) {
		displacement[r] = img[i++];
	    }
	    fi.Set (displacement);
	}
    }
    else if (vf->pix_type == PT_VF_FLOAT_PLANAR) {
	float** img = (float**) vf->img;
	int i = 0;
	for (fi.GoToBegin(); !fi.IsAtEnd(); ++fi, ++i) {
	    for (int r = 0; r < 3; r++) {
		displacement[r] = img[r][i];
	    }
	    fi.Set (displacement);
	}
    } else {
	print_and_exit ("Irregular pix_type used converting gpuit_xf -> itk\n");
    }

    /* Resample to requested resolution */
    if (pih) {
	itk_vf = xform_itk_vf_to_itk_vf (itk_vf, pih);
    }

    return itk_vf;
}

/* -----------------------------------------------------------------------
   Conversion to gpuit_bsp
   ----------------------------------------------------------------------- */
static Bspline_xform*
create_gpuit_bxf (Plm_image_header* pih, float* grid_spac)
{
    int d;
    Bspline_xform* bxf = (Bspline_xform*) malloc (sizeof(Bspline_xform));
    float img_origin[3];
    float img_spacing[3];
    int img_dim[3];
    int roi_offset[3];
    int roi_dim[3];
    int vox_per_rgn[3];

    pih->get_gpuit_origin (img_origin);
    pih->get_gpuit_dim (img_dim);
    pih->get_gpuit_spacing (img_spacing);

    for (d = 0; d < 3; d++) {
	/* Old ROI was whole image */
	roi_offset[d] = 0;
	roi_dim[d] = img_dim[d];
	/* Compute vox_per_rgn */
	vox_per_rgn[d] = ROUND_INT (grid_spac[d] / img_spacing[d]);
	if (vox_per_rgn[d] < 4) {
	    printf ("Warning: vox_per_rgn was less than 4.\n");
	    vox_per_rgn[d] = 4;
	}
    }
    bspline_xform_initialize (bxf, img_origin, img_spacing, img_dim, 
		roi_offset, roi_dim, vox_per_rgn);
    return bxf;
}

void
xform_any_to_gpuit_bsp (Xform* xf_out, Xform* xf_in, Plm_image_header* pih, 
    float* grid_spac)
{
    Xform xf_tmp;
    ImageRegionType roi;

    /* Initialize gpuit bspline data structure */
    Bspline_xform* bxf_new = create_gpuit_bxf (pih, grid_spac);

    if (xf_in->m_type != XFORM_NONE) {
	/* Output ROI is going to be whole image */
	roi = pih->m_region;

	/* Create itk_bsp xf using image specifications */
	xform_any_to_itk_bsp_nobulk (&xf_tmp, xf_in, pih, bxf_new->grid_spac);

	/* Copy from ITK coefficient array to gpuit coefficient array */
	int k = 0;
	for (int d = 0; d < 3; d++) {
	    for (int i = 0; i < bxf_new->num_knots; i++) {
		bxf_new->coeff[3*i+d] = xf_tmp.get_bsp()->GetParameters()[k];
		k++;
	    }
	}
    }

    /* Fixate gpuit bsp to xf */
    xf_out->set_gpuit_bsp (bxf_new);

    printf ("xform_any_to_gpuit_bsp complete\n");
}

void
xform_gpuit_bsp_to_gpuit_bsp (
    Xform* xf_out, 
    Xform* xf_in, 
    Plm_image_header* pih, 
    float* grid_spac
)
{
    Xform xf_tmp;
    ImageRegionType roi;

    /* Initialize gpuit bspline data structure */
    Bspline_xform* bxf_new = create_gpuit_bxf (pih, grid_spac);

    /* Output ROI is going to be whole image */
    roi = pih->m_region;

    /* Create itk_bsp xf using image specifications */
    xform_gpuit_bsp_to_itk_bsp (&xf_tmp, xf_in, pih, &roi, bxf_new->grid_spac);

    /* Copy from ITK coefficient array to gpuit coefficient array */
    int k = 0;
    for (int d = 0; d < 3; d++) {
	for (int i = 0; i < bxf_new->num_knots; i++) {
	    bxf_new->coeff[3*i+d] = xf_tmp.get_bsp()->GetParameters()[k];
	    k++;
	}
    }

    /* Fixate gpuit bsp to xf */
    xf_out->set_gpuit_bsp (bxf_new);
}


/* -----------------------------------------------------------------------
   Conversion to gpuit_vf
   ----------------------------------------------------------------------- */
Volume*
xform_gpuit_vf_to_gpuit_vf (Volume* vf_in, int* dim, float* offset, float* pix_spacing)
{
    Volume* vf_out;
    vf_out = volume_resample (vf_in, dim, offset, pix_spacing);
    return vf_out;
}

Volume*
xform_gpuit_bsp_to_gpuit_vf (Xform* xf_in, int* dim, float* offset, float* pix_spacing)
{
    Bspline_xform* bxf = xf_in->get_gpuit_bsp();
    Volume* vf_out;

    /* GCS FIX: Need direction cosines */
    vf_out = volume_create (dim, offset, pix_spacing, PT_VF_FLOAT_INTERLEAVED, 0, 0);
    bspline_interpolate_vf (vf_out, bxf);
    return vf_out;
}

Volume*
xform_itk_vf_to_gpuit_vf (DeformationFieldType::Pointer itk_vf, int* dim, float* offset, float* pix_spacing)
{
    /* GCS FIX: Need direction cosines */
    Volume* vf_out = volume_create (dim, offset, pix_spacing, PT_VF_FLOAT_INTERLEAVED, 0, 0);
    float* img = (float*) vf_out->img;
    FloatVectorType displacement;

    int i = 0;
    typedef itk::ImageRegionIterator< DeformationFieldType > FieldIterator;
    FieldIterator fi (itk_vf, itk_vf->GetLargestPossibleRegion());
    for (fi.GoToBegin(); !fi.IsAtEnd(); ++fi) {
	displacement = fi.Get ();
	for (int r = 0; r < 3; r++) {
	    img[i++] = displacement[r];
	}
    }
    return vf_out;
}


/* -----------------------------------------------------------------------
   Selection routines to convert from X to Y
   ----------------------------------------------------------------------- */
void
xform_to_trn (Xform *xf_out, 
	      Xform *xf_in, 
	      Plm_image_header *pih)
{
    switch (xf_in->m_type) {
    case XFORM_NONE:
	init_translation_default (xf_out);
	break;
    case XFORM_ITK_TRANSLATION:
	*xf_out = *xf_in;
	break;
    case XFORM_ITK_VERSOR:
    case XFORM_ITK_QUATERNION:
    case XFORM_ITK_AFFINE:
    case XFORM_ITK_BSPLINE:
    case XFORM_ITK_TPS:
    case XFORM_ITK_VECTOR_FIELD:
	print_and_exit ("Sorry, couldn't convert to trn\n");
	break;
    case XFORM_GPUIT_BSPLINE:
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, gpuit xforms not fully implemented\n");
	break;
    default:
	print_and_exit ("Program error.  Bad xform type.\n");
	break;
    }
}

void
xform_to_vrs (Xform *xf_out, 
	      Xform *xf_in, 
	      Plm_image_header *pih)
{
    switch (xf_in->m_type) {
    case XFORM_NONE:
	init_versor_default (xf_out);
	break;
    case XFORM_ITK_TRANSLATION:
	xform_trn_to_vrs (xf_out, xf_in);
	break;
    case XFORM_ITK_VERSOR:
	*xf_out = *xf_in;
	break;
    case XFORM_ITK_AFFINE:
    case XFORM_ITK_QUATERNION:
    case XFORM_ITK_BSPLINE:
    case XFORM_ITK_TPS:
    case XFORM_ITK_VECTOR_FIELD:
	print_and_exit ("Sorry, couldn't convert to vrs\n");
	break;
    case XFORM_GPUIT_BSPLINE:
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, gpuit xforms not fully implemented\n");
	break;
    default:
	print_and_exit ("Program error.  Bad xform type.\n");
	break;
    }
}

void
xform_to_quat (Xform *xf_out, 
    Xform *xf_in, 
    Plm_image_header *pih)
{
    switch (xf_in->m_type) {
    case XFORM_NONE:
	init_quaternion_default (xf_out);
	break;
    case XFORM_ITK_TRANSLATION:
	print_and_exit ("Sorry, couldn't convert to quaternion\n");
	break;
    case XFORM_ITK_VERSOR:
	xform_vrs_to_quat (xf_out, xf_in);
	break;
    case XFORM_ITK_QUATERNION:
	*xf_out = *xf_in;
	break;
    case XFORM_ITK_AFFINE:
    case XFORM_ITK_BSPLINE:
    case XFORM_ITK_TPS:
    case XFORM_ITK_VECTOR_FIELD:
    case XFORM_GPUIT_BSPLINE:
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, couldn't convert to quaternion\n");
	break;
    default:
	print_and_exit ("Program error.  Bad xform type.\n");
	break;
    }
}

void
xform_to_aff (Xform *xf_out, 
	      Xform *xf_in, 
	      Plm_image_header *pih)
{
    switch (xf_in->m_type) {
    case XFORM_NONE:
	init_affine_default (xf_out);
	break;
    case XFORM_ITK_TRANSLATION:
	xform_trn_to_aff (xf_out, xf_in);
	break;
    case XFORM_ITK_VERSOR:
	xform_vrs_to_aff (xf_out, xf_in);
	break;
    case XFORM_ITK_QUATERNION:
	print_and_exit ("Sorry, couldn't convert to aff\n");
	break;
    case XFORM_ITK_AFFINE:
	*xf_out = *xf_in;
	break;
    case XFORM_ITK_BSPLINE:
    case XFORM_ITK_TPS:
    case XFORM_ITK_VECTOR_FIELD:
	print_and_exit ("Sorry, couldn't convert to aff\n");
	break;
    case XFORM_GPUIT_BSPLINE:
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, gpuit xforms not fully implemented\n");
	break;
    default:
	print_and_exit ("Program error.  Bad xform type.\n");
	break;
    }
}

void
xform_to_itk_bsp (Xform *xf_out, 
		  Xform *xf_in, 
		  Plm_image_header* pih,
		  float* grid_spac)
{
    BsplineTransformType::Pointer bsp;

    switch (xf_in->m_type) {
    case XFORM_NONE:
	init_itk_bsp_default (xf_out);
	itk_bsp_set_grid_img (xf_out, pih, grid_spac);
	break;
    case XFORM_ITK_TRANSLATION:
	xform_trn_to_itk_bsp_bulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_VERSOR:
	xform_vrs_to_itk_bsp_bulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_QUATERNION:
	xform_quat_to_itk_bsp_bulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_AFFINE:
	xform_aff_to_itk_bsp_bulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_BSPLINE:
	xform_itk_bsp_to_itk_bsp (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_TPS:
	print_and_exit ("Sorry, couldn't convert itk_tps to itk_bsp\n");
	break;
    case XFORM_ITK_VECTOR_FIELD:
	print_and_exit ("Sorry, couldn't convert itk_vf to itk_bsp\n");
	break;
    case XFORM_GPUIT_BSPLINE:
	xform_gpuit_bsp_to_itk_bsp (xf_out, xf_in, pih, &pih->m_region, grid_spac);
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, couldn't convert gpuit_vf to itk_bsp\n");
	break;
    default:
	print_and_exit ("Program error.  Bad xform type.\n");
	break;
    }
}

void
xform_to_itk_bsp_nobulk (
    Xform *xf_out, 
    Xform *xf_in, 
    Plm_image_header* pih,
    float* grid_spac)
{
    switch (xf_in->m_type) {
    case XFORM_NONE:
	init_itk_bsp_default (xf_out);
	itk_bsp_set_grid_img (xf_out, pih, grid_spac);
	break;
    case XFORM_ITK_TRANSLATION:
	xform_any_to_itk_bsp_nobulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_VERSOR:
	xform_any_to_itk_bsp_nobulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_QUATERNION:
	xform_any_to_itk_bsp_nobulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_AFFINE:
	xform_any_to_itk_bsp_nobulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_BSPLINE:
	xform_any_to_itk_bsp_nobulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_TPS:
	xform_any_to_itk_bsp_nobulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_VECTOR_FIELD:
	xform_any_to_itk_bsp_nobulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_GPUIT_BSPLINE:
	xform_any_to_itk_bsp_nobulk (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	xform_any_to_itk_bsp_nobulk (xf_out, xf_in, pih, grid_spac);
	break;
    default:
	print_and_exit ("Program error.  Bad xform type.\n");
	break;
    }
}

void
xform_to_itk_vf (Xform* xf_out, Xform *xf_in, Plm_image_header* pih)
{
    DeformationFieldType::Pointer vf;

    switch (xf_in->m_type) {
    case XFORM_NONE:
	print_and_exit ("Sorry, couldn't convert to vf\n");
	break;
    case XFORM_ITK_TRANSLATION:
	vf = xform_itk_any_to_itk_vf (xf_in->get_trn(), pih);
	break;
    case XFORM_ITK_VERSOR:
	vf = xform_itk_any_to_itk_vf (xf_in->get_vrs(), pih);
	break;
    case XFORM_ITK_QUATERNION:
	vf = xform_itk_any_to_itk_vf (xf_in->get_quat(), pih);
	break;
    case XFORM_ITK_AFFINE:
	vf = xform_itk_any_to_itk_vf (xf_in->get_aff(), pih);
	break;
    case XFORM_ITK_BSPLINE:
	vf = xform_itk_bsp_to_itk_vf (xf_in, pih);
	break;
    case XFORM_ITK_TPS:
	vf = xform_itk_any_to_itk_vf (xf_in->get_itk_tps(), pih);
	break;
    case XFORM_ITK_VECTOR_FIELD:
	vf = xform_itk_vf_to_itk_vf (xf_in->get_itk_vf(), pih);
	break;
    case XFORM_GPUIT_BSPLINE:
	vf = xform_gpuit_bsp_to_itk_vf (xf_in, pih);
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	vf = xform_gpuit_vf_to_itk_vf (xf_in->get_gpuit_vf(), pih);
	break;
    default:
	print_and_exit ("Program error.  Bad xform type.\n");
	break;
    }
    xf_out->set_itk_vf (vf);
}

/* Overloaded fn.  Fix soon... */
void
xform_to_itk_vf (Xform* xf_out, Xform *xf_in, FloatImageType::Pointer image)
{
    Plm_image_header pih;
    pih.set_from_itk_image (image);
    xform_to_itk_vf (xf_out, xf_in, &pih);
}

void
xform_to_gpuit_bsp (Xform* xf_out, Xform* xf_in, Plm_image_header* pih, 
    float* grid_spac)
{
    switch (xf_in->m_type) {
    case XFORM_NONE:
	xform_any_to_gpuit_bsp (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_TRANSLATION:
	xform_any_to_gpuit_bsp (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_VERSOR:
	xform_any_to_gpuit_bsp (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_QUATERNION:
	xform_any_to_gpuit_bsp (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_AFFINE:
	xform_any_to_gpuit_bsp (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_BSPLINE:
	xform_any_to_gpuit_bsp (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_TPS:
	xform_any_to_gpuit_bsp (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_ITK_VECTOR_FIELD:
	xform_any_to_gpuit_bsp (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_GPUIT_BSPLINE:
	xform_gpuit_bsp_to_gpuit_bsp (xf_out, xf_in, pih, grid_spac);
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	print_and_exit ("Sorry, gpuit_vf to gpuit_bsp not implemented\n");
	break;
    default:
	print_and_exit ("Program error.  Bad xform type.\n");
	break;
    }
}

void
xform_to_gpuit_vf (Xform* xf_out, Xform *xf_in, int* dim, float* offset, float* pix_spacing)
{
    Volume* vf = 0;

    switch (xf_in->m_type) {
    case XFORM_NONE:
	print_and_exit ("Sorry, couldn't convert NONE to gpuit_vf\n");
	break;
    case XFORM_ITK_TRANSLATION:
	print_and_exit ("Sorry, itk_translation to gpuit_vf not implemented\n");
	break;
    case XFORM_ITK_VERSOR:
	print_and_exit ("Sorry, itk_versor to gpuit_vf not implemented\n");
	break;
    case XFORM_ITK_QUATERNION:
	print_and_exit ("Sorry, couldn't convert to gpuit vf\n");
	break;
    case XFORM_ITK_AFFINE:
	print_and_exit ("Sorry, itk_affine to gpuit_vf not implemented\n");
	break;
    case XFORM_ITK_BSPLINE:
	print_and_exit ("Sorry, itk_bspline to gpuit_vf not implemented\n");
	break;
    case XFORM_ITK_TPS:
	print_and_exit ("Sorry, itk_tps to gpuit_vf not implemented\n");
	break;
    case XFORM_ITK_VECTOR_FIELD:
	vf = xform_itk_vf_to_gpuit_vf (xf_in->get_itk_vf(), dim, offset, pix_spacing);
	break;
    case XFORM_GPUIT_BSPLINE:
	vf = xform_gpuit_bsp_to_gpuit_vf (xf_in, dim, offset, pix_spacing);
	break;
    case XFORM_GPUIT_VECTOR_FIELD:
	vf = xform_gpuit_vf_to_gpuit_vf (xf_in->get_gpuit_vf(), dim, offset, pix_spacing);
	break;
    default:
	print_and_exit ("Program error.  Bad xform type.\n");
	break;
    }

    xf_out->set_gpuit_vf (vf);
}
