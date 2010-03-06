/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rtds_h_
#define _rtds_h_

#include "plm_config.h"
#include "cxt_io.h"
#include "plm_image.h"

/* rtds = RT data set */
class Rtds {
public:
    PlmImage *m_img;                     /* CT image */
    Cxt_structure_list *m_cxt;           /* Structure set in polyline form */
    PlmImage *m_ss_img;                  /* Structure set in bitmap form */
    Cxt_structure_list *m_ss_list;       /* Names of structures/bitmap form */
    PlmImage *m_dose;                    /* RT dose */
public:
    Rtds () {
	m_img = 0;
	m_ss_img = 0;
	m_ss_list = 0;
	m_cxt = 0;
	m_dose = 0;
    }
    ~Rtds () {
	if (m_img) {
	    delete m_img;
	}
	if (m_cxt) {
	    cxt_destroy (m_cxt);
	}
	if (m_ss_img) {
	    delete m_ss_img;
	}
	if (m_ss_list) {
	    cxt_destroy (m_ss_list);
	}
	if (m_dose) {
	    delete m_dose;
	}
    }
    void load_xio (char *xio_dir);
    void load_ss_img (char *ss_img, char *ss_list);
    void convert_ss_img_to_cxt (void);
};

#if defined __cplusplus
extern "C" {
#endif


#if defined __cplusplus
}
#endif

#endif
