/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _rt_dose_timing_h_
#define _rt_dose_timing_h_

#include "plmdose_config.h"
#include "logfile.h"
#include "plm_timer.h"
#include "smart_pointer.h"

class Rt_dose_timing
{
public:
    SMART_POINTER_SUPPORT (Rt_dose_timing);
public:
    Plm_timer timer_sigma;
    Plm_timer timer_dose_calc;
    Plm_timer timer_reformat;
    Plm_timer timer_io;
    Plm_timer timer_misc;
public:
    void reset () {
        timer_sigma.reset ();
        timer_dose_calc.reset ();
        timer_reformat.reset ();
        timer_io.reset ();
        timer_misc.reset ();
    }
    void report () {
        lprintf ("Sigma:     %f seconds\n", timer_sigma.report());
        lprintf ("Calc:      %f seconds\n", timer_dose_calc.report());
        lprintf ("IO:        %f seconds\n", timer_io.report());
        lprintf ("Reformat:  %f seconds\n", timer_reformat.report());
        lprintf ("Misc:      %f seconds\n", timer_misc.report());
    };
};

#endif
