/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _regularization_parms_h_
#define _regularization_parms_h_

#include "plmregister_config.h"

enum Regularization_type {
    REGULARIZATION_NONE, 
    REGULARIZATION_BSPLINE_ANALYTIC, 
    REGULARIZATION_BSPLINE_SEMI_ANALYTIC, 
    REGULARIZATION_BSPLINE_NUMERIC
};

class Regularization_parms
{
public:
    Regularization_type regularization_type;
    mutable char implementation;
    float total_displacement_penalty;
    float diffusion_penalty;
    float curvature_penalty;
    float lame_coefficient_1;
    float lame_coefficient_2;
    float linear_elastic_multiplier;
    float third_order_penalty;
    float curvature_mixed_weight;
    
public:
    Regularization_parms () {
        this->regularization_type = REGULARIZATION_BSPLINE_ANALYTIC;
        this->implementation = '\0';
        this->total_displacement_penalty = 0.f;
        this->diffusion_penalty = 0.f;
        this->curvature_penalty = 0.f;
        this->lame_coefficient_1 = 0.f;
        this->lame_coefficient_2 = 0.f;
        this->linear_elastic_multiplier = 1.f;
        this->third_order_penalty = 0.f;
	this->curvature_mixed_weight = 1.f;
    }
};

#endif
