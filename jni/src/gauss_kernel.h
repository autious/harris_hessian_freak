#pragma once

#include <CL/opencl.h>
#include "opencl_config.h"

hh_float* generate_gauss_kernel_line( size_t* size, float sigma );
//hh_float* generate_gauss_kernel_2D( size_t* diameter, float sigma );
