#pragma once

#include <CL/opencl.h>

cl_float* generate_gauss_kernel_line( size_t* size, float sigma );
cl_float* generate_gauss_kernel_2D( size_t* diameter, float sigma );
