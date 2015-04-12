#pragma once

#include <CL/opencl.h>
#include <stdbool.h>

//Set to run kernel variants that are reference implementations.
extern bool opencl_run_reference_mode;

#ifdef HH_USE_HALF
typedef cl_half hh_float;
#else
typedef cl_float hh_float;
#endif

typedef cl_float param_float;
