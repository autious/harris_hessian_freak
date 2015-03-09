#pragma once
#include <CL/opencl.h>

cl_program opencl_program_load( const char* name );
void opencl_program_close();
