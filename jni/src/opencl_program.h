#pragma once
#include <CL/opencl.h>

void opencl_program_compile( const char** programs, const char *cfo );
cl_program opencl_program_load( const char* name );
void opencl_program_close();
