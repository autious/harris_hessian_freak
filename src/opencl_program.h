#pragma once
#include <CL/opencl.h>

cl_program opencl_program_load( const char* name );
void opencl_program_close();
void opencl_program_add_define_integer( const char* name, int value );
void opencl_program_add_compiler_flag( const char* value );
