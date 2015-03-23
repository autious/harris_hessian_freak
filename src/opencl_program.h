#pragma once
#include <CL/opencl.h>

void opencl_program_compile( const char** programs );
cl_program opencl_program_load( const char* name );
void opencl_program_close();
void opencl_program_add_define_integer( const char* name, int value );
void opencl_program_add_compiler_flag( const char* value );
