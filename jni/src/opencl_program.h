#pragma once
#include <CL/opencl.h>
#include "compile_flag_object.h"

void opencl_program_compile( const char** programs, struct CompileFlagObject *cfo );
cl_program opencl_program_load( const char* name );
void opencl_program_close();
