#pragma once

#include <stdbool.h>
#include <CL/opencl.h>

bool opencl_loader_init();
void opencl_loader_close();
void opencl_loader_get_error( char* value );
size_t opencl_loader_get_error_size(); 
cl_program opencl_loader_load_program( const char* name );

cl_context opencl_loader_get_context();
cl_command_queue opencl_loader_get_command_queue();
