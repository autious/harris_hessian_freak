#pragma once

#include <CL/opencl.h>

struct CompileFlagObject
{
    char* compile_macro;
    int compile_macro_count;
    int compile_macro_size;
};

void compile_flag_object_init( struct CompileFlagObject* cfo );
void compile_flag_object_free( struct CompileFlagObject* cfo );

void compile_flag_object_add_define_integer( struct CompileFlagObject *cfo, const char* name, int value );
void compile_flag_object_add_compiler_flag( struct CompileFlagObject *cfo, const char* value );
