#pragma once

#include <CL/opencl.h>

struct StringObject
{
    char* str;
    int str_count;
    int str_size;
};

void string_object_init( struct StringObject* cfo );
void string_object_free( struct StringObject* cfo );

void string_object_append_string( struct StringObject *cfo, const char* value );

void string_object_add_define_integer( struct StringObject *cfo, const char* name, int value );
void string_object_add_compiler_flag( struct StringObject *cfo, const char* value );
