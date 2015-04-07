#include "compile_flag_object.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "util.h"
#include "log.h"

static void resize( struct CompileFlagObject *cfo, size_t additional_len )
{
    if( cfo->compile_macro_count + additional_len + 1 > cfo->compile_macro_size )
    {
        cfo->compile_macro_size = cfo->compile_macro_count + additional_len + 1;
        cfo->compile_macro = realloc( cfo->compile_macro, sizeof( char ) * cfo->compile_macro_size );
    }
}

void compile_flag_object_init( struct CompileFlagObject* cfo )
{
    cfo->compile_macro = NULL;
    cfo->compile_macro_count = 0;
    cfo->compile_macro_size = 0;
}

void compile_flag_object_free( struct CompileFlagObject *cfo )
{
    free( cfo->compile_macro );
    cfo->compile_macro = NULL;
}

void compile_flag_object_add_define_integer( struct CompileFlagObject* cfo, const char* name, int value )
{
    size_t additional_len = strlen( " -D " ) + strlen( name ) + 1 + count_base_10_digits( value );

    resize( cfo, additional_len );

    snprintf( cfo->compile_macro + cfo->compile_macro_count, additional_len + 1, " -D %s=%d", name, value );
    cfo->compile_macro_count += additional_len;
   
    long unsigned int output_count_base = count_base_10_digits( value );
    LOGV( "new define string, %lu digits: \"%s\"", output_count_base, cfo->compile_macro );
}

void compile_flag_object_add_compiler_flag( struct CompileFlagObject* cfo, const char* value )
{
    size_t additional_len = strlen(" ") + strlen( value );

    resize( cfo, additional_len );

    snprintf( cfo->compile_macro + cfo->compile_macro_count, additional_len + 1, " %s", value );
    cfo->compile_macro_count += additional_len;

    LOGV( "new define string: \"%s\"", cfo->compile_macro );
}
