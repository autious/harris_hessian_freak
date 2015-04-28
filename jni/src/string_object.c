/*
* Copyright (c) 2015, Max Danielsson <work@autious.net> and Thomas Sievert
* All rights reserved.
* 
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the organization nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "string_object.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "util.h"
#include "log.h"

static void resize( struct StringObject *cfo, size_t additional_len )
{
    if( cfo->str_count + additional_len + 1 > cfo->str_size )
    {
        cfo->str_size = cfo->str_count + additional_len + 1;
        cfo->str = realloc( cfo->str, sizeof( char ) * cfo->str_size );
    }
}

void string_object_init( struct StringObject* cfo )
{
    cfo->str = NULL;
    cfo->str_count = 0;
    cfo->str_size = 0;
}

void string_object_free( struct StringObject *cfo )
{
    free( cfo->str );
    cfo->str = NULL;
}

void string_object_append_string( struct StringObject *cfo, const char* value )
{
    size_t additional_len = strlen( value );

    resize( cfo, additional_len );

    snprintf( cfo->str + cfo->str_count, additional_len + 1, "%s", value );
    cfo->str_count += additional_len;
}

void string_object_add_define_integer( struct StringObject* cfo, const char* name, int value )
{
    size_t additional_len = strlen( " -D " ) + strlen( name ) + 1 + count_base_10_digits( value );

    resize( cfo, additional_len );

    snprintf( cfo->str + cfo->str_count, additional_len + 1, " -D %s=%d", name, value );
    cfo->str_count += additional_len;
   
    long unsigned int output_count_base = count_base_10_digits( value );
    LOGV( "new define string, %lu digits: \"%s\"", output_count_base, cfo->str );
}

void string_object_add_define( struct StringObject* cfo, const char* name )
{
    size_t additional_len = strlen( " -D" ) + strlen( name );

    resize( cfo, additional_len );

    snprintf( cfo->str + cfo->str_count, additional_len + 1, " -D%s", name );
    cfo->str_count += additional_len;
   
    LOGV( "new define string \"%s\"",  cfo->str );
}

void string_object_add_compiler_flag( struct StringObject* cfo, const char* value )
{
    size_t additional_len = strlen(" ") + strlen( value );

    resize( cfo, additional_len );

    snprintf( cfo->str + cfo->str_count, additional_len + 1, " %s", value );
    cfo->str_count += additional_len;

    LOGV( "new define string: \"%s\"", cfo->str );
}
