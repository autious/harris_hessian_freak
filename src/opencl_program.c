#include "opencl_program.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "log.h"
#include "opencl_handler.h"
#include "opencl_error.h"

struct ProgramList
{
    char* name;
    cl_program program;
    struct ProgramList* next;
};

static struct ProgramList* program_root = NULL;

static void opencl_program_add( const char* name, cl_program program )
{
    struct ProgramList ** cur = &program_root;

    while( (*cur) != NULL )
    {
        cur = (&(*cur)->next);
    }

    (*cur) = (struct ProgramList*)malloc( sizeof(struct ProgramList) );

    (*cur)->name = malloc( sizeof( char ) * (strlen( name ) + 1) );
    strcpy( (*cur)->name, name );
    (*cur)->program = program;
    (*cur)->next = NULL;
    clRetainProgram( program ); 
}

static cl_program opencl_program_get( const char* name )
{
    struct ProgramList *cur = program_root; 

    while( cur != NULL )
    {
        if( strcmp( cur->name, name ) == 0 )
        {
            clRetainProgram( cur->program );
            return cur->program;
        }
        cur = cur->next;
    }

    return NULL;
}

void opencl_program_free()
{
    struct ProgramList *cur = program_root, *next;

    while( cur != NULL )
    {
        next = cur->next;
        free( cur->name );
        clReleaseProgram( cur->program );
        free(cur);
        cur = next;
    }
}

cl_program opencl_program_load( const char* name )
{
    cl_program program = opencl_program_get( name );
    cl_context context = opencl_loader_get_context();
    cl_device_id device = opencl_loader_get_device();

    if( !program )
    {
        const int BUFSIZE = 128;
        size_t size = 0;     
        size_t read_bytes = 0;

        char *string = NULL;

        FILE* f = fopen( name, "r" ); 

        if( f )
        {
            while( !feof( f ) && !ferror( f ) )
            {
                if( size < read_bytes + BUFSIZE )
                {
                    size += BUFSIZE;
                    string = (char*)realloc( string, size );
                }

                if( string != NULL )
                {
                    read_bytes += fread( string + read_bytes, 1, BUFSIZE, f );
                }
                else
                {
                    LOGE( "Unable to allocate memory for opencl kernel temp string" );
                    break; //Unable to allocate more memory for string
                }
            }

            if( !ferror( f ) && string != NULL )
            {
                char* outputstream = (char*)calloc( read_bytes + 1,1 );
                memcpy( outputstream, string, read_bytes );
                LOGV( "Program:%s,%zu,%zu\n%s", name, size, read_bytes, outputstream );
                free( outputstream );

                cl_int errcode_ret; 
                program = clCreateProgramWithSource( 
                        context, 
                        1, 
                        (const char**)&string, 
                        &read_bytes, 
                        &errcode_ret 
                );
        
                if( errcode_ret != CL_SUCCESS )
                {
                    CLERR( "Unable to load program:%s", errcode_ret);
                }
                else
                {
                    errcode_ret = clBuildProgram( 
                            program, 
                            1, 
                            &device, 
                            "-cl-fast-relaxed-math -cl-std=CL1.1", 
                            NULL, 
                            NULL 
                    );

                    if( errcode_ret != CL_SUCCESS )
                    {
                        CLERR( "Unable to compile program", errcode_ret );
                    }

                    LOGV( "Program compile output" );

                    size_t compile_output_size;

                    errcode_ret = clGetProgramBuildInfo( program, 
                        device, 
                        CL_PROGRAM_BUILD_LOG,
                        0,
                        NULL,
                        &compile_output_size );

                    if( errcode_ret != CL_SUCCESS )
                    {
                        CLERR( "Unable to get size of compile log", errcode_ret );
                    }
                    else
                    {
                        char compile_log[compile_output_size];
                        errcode_ret = clGetProgramBuildInfo( program, 
                            device, 
                            CL_PROGRAM_BUILD_LOG,
                            compile_output_size,
                            compile_log,
                            NULL );
                        LOGV( "%lu:%s", compile_output_size, compile_log );
                    }
                }
            }
            else
            {
                LOGE( "Error reading file:\"%s\"", name );
            }
        }
        else
        {
            LOGE( "Unable to open the file:\"%s\"", name );
        }

        free( string );

        opencl_program_add( name, program );
    }
    else
    {
        LOGV( "Program %s was cached", name );
    }

    return program;
}
