#include "opencl_program.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "log.h"
#include "opencl_loader.h"
#include "opencl_error.h"
#include "opencl_config.h"
#include "util.h"
#include "string_object.h"
#include "_ref_opencl_kernels.h"
#include "_opt_opencl_kernels.h"

struct ProgramList
{
    char* name;
    cl_program program;
    struct ProgramList* next;
};

static struct ProgramList* program_root = NULL;

static void program_add( const char* name, cl_program program )
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

void opencl_program_close()
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

struct KernelData
{
    const char* data;
    size_t size;
};

struct KernelData get_kernel( const char* name, bool reference_mode )
{
    struct KernelData ret;
    const void** kernel_files;
    const size_t* kernel_sizes;
    if( reference_mode )
    {
        kernel_files = ref_kernel_files;
        kernel_sizes = ref_kernel_sizes;
        LOGV( "Warning: using the reference kernel" );
    }
    else
    {
        kernel_files = opt_kernel_files;
        kernel_sizes = opt_kernel_sizes;
    }

    int i = 0;

    while( kernel_files[i*2] != NULL && strcmp( kernel_files[i*2], name ) != 0  )
    {
        i++;
    }

    ret.data = kernel_files[i*2+1];

    if( ret.data == NULL )
    {
        //If we lack the optimized version we fallback on the reference impl
        if( reference_mode == false )
        {
            ret = get_kernel( name, !reference_mode );
        }
        //Otherwise we are unable to find the kernel
    }
    else
    {
        ret.size = kernel_sizes[i];
    }

    return ret;
}

static void compile( const char* name, const char* cfo )
{
    cl_context context = opencl_loader_get_context();
    cl_device_id device = opencl_loader_get_device();
    cl_program program;

    struct KernelData kd = get_kernel( name, opencl_run_reference_mode );

    if( kd.data != NULL )
    {
        LOGV( "Compiling -> program:%s,len:%zu\n", name, kd.size );

        cl_int errcode_ret; 
        program = clCreateProgramWithSource( 
                context, 
                1, 
                (const char**)&kd.data,
                &kd.size, 
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
                    cfo,
                    NULL, 
                    NULL 
            );

            if( errcode_ret != CL_SUCCESS )
            {
                CLERR( "Unable to compile program", errcode_ret );
            }

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
                LOGV( "Program compile output" );
                char compile_log[compile_output_size];
                errcode_ret = clGetProgramBuildInfo( program, 
                    device, 
                    CL_PROGRAM_BUILD_LOG,
                    compile_output_size,
                    compile_log,
                    NULL );
                long unsigned int c_compile_output_size = compile_output_size;
                LOGV( "%lu:%s", c_compile_output_size, compile_log );
            }

            program_add( name, program );
        }

    }
    else
    {
        LOGE( "Missing kernel, unable to compile: %s", name );
    }
}

void opencl_program_compile( const char** programs, const char *cfo )
{
    int i = 0;

    while( programs[i] != NULL )
    {
        LOGV( "Compiling: %s with: %s", programs[i], cfo );
        compile( programs[i], cfo );
        i++;
    }

    //clUnloadCompiler(); //Hint to the program that we are done using the compiler.
}

cl_program opencl_program_load( const char* name )
{
    cl_program program = opencl_program_get( name );

    if( !program )
    {
        LOGE( "Trying to get nonexistant program\n" );
    }

    return program;
}
