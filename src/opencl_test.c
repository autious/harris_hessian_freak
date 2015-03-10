#include "opencl_test.h"

#include <assert.h>

#include "opencl_handler.h"
#include "opencl_error.h"
#include "opencl_util.h"
#include "opencl_program.h"
#include "log.h"
#include "lodepng.h"
#include "gauss_kernel.h"


bool opencl_test_run()
{
    cl_program program = opencl_program_load( "kernels/test_1.cl" );
    cl_int errcode_ret;
    cl_kernel kernel = clCreateKernel( program, "main", &errcode_ret );

    const int TEST_SIZE = 1024*256;
    if( errcode_ret == CL_SUCCESS )
    {
        cl_mem mem_in = clCreateBuffer( opencl_loader_get_context(), CL_MEM_READ_WRITE, sizeof( cl_float ) * TEST_SIZE, NULL, &errcode_ret );
        cl_mem mem_out = clCreateBuffer( opencl_loader_get_context(), CL_MEM_READ_WRITE, sizeof( cl_float ) * TEST_SIZE, NULL, &errcode_ret );


        if( errcode_ret == CL_SUCCESS )
        {
            cl_event write_event;
            cl_float inputValues[TEST_SIZE];
            cl_float outputValues[TEST_SIZE];
            for( int i = 0; i < TEST_SIZE; i++ )
                inputValues[i] = i;

            errcode_ret = clEnqueueWriteBuffer( 
                    opencl_loader_get_command_queue(), 
                    mem_in, 
                    false, 
                    0, 
                    sizeof( cl_float ) * TEST_SIZE, 
                    inputValues, 
                    0, 
                    NULL, 
                    &write_event );

            if( errcode_ret != CL_SUCCESS )
                CLERR( "Unable to enq range write", errcode_ret );

            cl_event waitfor[] = { write_event };
            cl_event kernel_event;
            size_t global_work_size[] = { TEST_SIZE };
            size_t local_work_size[] = { 16 };

            clSetKernelArg( kernel, 0, sizeof( cl_mem ), &mem_in );
            clSetKernelArg( kernel, 1, sizeof( cl_mem ), &mem_out );
            errcode_ret = clEnqueueNDRangeKernel( 
                    opencl_loader_get_command_queue(), 
                    kernel, 
                    1, 
                    NULL, 
                    global_work_size, 
                    local_work_size, 
                    1, 
                    waitfor, 
                    &kernel_event );

            if( errcode_ret != CL_SUCCESS )
                CLERR( "Unable to enq range kernel", errcode_ret );

            cl_event read_event;
            errcode_ret = clEnqueueReadBuffer( opencl_loader_get_command_queue(), 
                    mem_out, 
                    false, 
                    0, 
                    sizeof( cl_float ) * TEST_SIZE, 
                    outputValues, 
                    1, 
                    &kernel_event, 
                    &read_event );

            if( errcode_ret != CL_SUCCESS )
                CLERR( "Unable to enq read buffer", errcode_ret );

            clWaitForEvents( 1, &read_event ); 

            clReleaseMemObject( mem_in );
            clReleaseMemObject( mem_out );

            cl_ulong start=0, end=0;
            clGetEventProfilingInfo( kernel_event, CL_PROFILING_COMMAND_START, sizeof( cl_ulong ), &start, NULL );
            clGetEventProfilingInfo( kernel_event, CL_PROFILING_COMMAND_END, sizeof( cl_ulong ), &end, NULL );
            
            LOGV( "Kernel execution time: %lu", end-start ); 


        }
        else
        {
           CLERR( "Unable to create memory object", errcode_ret );
        }

        clReleaseKernel( kernel );
    }
    else
    {
        CLERR("Unable to create kernel from program",errcode_ret );  
    }

    clReleaseProgram( program );

    return true;
}

