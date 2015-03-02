#include "opencl_test.h"
#include "opencl_handler.h"
#include "opencl_error.h"
#include "log.h"

bool opencl_test_run()
{
    cl_program program = opencl_loader_load_program( "kernels/test_1.cl" );
    cl_int errcode_ret;
    cl_kernel kernel = clCreateKernel( program, "main", &errcode_ret );

    if( errcode_ret == CL_SUCCESS )
    {
        cl_mem mem_in = clCreateBuffer( opencl_loader_get_context(), CL_MEM_READ_WRITE, sizeof( cl_float ) * 128, NULL, &errcode_ret );
        cl_mem mem_out = clCreateBuffer( opencl_loader_get_context(), CL_MEM_READ_WRITE, sizeof( cl_float ) * 128, NULL, &errcode_ret );


        if( errcode_ret == CL_SUCCESS )
        {
            cl_event write_event;
            cl_float inputValues[128];
            cl_float outputValues[128];
            for( int i = 0; i < 128; i++ )
                inputValues[i] = i;

            errcode_ret = clEnqueueWriteBuffer( 
                    opencl_loader_get_command_queue(), 
                    mem_in, 
                    false, 
                    0, 
                    sizeof( cl_float ) * 128, 
                    inputValues, 
                    0, 
                    NULL, 
                    &write_event );

            if( errcode_ret != CL_SUCCESS )
                CLERR( "Unable to enq range write", errcode_ret );

            cl_event waitfor[] = { write_event };
            cl_event kernel_event;
            size_t global_work_size[] = { 128 };
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
                    sizeof( cl_float ) * 128, 
                    outputValues, 
                    1, 
                    &kernel_event, 
                    &read_event );

            if( errcode_ret != CL_SUCCESS )
                CLERR( "Unable to enq read buffer", errcode_ret );

            clWaitForEvents( 1, &read_event ); 

            for( int i = 0; i < 128; i++ )
                LOGV( "%f\n", outputValues[i] );

            clReleaseMemObject( mem_in );
            clReleaseMemObject( mem_out );
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
