#include "opencl_test.h"
#include "opencl_handler.h"
#include "opencl_error.h"
#include "log.h"
#include "lodepng.h"

bool opencl_test_run()
{
    cl_program program = opencl_loader_load_program( "kernels/test_1.cl" );
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

void opencl_test_desaturate_image( const char *input_filename, const char* output_filename )
{
   cl_kernel kernel_desaturate = opencl_loader_load_kernel( "kernels/gauss.cl", "desaturate" );

   if( kernel_desaturate )
   {
        uint8_t *data;
        unsigned width,height;
        unsigned lode_error;

        lode_error = lodepng_decode32_file( &data, &width, &height, input_filename );

        LOGV( "Picture dimensions: (%d,%d)", width, height );

        cl_float *output_desaturated_image = malloc( sizeof(cl_float) * width * height );
        uint8_t *output_desaturated_image_rgba = malloc( sizeof(uint8_t) * 4 * width * height );
        memset( output_desaturated_image, 0, sizeof(uint8_t) * 4 * width * height );

        if( lode_error == 0 )
        {
            cl_int errcode_ret;
            cl_image_format image_format = {
                CL_RGBA,
                CL_UNSIGNED_INT8
            };

            cl_mem input_image = clCreateImage2D(
                opencl_loader_get_context(),
                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                &image_format,
                width,
                height,
                width*sizeof(uint8_t)*4,
                data,
                &errcode_ret 
            );
           CHKBUF(input_image, errcode_ret);

           cl_mem output_buffer = clCreateBuffer(
                opencl_loader_get_context(),
                CL_MEM_WRITE_ONLY,
                sizeof( cl_float ) * width * height,
                NULL,
                &errcode_ret 
           );
           CHKBUF(output_buffer, errcode_ret);
           CLERR( "Test", errcode_ret );

            if( input_image && output_buffer )
            {
                cl_int cl_width = width;
                clSetKernelArg( kernel_desaturate, 0, sizeof( cl_mem ), &input_image );
                clSetKernelArg( kernel_desaturate, 1, sizeof( cl_mem ), &output_buffer );
                clSetKernelArg( kernel_desaturate, 2, sizeof( cl_int ), &cl_width );

                cl_command_queue command_queue = opencl_loader_get_command_queue();

                const size_t global_work_offset[] = { 0,0 };
                const size_t global_work_size[] = { width, height };
                const size_t local_work_size[] = { 2, 2 };

                cl_event kernel_event;

                errcode_ret = clEnqueueNDRangeKernel( command_queue,
                    kernel_desaturate,
                    2,
                    global_work_offset,
                    global_work_size,
                    local_work_size,
                    0,
                    NULL,
                    &kernel_event 
                );

                if( errcode_ret == CL_SUCCESS )
                {

                    cl_event buffer_read_event;
                    errcode_ret = clEnqueueReadBuffer( command_queue,
                        output_buffer,
                        true,
                        0,
                        sizeof( cl_float ) * width * height,
                        output_desaturated_image,
                        1,
                        &kernel_event,
                        &buffer_read_event
                    );

                    if( errcode_ret == CL_SUCCESS )
                    {

                        clWaitForEvents( 1, &buffer_read_event );

                        for( int i = 0; i < width * height; i++ )
                        {
                            LOGV( "%f", output_desaturated_image[i] );
                            output_desaturated_image_rgba[i*4+0] = output_desaturated_image[i];
                            output_desaturated_image_rgba[i*4+1] = output_desaturated_image[i];
                            output_desaturated_image_rgba[i*4+2] = output_desaturated_image[i];
                            output_desaturated_image_rgba[i*4+3] = 255;
                        } 

                        lodepng_encode32_file( output_filename, output_desaturated_image_rgba, width, height );
                    }
                    else
                    {
                        CLERR( "Unable to read output buffer", errcode_ret );
                    }
                }
                else
                {
                     CLERR( "Unable to enqueue kernel", errcode_ret );
                }

            }

            clReleaseMemObject(input_image);
            clReleaseMemObject(output_buffer);

            free( data );
        }
        else
        {
            LOGE(":Unable to load image: %s", input_filename );
        }

        clReleaseKernel( kernel_desaturate );
        free( output_desaturated_image );
        free( output_desaturated_image_rgba );
    }
}
