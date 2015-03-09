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

cl_mem opencl_load_image( const char *input_filename )
{
    uint8_t *data;
    unsigned width,height;
    unsigned lode_error;
    cl_mem input_image = NULL;

    lode_error = lodepng_decode32_file( &data, &width, &height, input_filename );

    LOGV( "Picture dimensions: (%d,%d)", width, height );

    if( lode_error == 0 )
    {
        cl_int errcode_ret;
        cl_image_format image_format = {
            CL_RGBA,
            CL_UNSIGNED_INT8
        };

        input_image = clCreateImage2D(
            opencl_loader_get_context(),
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            &image_format,
            width,
            height,
            width*sizeof(uint8_t)*4,
            data,
            &errcode_ret 
        );
        ASSERT_BUF(input_image, errcode_ret);
    }
    else
    {
        LOGE("Unable to load image: %s", input_filename );
    }

    return input_image;
}

static const size_t global_work_offset[] = { 0,0 };
static const size_t global_work_size[] = { width, height };
static const size_t local_work_size[] = { 2, 2 };

void opencl_save_buffer_to_image( const char *output_filename )
{
    cl_event buffer_read_event;
    errcode_ret = clEnqueueReadBuffer( command_queue,
        desaturated_image,
        false,
        0,
        sizeof( cl_float ) * width * height,
        output_desaturated_image,
        1,
        &kernel_gaussy_event,
        &buffer_read_event
    );

    if( errcode_ret == CL_SUCCESS )
    {
        clWaitForEvents( 1, &buffer_read_event );

        printf( "profile: %lu %lu %lu\n",
            opencl_util_getduration( kernel_desaturate_event ),
            opencl_util_getduration( kernel_gaussx_event ),
            opencl_util_getduration( kernel_gaussy_event ) 
        );

        for( int i = 0; i < width * height; i++ )
        {
            //LOGV( "%f", output_desaturated_image[i] );
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

//Takes a monochrome single channel cl_float array and blurs it according to sigma
void opencl_test_run_gaussxy( 
    cl_mem buf_input, 
    cl_mem buf_output, 
    int width, 
    int height, 
    float sigma, 
    cl_uint num_events_in_wait_list, 
    const cl_event* event_wait_list, 
    cl_event* event )
{
    cl_program program_gauss_cl  = opencl_program_load( "kernels/gauss.cl" );
    cl_kernel kernel_gaussx      = opencl_loader_load_kernel( program_gauss_cl, "gaussx" );
    cl_kernel kernel_gaussy      = opencl_loader_load_kernel( program_gauss_cl, "gaussy" );

    const cl_int cl_width = width;
    const cl_int cl_height = height;

    size_t gauss_kernel_size;
    cl_float* gauss_kernel_line = generate_gauss_kernel_line( &gauss_kernel_size, sigma );

    /* Kernel output
    for( int i = 0; i < gauss_kernel_size; i++ )
    {
        LOGV( "%f", gauss_kernel_line[i] );
    }

    LOGV( "Gauss kernel size %zd", gauss_kernel_size );
    */

    cl_mem gaussx_image = clCreateBuffer( 
        opencl_loader_get_context(),
        CL_MEM_READ_WRITE,
        sizeof( cl_float ) * width * height,
        NULL,
        &errcode_ret
    );
    ASSERT_BUF(gaussx_image, errcode_ret);
    
    cl_mem gauss_kernel_buffer = clCreateBuffer(
            opencl_loader_get_context(),
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof( cl_float ) * gauss_kernel_size,
            gauss_kernel_line,
            &errcode_ret
    );
    ASSERT_BUF(gauss_kernel_buffer, errcode_ret);

    cl_event kernel_gaussx_event;
    const cl_int kernel_radius = gauss_kernel_size/2;
    clSetKernelArg( kernel_gaussx, 0, sizeof( cl_mem ), &gauss_kernel_buffer );
    clSetKernelArg( kernel_gaussx, 1, sizeof( cl_int ), &kernel_radius );
    clSetKernelArg( kernel_gaussx, 2, sizeof( cl_mem ), &buf_input );
    clSetKernelArg( kernel_gaussx, 3, sizeof( cl_mem ), &gaussx_image );
    clSetKernelArg( kernel_gaussx, 4, sizeof( cl_int ), &cl_width );
    errcode_ret = clEnqueueNDRangeKernel( command_queue, 
        kernel_gaussx, 
        2,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        &kernel_gaussx_event
    );
    ASSERT_ENQ( kernel_gaussx, errcode_ret );

    cl_event kernel_gaussy_event;
    cl_int cl_height = height;
    clSetKernelArg( kernel_gaussy, 0, sizeof( cl_mem ), &gauss_kernel_buffer );
    clSetKernelArg( kernel_gaussy, 1, sizeof( cl_int ), &kernel_radius );
    clSetKernelArg( kernel_gaussy, 2, sizeof( cl_mem ), &gaussx_image );
    clSetKernelArg( kernel_gaussy, 3, sizeof( cl_mem ), &buf_output );
    clSetKernelArg( kernel_gaussy, 4, sizeof( cl_int ), &cl_width );
    clSetKernelArg( kernel_gaussy, 5, sizeof( cl_int ), &cl_height );
    errcode_ret = clEnqueueNDRangeKernel( command_queue, 
        kernel_gaussy, 
        2,
        global_work_offset,
        global_work_size,
        local_work_size,
        1,
        &kernel_gaussx_event,
        finish
    );
    ASSERT_ENQ( kernel_gaussy, errcode_ret );

    clReleaseKernel( kernel_gaussx );
    clReleaseKernel( kernel_gaussy );
    clReleaseProgram( program_gauss_cl );
    clReleaseMemObject( gaussx_image );
}

void opencl_test_desaturate_image( cl_mem input_buffer,  )
{
    cl_program program_gauss_cl  = opencl_program_load( "kernels/gauss.cl" );
    cl_kernel kernel_desaturate  = opencl_loader_load_kernel( program_gauss_cl, "desaturate" );

    if( kernel_desaturate )
    {
        cl_float *output_desaturated_image = malloc( sizeof(cl_float) * width * height );
        uint8_t *output_desaturated_image_rgba = malloc( sizeof(uint8_t) * 4 * width * height );
        memset( output_desaturated_image, 0, sizeof(uint8_t) * 4 * width * height );


        cl_mem desaturated_image = clCreateBuffer(
            opencl_loader_get_context(),
            CL_MEM_READ_WRITE,
            sizeof( cl_float ) * width * height,
            NULL,
            &errcode_ret 
        );
        ASSERT_BUF(output_buffer, errcode_ret);
        //cl_mem gaussxy_image = desaturated_image;


        cl_command_queue command_queue = opencl_loader_get_command_queue();

        clSetKernelArg( kernel_desaturate, 0, sizeof( cl_mem ), &input_image );
        clSetKernelArg( kernel_desaturate, 1, sizeof( cl_mem ), &desaturated_image );
        clSetKernelArg( kernel_desaturate, 2, sizeof( cl_int ), &cl_width );
        cl_event kernel_desaturate_event;
        errcode_ret = clEnqueueNDRangeKernel( command_queue,
            kernel_desaturate,
            2,
            global_work_offset,
            global_work_size,
            local_work_size,
            0,
            NULL,
            &kernel_desaturate_event 
        );
        ASSERT_ENQ( kernel_desaturate, errcode_ret );

        clReleaseMemObject(desaturated_image);
        }

        clReleaseKernel( kernel_desaturate );
    }
}
