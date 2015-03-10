#include "opencl_fd.h"

#include <stdbool.h>
#include <stdint.h>

#include "opencl_error.h"
#include "opencl_program.h"
#include "lodepng.h"

bool opencl_fd_load_image( const char *input_filename, struct FD* state )
{
    unsigned lode_error;

    state->buffer_l_float_1 = NULL;
    state->buffer_l_float_2 = NULL;

    lode_error = lodepng_decode32_file( &state->rgba_host, &state->width, &state->height, input_filename );

    LOGV( "Picture dimensions: (%d,%d)", state->width, state->height );

    if( lode_error == 0 )
    {
        cl_int errcode_ret;
        cl_image_format image_format = {
            CL_RGBA,
            CL_UNSIGNED_INT8
        };

        state->image_rgba_char = clCreateImage2D(
            opencl_loader_get_context(),
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            &image_format,
            state->width,
            state->height,
            state->width*sizeof(uint8_t)*4,
            state->rgba_host,
            &errcode_ret 
        );
        ASSERT_BUF(input_image, errcode_ret);

        state->buffer_l_float_1 = clCreateBuffer(
            opencl_loader_get_context(),
            CL_MEM_READ_WRITE,
            sizeof( cl_float ) * state->width * state->height,
            NULL,
            &errcode_ret 
        );
        ASSERT_BUF(state->buffer_l_float_1, errcode_ret);

        state->buffer_l_float_2 = clCreateBuffer(
            opencl_loader_get_context(),
            CL_MEM_READ_WRITE,
            sizeof( cl_float ) * state->width * state->height,
            NULL,
            &errcode_ret 
        );
        ASSERT_BUF(state->buffer_l_float_2, errcode_ret);

        return true;
    }
    else
    {
        LOGE("Unable to load image: %s", input_filename );
        return false;
    }
}

void opencl_fd_save_buffer_to_image( 
    const char * name, 
    struct FD* state, 
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list
    )
{
    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_event buffer_read_event;
    cl_int errcode_ret;

    cl_float* output_desaturated_image = (cl_float*)malloc(state->width*state->height*sizeof(cl_float));

    errcode_ret = clEnqueueReadBuffer( command_queue,
        state->buffer_l_float_1,
        false,
        0,
        sizeof( cl_float ) * state->width * state->height,
        output_desaturated_image,
        num_events_in_wait_list,
        event_wait_list,
        &buffer_read_event
    );

    if( errcode_ret == CL_SUCCESS )
    {
        clWaitForEvents( 1, &buffer_read_event );
        uint8_t* output_desaturated_image_l = malloc( sizeof( uint8_t ) * state->width * state->height );
        for( int i = 0; i < state->width * state->height; i++ )
        {
            //LOGV( "%f", output_desaturated_image[i] );
            output_desaturated_image_l[i] = output_desaturated_image[i];
        } 

        lodepng_encode_file( name, output_desaturated_image_l, state->width, state->height, LCT_GREY, 8 );
        free( output_desaturated_image_l );
    }
    else
    {
        CLERR( "Unable to read output buffer", errcode_ret );
    }
}

//Takes a monochrome single channel cl_float array and blurs it according to sigma
/*
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
    static const size_t global_work_offset[] = { 0,0 };
    static const size_t global_work_size[] = { width, height };
    static const size_t local_work_size[] = { 2, 2 };

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
    /

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
*/

bool opencl_fd_desaturate_image( 
    struct FD *state,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
)
{
    const size_t global_work_offset[] = { 0,0 };
    const size_t global_work_size[] = { state->width, state->height };
    const size_t local_work_size[] = { 2, 2 };

    cl_program program_gauss_cl  = opencl_program_load( "kernels/gauss.cl" );
    cl_kernel kernel_desaturate  = opencl_loader_load_kernel( program_gauss_cl, "desaturate" );

    if( kernel_desaturate )
    {
        cl_command_queue command_queue = opencl_loader_get_command_queue();

        cl_int cl_width = state->width;
        clSetKernelArg( kernel_desaturate, 0, sizeof( cl_mem ), &state->image_rgba_char );
        clSetKernelArg( kernel_desaturate, 1, sizeof( cl_mem ), &state->buffer_l_float_1 );
        clSetKernelArg( kernel_desaturate, 2, sizeof( cl_int ), &cl_width );

        cl_int errcode_ret = clEnqueueNDRangeKernel( command_queue,
            kernel_desaturate,
            2,
            global_work_offset,
            global_work_size,
            local_work_size,
            num_events_in_wait_list,
            event_wait_list,
            event
        );
        ASSERT_ENQ( kernel_desaturate, errcode_ret );

        clReleaseKernel( kernel_desaturate );

        return true;
    }
    return false;
}
