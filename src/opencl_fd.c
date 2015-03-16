#include "opencl_fd.h"

#include <stdbool.h>
#include <stdint.h>

#include "opencl_error.h"
#include "opencl_program.h"
#include "gauss_kernel.h"
#include "lodepng.h"

bool opencl_fd_load_png_file( const char *input_filename, struct FD* state )
{
    unsigned lode_error;

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
    cl_mem in,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list
    )
{
    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_event buffer_read_event;
    cl_int errcode_ret;

    cl_float* output_desaturated_image = (cl_float*)malloc(state->width*state->height*sizeof(cl_float));

    errcode_ret = clEnqueueReadBuffer( command_queue,
        in,
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

cl_mem opencl_fd_create_image_buffer( struct FD* state )
{
    cl_mem ret;
    opencl_fd_create_image_buffers( state, &ret, 1 );
    return ret;
}

void opencl_fd_create_image_buffers( struct FD* state, cl_mem* buffers, size_t count )
{
    cl_int errcode_ret;
    for( int i = 0; i < count; i++ )
    {
        buffers[i] = clCreateBuffer(
            opencl_loader_get_context(),
            CL_MEM_READ_WRITE,
            sizeof( cl_float ) * state->width * state->height,
            NULL,
            &errcode_ret 
        );
        ASSERT_BUF(create_image_buffers, errcode_ret);
    }
}

void opencl_fd_release_image_buffer( struct FD* state, cl_mem buffer )
{
    opencl_fd_release_image_buffers( state, &buffer, 1 );
}

void opencl_fd_release_image_buffers( struct FD* state, cl_mem* buffers, size_t count )
{
    for( int i = 0; i < count; i++ )
    {
        clReleaseMemObject( buffers[i] );
    }
}

static void free_host_ptr_callback( 
    cl_event event, 
    cl_int event_command_exec_status, 
    void *user_data
)
{
    free( user_data );
}


//Takes a monochrome single channel cl_float array and blurs it according to sigma
bool opencl_fd_run_gaussxy( 
    float sigma, 
    struct FD* state,
    cl_mem in,
    cl_mem middle,
    cl_mem out,
    cl_uint num_events_in_wait_list, 
    const cl_event* event_wait_list, 
    cl_event* event )
{
    const size_t global_work_offset[] = { 0,0 };
    const size_t global_work_size[] = { state->width, state->height };
    const size_t local_work_size[] = { 2, 2 };

    cl_program program_gauss_cl  = opencl_program_load( "kernels/gauss.cl" );
    cl_kernel kernel_gaussx      = opencl_loader_load_kernel( program_gauss_cl, "gaussx" );
    cl_kernel kernel_gaussy      = opencl_loader_load_kernel( program_gauss_cl, "gaussy" );
    cl_command_queue command_queue = opencl_loader_get_command_queue();

    const cl_int cl_width = state->width;
    const cl_int cl_height = state->height;

    size_t gauss_kernel_size;
    cl_float* gauss_kernel_line = generate_gauss_kernel_line( &gauss_kernel_size, sigma );

    /* Kernel output
    for( int i = 0; i < gauss_kernel_size; i++ )
    {
        LOGV( "%f", gauss_kernel_line[i] );
    }

    LOGV( "Gauss kernel size %zd", gauss_kernel_size );
    */

    cl_int errcode_ret;
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
    clSetKernelArg( kernel_gaussx, 2, sizeof( cl_mem ), &in );
    clSetKernelArg( kernel_gaussx, 3, sizeof( cl_mem ), &middle );
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

    clSetKernelArg( kernel_gaussy, 0, sizeof( cl_mem ), &gauss_kernel_buffer );
    clSetKernelArg( kernel_gaussy, 1, sizeof( cl_int ), &kernel_radius );
    clSetKernelArg( kernel_gaussy, 2, sizeof( cl_mem ), &middle );
    clSetKernelArg( kernel_gaussy, 3, sizeof( cl_mem ), &out );
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
        event
    );
    ASSERT_ENQ( kernel_gaussy, errcode_ret );

    clReleaseKernel( kernel_gaussx );
    clReleaseKernel( kernel_gaussy );
    clReleaseProgram( program_gauss_cl );
    clReleaseMemObject( gauss_kernel_buffer );
    //Free the kernel memory when done.
    clSetEventCallback( *event, CL_COMPLETE, free_host_ptr_callback, gauss_kernel_line );

    return true;
}

bool opencl_fd_desaturate_image( 
    struct FD *state,
    cl_mem out,
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
        clSetKernelArg( kernel_desaturate, 1, sizeof( cl_mem ), &out );
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

bool opencl_fd_derivate_image( struct FD* state,
    cl_mem in,
    cl_mem ddxout,
    cl_mem ddyout,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
)
{
    const size_t global_work_offset[] = { 0,0 };
    const size_t global_work_size[] = { state->width, state->height };
    const size_t local_work_size[] = { 2, 2 };

    cl_program program_gauss_cl  = opencl_program_load( "kernels/derivate.cl" );
    cl_kernel kernel_derivate  = opencl_loader_load_kernel( program_gauss_cl, "derivate" );

    if( kernel_derivate )
    {
        cl_command_queue command_queue = opencl_loader_get_command_queue();

        cl_int cl_width = state->width;
        cl_int cl_height = state->height;
        clSetKernelArg( kernel_derivate, 0, sizeof( cl_mem ), &in );
        clSetKernelArg( kernel_derivate, 1, sizeof( cl_mem ), &ddxout );
        clSetKernelArg( kernel_derivate, 2, sizeof( cl_mem ), &ddyout );
        clSetKernelArg( kernel_derivate, 3, sizeof( cl_int ), &cl_width );
        clSetKernelArg( kernel_derivate, 4, sizeof( cl_int ), &cl_height );

        cl_int errcode_ret = clEnqueueNDRangeKernel( command_queue,
            kernel_derivate,
            2,
            global_work_offset,
            global_work_size,
            local_work_size,
            num_events_in_wait_list,
            event_wait_list,
            event
        );
        ASSERT_ENQ( kernel_desaturate, errcode_ret );

        clReleaseKernel( kernel_derivate );

        return true;
    }
    return false;
}

bool opencl_fd_second_moment_matrix_elements( struct FD* state,
    cl_mem ddx,
    cl_mem ddy,
    cl_mem xx,
    cl_mem xy,
    cl_mem yy,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
)
{
    const size_t global_work_offset[] = { 0 };
    const size_t global_work_size[] = { state->width * state->height };
    const size_t local_work_size[] = { 16 };

    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_program program = opencl_program_load( "kernels/smme.cl" );
    cl_kernel smme_kernel 
        = opencl_loader_load_kernel( program, "second_moment_matrix_elements" );

    cl_int errcode_ret; 

    clSetKernelArg( smme_kernel, 0, sizeof( cl_mem ), &ddx );
    clSetKernelArg( smme_kernel, 1, sizeof( cl_mem ), &ddy );
    clSetKernelArg( smme_kernel, 2, sizeof( cl_mem ), &xx );
    clSetKernelArg( smme_kernel, 3, sizeof( cl_mem ), &xy );
    clSetKernelArg( smme_kernel, 4, sizeof( cl_mem ), &yy );

    errcode_ret = clEnqueueNDRangeKernel( command_queue,
        smme_kernel,
        1,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        event
    );
    ASSERT_ENQ( smme_kernel, errcode_ret );
    
    clReleaseProgram( program );
    clReleaseKernel( smme_kernel );

    return true;
}

bool opencl_fd_run_harris_corner_response( struct FD* state,
        cl_mem xx,
        cl_mem xy,
        cl_mem yy,
        cl_mem output,
        cl_float sigmaD,
        cl_uint num_events_in_wait_list,
        cl_event *event_wait_list,
        cl_event *event
)
{
    const size_t global_work_offset[] = { 0 };
    const size_t global_work_size[] = { state->width * state->height };
    const size_t local_work_size[] = { 16 };

    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_program program = opencl_program_load( "kernels/harris.cl" );
    cl_kernel harris_corner_response
        = opencl_loader_load_kernel( program, "harris_corner_response" );

    cl_int errcode_ret; 

    clSetKernelArg( harris_corner_response, 0, sizeof( cl_mem ), &xx );
    clSetKernelArg( harris_corner_response, 1, sizeof( cl_mem ), &xy );
    clSetKernelArg( harris_corner_response, 2, sizeof( cl_mem ), &yy );
    clSetKernelArg( harris_corner_response, 3, sizeof( cl_mem ), &output );
    clSetKernelArg( harris_corner_response, 4, sizeof( cl_float ), &sigmaD );

    errcode_ret = clEnqueueNDRangeKernel( command_queue,
        harris_corner_response,
        1,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        event
    );
    ASSERT_ENQ( harris_corner_response, errcode_ret );
    
    clReleaseProgram( program );
    clReleaseKernel( harris_corner_response );

    return true;
}

bool opencl_fd_run_harris_corner_suppression( struct FD* state,
        cl_mem in,
        cl_mem out,
        cl_uint num_events_in_wait_list,
        cl_event *event_wait_list,
        cl_event *event
)
{
    const size_t global_work_offset[] = { 0,0 };
    const size_t global_work_size[] = { state->width, state->height };
    const size_t local_work_size[] = { 2,2 };

    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_program program = opencl_program_load( "kernels/harris.cl" );
    cl_kernel harris_corner_suppression
        = opencl_loader_load_kernel( program, "harris_corner_suppression" );

    cl_int errcode_ret; 

    cl_int cl_width = state->width;
    cl_int cl_height = state->height;
    clSetKernelArg( harris_corner_suppression, 0, sizeof( cl_mem ), &in );
    clSetKernelArg( harris_corner_suppression, 1, sizeof( cl_mem ), &out );
    clSetKernelArg( harris_corner_suppression, 2, sizeof( cl_int ), &cl_width );
    clSetKernelArg( harris_corner_suppression, 3, sizeof( cl_int ), &cl_height );

    errcode_ret = clEnqueueNDRangeKernel( command_queue,
        harris_corner_suppression,
        2,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        event
    );
    ASSERT_ENQ( harris_corner_suppression, errcode_ret );
    
    clReleaseProgram( program );
    clReleaseKernel( harris_corner_suppression );

    return true;
}

void opencl_fd_free( struct FD* state, 
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list
  )
{
    clWaitForEvents( num_events_in_wait_list, event_wait_list );
    clReleaseMemObject( state->image_rgba_char );
    free( state->rgba_host );

    state->rgba_host = NULL;
    state->image_rgba_char = NULL;
    state->width = 0;
    state->height = 0;
}

/*
bool opencl_fd_run_gauss2d(
    struct FD* state,
    float sigma, 
    cl_uint num_events_in_wait_list, 
    const cl_event* event_wait_list, 
    cl_event* event )
{
    const size_t global_work_offset[] = { 0,0 };
    const size_t global_work_size[] = { state->width, state->height };
    const size_t local_work_size[] = { 2, 2 };

    cl_program program_gauss_cl  = opencl_program_load( "kernels/gauss.cl" );
    cl_kernel kernel_gauss2d = opencl_loader_load_kernel( program_gauss_cl, "gauss2d" );
    cl_command_queue command_queue = opencl_loader_get_command_queue();

    size_t gauss_kernel_diameter;
    cl_float * gauss_kernel_line = generate_gauss_kernel_2D( &gauss_kernel_diameter, sigma );
    size_t gauss_kernel_size = gauss_kernel_diameter * gauss_kernel_diameter;
    cl_int halfwidth = gauss_kernel_diameter/2;

    printf( "Kernel contents %lu %lu %f\n", gauss_kernel_size, gauss_kernel_diameter, sigma);
    for( int y = 0; y < gauss_kernel_diameter; y++ )
    {
        printf( "\n" );
        for( int x = 0; x < gauss_kernel_diameter; x++ )
        {
            printf( "%f ", gauss_kernel_line[x+y*gauss_kernel_diameter]);
        }
    }

    cl_int errcode_ret;
    cl_mem gauss_kernel_buffer = clCreateBuffer(
            opencl_loader_get_context(),
            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
            sizeof( cl_float ) * gauss_kernel_size,
            gauss_kernel_line,
            &errcode_ret
    );
    ASSERT_BUF(gauss_kernel_buffer, errcode_ret);

    cl_mem debug = clCreateBuffer(
            opencl_loader_get_context(),
            CL_MEM_WRITE_ONLY,
            sizeof( cl_float ) * gauss_kernel_size,
            NULL,
            &errcode_ret
    );
    ASSERT_BUF(debug, errcode_ret);

    cl_int cl_width = state->width;
    cl_int cl_height = state->height;
    clSetKernelArg( kernel_gauss2d, 0, sizeof( cl_mem ), &gauss_kernel_buffer );
    clSetKernelArg( kernel_gauss2d, 1, sizeof( cl_int ), &halfwidth );
    clSetKernelArg( kernel_gauss2d, 2, sizeof( cl_mem ), &state->buffer_l_float_1 );
    clSetKernelArg( kernel_gauss2d, 3, sizeof( cl_mem ), &state->buffer_l_float_2 );
    clSetKernelArg( kernel_gauss2d, 4, sizeof( cl_int ), &cl_width );
    clSetKernelArg( kernel_gauss2d, 5, sizeof( cl_int ), &cl_height );
    clSetKernelArg( kernel_gauss2d, 6, sizeof( cl_mem ), &debug );
    errcode_ret = clEnqueueNDRangeKernel( command_queue, 
        kernel_gauss2d, 
        2,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        event
    );
    ASSERT_ENQ( kernel_gauss2d, errcode_ret );

    cl_float* output = malloc( sizeof( cl_float ) * gauss_kernel_size );
    clEnqueueReadBuffer( command_queue,
            debug,
            true,
            0,
            gauss_kernel_size * sizeof( cl_float ),
            output,
            1,
            event,
            NULL
    );

    printf( "Kernel contents read %lu %f\n", gauss_kernel_diameter, sigma);
    for( int y = 0; y < gauss_kernel_diameter; y++ )
    {
        for( int x = 0; x < gauss_kernel_diameter; x++ )
        {
            printf( "%f ", output[x+y*gauss_kernel_diameter]);
        }
        printf( "\n" );
    }
            

    clReleaseKernel( kernel_gauss2d );
    clReleaseProgram( program_gauss_cl );
    clReleaseMemObject( gauss_kernel_buffer );
    //Free the kernel memory when done.
    clSetEventCallback( *event, CL_COMPLETE, free_host_ptr_callback, gauss_kernel_line );

    //Swap buffer positions so that buffer_l_float_1 is still the "incoming" memory zone
    cl_mem tmp = state->buffer_l_float_1;
    state->buffer_l_float_1 = state->buffer_l_float_2;
    state->buffer_l_float_2 = tmp;
    return true;
}
*/
