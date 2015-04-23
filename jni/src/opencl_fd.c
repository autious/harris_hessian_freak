#include "opencl_fd.h"

#include <stdbool.h>
#include <stdint.h>
#include <math.h>

#include "opencl_error.h"
#include "opencl_program.h"
#include "gauss_kernel.h"
#include "lodepng.h"
#include "ieeehalfprecision.h"

bool opencl_fd_load_rgba( uint8_t* data, struct FD* state )
{
    cl_int errcode_ret;
    cl_image_format image_format = {
        CL_RGBA,
        CL_UNORM_INT8
    };

    state->image_rgba_char = clCreateImage2D(
        opencl_loader_get_context(),
        CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
        &image_format,
        state->width,
        state->height,
        state->source_width*sizeof(uint8_t)*4,
        data,
        &errcode_ret 
    );
    ASSERT_BUF(input_image, errcode_ret);

    return true;
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

    hh_float* output_desaturated_image_hh_float = (hh_float*)malloc(state->width*state->height*sizeof(hh_float));

    errcode_ret = clEnqueueReadBuffer( command_queue,
        in,
        false,
        0,
        sizeof( hh_float ) * state->width * state->height,
        output_desaturated_image_hh_float,
        num_events_in_wait_list,
        event_wait_list,
        &buffer_read_event
    );

    if( errcode_ret == CL_SUCCESS )
    {
        errcode_ret = clWaitForEvents( 1, &buffer_read_event );

        ASSERT_WAIT( buffer_read_event, errcode_ret );  

        cl_float* output_desaturated_image;
#ifdef HH_USE_HALF
        output_desaturated_image = (cl_float*)malloc(state->width*state->height*sizeof(cl_float));
        halfp2singles( output_desaturated_image, output_desaturated_image_hh_float, state->width*state->height );
#else
        output_desaturated_image = output_desaturated_image_hh_float;
        output_desaturated_image_hh_float = NULL;
#endif

            if( output_desaturated_image_hh_float )
                free( output_desaturated_image_hh_float );

        uint8_t* output_desaturated_image_l = malloc( sizeof( uint8_t ) * state->width * state->height );

        float factor = 1.0f;
        float avg = 0.0f;
        float maxval = 0.0f;

        for( int i = 0; i < state->width * state->height; i++ )
        {
            avg += output_desaturated_image[i] / (float)(state->width * state->height);

            if( output_desaturated_image[i] > maxval )
                maxval = output_desaturated_image[i];
        } 

        if( avg < 0.25f && avg > 0.0f )
        {
            factor = 1.0f/maxval;
        }

        for( int i = 0; i < state->width * state->height; i++ )
        {
            //LOGV( "%f", output_desaturated_image[i] );
            int val = (int)(output_desaturated_image[i] * 255.0f * factor);
            val = val > 255 ? 255 : val;
            val = val < 0 ? 0 : val;
            output_desaturated_image_l[i] = val;
        } 

        lodepng_encode_file( name, output_desaturated_image_l, state->width, state->height, LCT_GREY, 8 );
        free( output_desaturated_image_l );

        free(output_desaturated_image);
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
            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            sizeof( hh_float ) * state->width * state->height,
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


//Takes a monochrome single channel float array and blurs it according to sigma
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
    const size_t local_work_size_gaussx[] = { 32, 1 };
    const size_t local_work_size_gaussy[] = { 4, 8 };

    cl_program program_gauss_cl = opencl_program_load( "gauss.cl" );
    cl_kernel kernel_gaussx      = opencl_loader_load_kernel( program_gauss_cl, "gaussx" );
    cl_kernel kernel_gaussy      = opencl_loader_load_kernel( program_gauss_cl, "gaussy" );
    cl_command_queue command_queue = opencl_loader_get_command_queue();

    const cl_int cl_width = state->width;
    const cl_int cl_height = state->height;

    size_t gauss_kernel_size;
    hh_float* gauss_kernel_line = generate_gauss_kernel_line( &gauss_kernel_size, sigma );

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
            sizeof( hh_float ) * gauss_kernel_size,
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
    clSetKernelArg( kernel_gaussx, 5, sizeof( hh_float ) * (gauss_kernel_size+local_work_size_gaussx[0]) * local_work_size_gaussx[1], NULL );
    errcode_ret = clEnqueueNDRangeKernel( command_queue, 
        kernel_gaussx, 
        2,
        global_work_offset,
        global_work_size,
        local_work_size_gaussx,
        num_events_in_wait_list,
        event_wait_list,
        &kernel_gaussx_event
    );
    ASSERT_ENQ( kernel_gaussx, errcode_ret );
#ifdef PROFILE
    PROFILE_PE( kernel_gaussx, kernel_gaussx_event );
#endif


    clSetKernelArg( kernel_gaussy, 0, sizeof( cl_mem ), &gauss_kernel_buffer );
    clSetKernelArg( kernel_gaussy, 1, sizeof( cl_int ), &kernel_radius );
    clSetKernelArg( kernel_gaussy, 2, sizeof( cl_mem ), &middle );
    clSetKernelArg( kernel_gaussy, 3, sizeof( cl_mem ), &out );
    clSetKernelArg( kernel_gaussy, 4, sizeof( cl_int ), &cl_width );
    clSetKernelArg( kernel_gaussy, 5, sizeof( cl_int ), &cl_height );
    clSetKernelArg( kernel_gaussy, 6, sizeof( hh_float ) * (gauss_kernel_size+local_work_size_gaussy[1]) * local_work_size_gaussy[0], NULL );
    errcode_ret = clEnqueueNDRangeKernel( command_queue, 
        kernel_gaussy, 
        2,
        global_work_offset,
        global_work_size,
        local_work_size_gaussy,
        1,
        &kernel_gaussx_event,
        event
    );
    ASSERT_ENQ( kernel_gaussy, errcode_ret );
#ifdef PROFILE
    PROFILE_PE( kernel_gaussy, *event );
#endif

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
    const size_t local_work_size[] = { 8,4 };

    cl_program program_gauss_cl  = opencl_program_load( "desaturate.cl" );
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
#ifdef PROFILE
        PROFILE_PE( kernel_desaturate, *event );
#endif


        clReleaseKernel( kernel_desaturate );

        return true;
    }
    return false;
}

bool opencl_fd_derivate_image( struct FD* state,
    cl_int cl_sample_width,
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
    const size_t local_work_size[] = { 8, 4 };

    cl_program program_gauss_cl  = opencl_program_load( "derivate.cl" );
    cl_kernel kernel_derivate;
        
    if( ddyout == NULL )
    {
        kernel_derivate = opencl_loader_load_kernel( program_gauss_cl, "derivate_x" ); 
    }
    else if( ddxout == NULL )
    {
        kernel_derivate = opencl_loader_load_kernel( program_gauss_cl, "derivate_y" );
    }
    else
    {
        kernel_derivate = opencl_loader_load_kernel( program_gauss_cl, "derivate" );
    }

    cl_command_queue command_queue = opencl_loader_get_command_queue();

    cl_int cl_width = state->width;
    cl_int cl_height = state->height;

    int kernelIndex = 0;
    clSetKernelArg( kernel_derivate, kernelIndex++, sizeof( cl_mem ), &in );

    if( ddxout != NULL )
    {
        clSetKernelArg( kernel_derivate, kernelIndex++, sizeof( cl_mem ), &ddxout );
    }

    if( ddyout != NULL )
    {
        clSetKernelArg( kernel_derivate, kernelIndex++, sizeof( cl_mem ), &ddyout );
    }

    clSetKernelArg( kernel_derivate, kernelIndex++, sizeof( cl_int ), &cl_width );
    clSetKernelArg( kernel_derivate, kernelIndex++, sizeof( cl_int ), &cl_height );
    clSetKernelArg( kernel_derivate, kernelIndex++, sizeof( cl_int ), &cl_sample_width );

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
    ASSERT_ENQ( kernel_derivate, errcode_ret );
#ifdef PROFILE
    PROFILE_PE( kernel_derivate, *event );
#endif

    clReleaseKernel( kernel_derivate );

    return true;
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
    const size_t local_work_size[] = { 32 };

    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_program program = opencl_program_load( "smme.cl" );
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
#ifdef PROFILE
    PROFILE_PE( smme_kernel, *event );
#endif
    
    clReleaseProgram( program );
    clReleaseKernel( smme_kernel );

    return true;
}

bool opencl_fd_run_harris_corner_response( struct FD* state,
        cl_mem xx,
        cl_mem xy,
        cl_mem yy,
        cl_mem output,
        param_float sigmaD,
        param_float corner_response_alpha,
        cl_uint num_events_in_wait_list,
        cl_event *event_wait_list,
        cl_event *event
)
{
    const size_t global_work_offset[] = { 0 };
    const size_t global_work_size[] = { state->width * state->height };
    const size_t local_work_size[] = { 32 };

    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_program program = opencl_program_load( "harris.cl" );
    cl_kernel harris_corner_response
        = opencl_loader_load_kernel( program, "harris_corner_response" );

    cl_int errcode_ret; 

    clSetKernelArg( harris_corner_response, 0, sizeof( cl_mem ), &xx );
    clSetKernelArg( harris_corner_response, 1, sizeof( cl_mem ), &xy );
    clSetKernelArg( harris_corner_response, 2, sizeof( cl_mem ), &yy );
    clSetKernelArg( harris_corner_response, 3, sizeof( cl_mem ), &output );
    clSetKernelArg( harris_corner_response, 4, sizeof( param_float ), &sigmaD );
    clSetKernelArg( harris_corner_response, 5, sizeof( param_float ), &corner_response_alpha );

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
#ifdef PROFILE
    PROFILE_PE( harris_corner_response, *event );
#endif
    
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
    const size_t local_work_size[] = { 8,4 };

    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_program program = opencl_program_load( "harris.cl" );
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
#ifdef PROFILE
    PROFILE_PE( harris_corner_suppression, *event );
#endif
    
    clReleaseProgram( program );
    clReleaseKernel( harris_corner_suppression );

    return true;
}

bool opencl_fd_run_hessian( struct FD* state,
        cl_mem xx,
        cl_mem xy,
        cl_mem yy,
        cl_mem out,
        param_float sigmaD,
        cl_uint num_events_in_wait_list,
        cl_event *event_wait_list,
        cl_event *event
)
{
    const size_t global_work_offset[] = { 0 };
    const size_t global_work_size[] = { state->width * state->height };
    const size_t local_work_size[] = { 16 };

    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_program program = opencl_program_load( "hessian.cl" );
    cl_kernel hessian
        = opencl_loader_load_kernel( program, "hessian" );

    cl_int errcode_ret; 

    clSetKernelArg( hessian, 0, sizeof( cl_mem ), &xx );
    clSetKernelArg( hessian, 1, sizeof( cl_mem ), &xy );
    clSetKernelArg( hessian, 2, sizeof( cl_mem ), &yy );
    clSetKernelArg( hessian, 3, sizeof( cl_mem ), &out );
    clSetKernelArg( hessian, 4, sizeof( param_float ), &sigmaD );

    errcode_ret = clEnqueueNDRangeKernel( command_queue,
        hessian,
        1,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        event
    );
    ASSERT_ENQ( hessian, errcode_ret );
#ifdef PROFILE
    PROFILE_PE( hessian, *event );
#endif
    
    clReleaseProgram( program );
    clReleaseKernel( hessian );

    return true;
        
}

bool opencl_fd_harris_corner_count( struct FD* state,
        cl_mem corners_in,
        cl_mem strong_responses,
        cl_mem corner_count,
        param_float harris_threshold,
        cl_uint num_events_in_wait_list,
        cl_event *event_wait_list,
        cl_event *event
)
{
    const size_t global_work_offset[] = { 0 };
    const size_t global_work_size[] = { state->width * state->height };
    const size_t local_work_size[] = { 32 };

    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_program program = opencl_program_load( "harris.cl" );
    cl_kernel harris_count
        = opencl_loader_load_kernel( program, "harris_count" );

    cl_int errcode_ret; 

    clSetKernelArg( harris_count, 0, sizeof( cl_mem ), &corners_in );
    clSetKernelArg( harris_count, 1, sizeof( cl_mem ), &strong_responses );
    clSetKernelArg( harris_count, 2, sizeof( cl_mem ), &corner_count );
    clSetKernelArg( harris_count, 3, sizeof( param_float ), &harris_threshold );

    errcode_ret = clEnqueueNDRangeKernel( command_queue,
        harris_count,
        1,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        event
    );
    ASSERT_ENQ( harris_count, errcode_ret );
#ifdef PROFILE
    PROFILE_PE( harris_count, *event );
#endif

    clReleaseProgram( program );
    clReleaseKernel( harris_count );

    return true;
}

bool opencl_fd_find_keypoints( 
        struct FD* state,
        cl_mem source_det, 
        cl_mem corner_counts, 
        cl_mem keypoints_data, 
        cl_mem hessian_determinant_indices,
        param_float hessian_determinant_threshold,
        cl_uint num_events_in_wait_list,
        cl_event *event_wait_list,
        cl_event *event
)
{
    const size_t global_work_offset[] = { 0 };
    const size_t global_work_size[] = { state->width * state->height };
    const size_t local_work_size[] = { 32 };

    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_program program = opencl_program_load( "hessian.cl" );
    cl_kernel find_keypoints
        = opencl_loader_load_kernel( program, "find_keypoints" );

    cl_int errcode_ret; 

    cl_int width = state->width;
    cl_int height = state->height;

    clSetKernelArg( find_keypoints, 0, sizeof( cl_mem ), &source_det );
    clSetKernelArg( find_keypoints, 1, sizeof( cl_mem ), &corner_counts );
    clSetKernelArg( find_keypoints, 2, sizeof( cl_mem ), &keypoints_data );
    clSetKernelArg( find_keypoints, 3, sizeof( cl_mem ), &hessian_determinant_indices);
    clSetKernelArg( find_keypoints, 4, sizeof( cl_int ), &width );
    clSetKernelArg( find_keypoints, 5, sizeof( cl_int ), &height );
    clSetKernelArg( find_keypoints, 6, sizeof( param_float ), &hessian_determinant_threshold );

    errcode_ret = clEnqueueNDRangeKernel( command_queue,
        find_keypoints,
        1,
        global_work_offset,
        global_work_size,
        local_work_size,
        num_events_in_wait_list,
        event_wait_list,
        event
    );
    ASSERT_ENQ( find_keypoints, errcode_ret );
#ifdef PROFILE
    PROFILE_PE( find_keypoints, *event );
#endif

    clReleaseProgram( program );
    clReleaseKernel( find_keypoints  );

    return true;
}

void opencl_fd_free( struct FD* state, 
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list
  )
{
    /*
    cl_int errcode_ret = clWaitForEvents( num_events_in_wait_list, event_wait_list );
    ASSERT_WAIT( event_wait_list, errcode_ret );  
    */

    clReleaseMemObject( state->image_rgba_char );

    state->image_rgba_char = NULL;
}
