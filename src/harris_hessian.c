
#include "opencl_handler.h"
#include "opencl_test.h"
#include "opencl_error.h"
#include "opencl_program.h"
#include "opencl_fd.h"
#include "log.h"
#include "util.h"

#include <string.h>

//Decided for the H-H method.

struct BufferMemory
{
    cl_mem gauss_blur;
    cl_mem ddx;
    cl_mem ddy;
    cl_mem xx;
    cl_mem xy;
    cl_mem yy;

    cl_mem tempxx;
    cl_mem tempxy;
    cl_mem tempyy;
    cl_mem harris_response;
    cl_mem harris_suppression;

    //Double derivate
    cl_mem ddxx;
    cl_mem ddxy;
    cl_mem ddyy;
};

void harris_hessian_init()
{
}

static void init_harris_buffers( struct FD* state, struct BufferMemory* mem )
{
    mem->gauss_blur = opencl_fd_create_image_buffer( state );
    mem->ddx = opencl_fd_create_image_buffer( state );
    mem->ddy = opencl_fd_create_image_buffer( state );
    mem->xx = opencl_fd_create_image_buffer( state );
    mem->xy = opencl_fd_create_image_buffer( state );
    mem->yy = opencl_fd_create_image_buffer( state );

    mem->tempxx = opencl_fd_create_image_buffer( state );
    mem->tempxy = opencl_fd_create_image_buffer( state );
    mem->tempyy = opencl_fd_create_image_buffer( state );
    mem->harris_response = opencl_fd_create_image_buffer( state );
    mem->harris_suppression = opencl_fd_create_image_buffer( state );

    //Double derivate
    mem->ddxx = opencl_fd_create_image_buffer( state );
    mem->ddxy = opencl_fd_create_image_buffer( state );
    mem->ddyy = opencl_fd_create_image_buffer( state );
}

static bool do_harris( 
        struct FD* state, 
        struct BufferMemory* mem, 
        cl_mem hessian_determinant, 
        cl_mem strong_responses,
        cl_mem corner_count,
        float sigmaI, 
        cl_int wait_for_event_count,
        cl_event* wait_for, 
        cl_event* event 
)
{
    float sigmaD = 0.7f * sigmaI;
    //float alpha  = 0.04f;
    
    cl_event gauss_event, derivate_event, second_moment_elements;

    //borrow ddx for temp storage
    opencl_fd_run_gaussxy( sigmaD, state, mem->gauss_blur, mem->ddx, mem->gauss_blur, wait_for_event_count, wait_for, &gauss_event );

    opencl_fd_derivate_image( state, mem->gauss_blur, mem->ddx, mem->ddy, 1, &gauss_event, &derivate_event );

    opencl_fd_second_moment_matrix_elements( state, 
            mem->ddx, 
            mem->ddy, 
            mem->xx, 
            mem->xy, 
            mem->yy, 
            1, 
            &derivate_event, 
            &second_moment_elements 
    );

    cl_event moment_gauss[3];
    opencl_fd_run_gaussxy( sigmaI, state, mem->xx, mem->tempxx, mem->xx, 1, &second_moment_elements, &moment_gauss[0] );
    opencl_fd_run_gaussxy( sigmaI, state, mem->xy, mem->tempxy, mem->xy, 1, &second_moment_elements, &moment_gauss[1] );
    opencl_fd_run_gaussxy( sigmaI, state, mem->yy, mem->tempyy, mem->yy, 1, &second_moment_elements, &moment_gauss[2] );

    cl_event harris_response_event;
    opencl_fd_run_harris_corner_response( state, 
            mem->xx, 
            mem->xy, 
            mem->yy, 
            mem->harris_response, 
            sigmaD, 
            3, 
            moment_gauss, 
            &harris_response_event 
    );

    cl_event harris_suppression_event;
    opencl_fd_run_harris_corner_suppression( state, 
            mem->harris_response, 
            mem->harris_suppression, 
            1, 
            &harris_response_event, 
            &harris_suppression_event 
    );

    //Get second derivates for hessian
    cl_event second_derivate_events[2];
    opencl_fd_derivate_image( state, mem->ddx, mem->ddxx, mem->ddxy, 1, &derivate_event, &second_derivate_events[0] );
    opencl_fd_derivate_image( state, mem->ddy, NULL, mem->ddyy, 1, &derivate_event, &second_derivate_events[1] );

    opencl_fd_run_hessian( 
            state, 
            mem->ddxx, 
            mem->ddxy, 
            mem->ddyy, 
            hessian_determinant,
            sigmaD, 
            2, 
            second_derivate_events, 
            NULL
    );

    opencl_fd_harris_corner_count( state, mem->harris_suppression, strong_responses, corner_count, 1, &harris_suppression_event, NULL  );

    //Need a gathered event for output.
    cl_int errcode_ret = clEnqueueMarker( opencl_loader_get_command_queue(), event );
    ASSERT_ENQ( enq_marker, errcode_ret );

    return true;
}

static void free_harris_buffers( struct FD* state, struct BufferMemory* mem )
{
    opencl_fd_release_image_buffer( state, mem->gauss_blur );
    opencl_fd_release_image_buffer( state, mem->ddx );
    opencl_fd_release_image_buffer( state, mem->ddy );
    opencl_fd_release_image_buffer( state, mem->xx );
    opencl_fd_release_image_buffer( state, mem->xy );
    opencl_fd_release_image_buffer( state, mem->yy );
    opencl_fd_release_image_buffer( state, mem->tempxx );
    opencl_fd_release_image_buffer( state, mem->tempxy );
    opencl_fd_release_image_buffer( state, mem->tempyy );

    opencl_fd_release_image_buffer( state, mem->harris_response );
    opencl_fd_release_image_buffer( state, mem->harris_suppression );

    opencl_fd_release_image_buffer( state, mem->ddxx );
    opencl_fd_release_image_buffer( state, mem->ddxy );
    opencl_fd_release_image_buffer( state, mem->ddyy );
}

static void save_image( struct FD* state, const char* prefix, const char* filename, cl_mem mem, cl_uint count, cl_event *events )
{
    size_t len = strlen( filename ) + strlen( prefix ) + 2;
    char buf[len];
    snprintf( buf, len, "%s_%s", prefix, filename );
    opencl_fd_save_buffer_to_image( buf, state, mem, count, events );
}

void harris_hessian_detection( uint8_t *rgba_data, int width, int height )
{
    LOGV("Running desaturation");
    struct FD state;
    struct BufferMemory harris_data;
    float sigmaIs[] = { 0.7, 2, 4, 6, 8, 12, 16, 20, 24 };
    size_t buffer_count = NELEMS(sigmaIs) + 2;

    cl_context context = opencl_loader_get_context();
    //cl_command_queue command_queue = opencl_loader_get_command_queue();

    opencl_fd_load_rgba( rgba_data, width, height, &state );
    init_harris_buffers( &state, &harris_data );
    
    cl_event desaturate_event;
    opencl_fd_desaturate_image( &state, harris_data.gauss_blur, 0, NULL, &desaturate_event );
    
    cl_int errcode_ret;
    cl_mem hessian_determinants[buffer_count];
    cl_mem strong_responses[buffer_count];
    cl_mem corner_count[buffer_count];
    cl_event harris_queue[buffer_count];

    for( int i = 0; i < buffer_count; i++ )
    {
        hessian_determinants[i] = clCreateBuffer( context,
                        CL_MEM_WRITE_ONLY,
                        sizeof( cl_float ) * state.width * state.height,
                        NULL,
                        &errcode_ret
        );
        ASSERT_BUF( strong_arr_buf, errcode_ret );

        strong_responses[i] = clCreateBuffer( context,
                        CL_MEM_WRITE_ONLY,
                        sizeof( cl_int ) * state.width * state.height,
                        NULL,
                        &errcode_ret
        );
        ASSERT_BUF( strong_arr_buf, errcode_ret );
        
        cl_int value = 0;
        corner_count[i] = clCreateBuffer( context,
                        CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof( cl_int ),
                        &value,
                        &errcode_ret
        );
        ASSERT_BUF( strong_arr_buf, errcode_ret );

    }

    for( int i = 0; i < buffer_count-2; i++ )
    {
        printf( "dispatch Harris sigma:%f\n", sigmaIs[i] );
        do_harris( &state, 
                &harris_data, 
                hessian_determinants[i], 
                strong_responses[i], 
                corner_count[i], 
                sigmaIs[i], 
                i == 0 ? 0 : 1, 
                i == 0 ? NULL : &harris_queue[i-1], &harris_queue[i] );
    }

    /*
    cl_int *strong_corner_counts = malloc( sizeof(cl_int) * state.width * state.height );
    clEnqueueReadBuffer( 
            command_queue,
            strong_responses,
            false,
            0,
            sizeof( cl_int ) * state.width * state.height,
            strong_corner_counts,
            buffer_count,
            harris_queue,
            NULL
    );

    cl_int count;

    clEnqueueReadBuffer( command_queue,
        corner_count,
        false,
        0,
        sizeof( cl_int ) * 1,
        &count,
        buffer_count,
        harris_queue,
        NULL
    );
    */

    clFinish( opencl_loader_get_command_queue() ); //Finish doing all the calculations before saving.

    /*
    for( int x = 0; x < state.width; x++ )
    {
        for( int y = 0; y < state.height; y++ )
        {
            int c = strong_corner_counts[x+y*state.width];
            if( c > 0 )
                printf( "%d,%d:%d\n", x,y,c );
        }
    }
    */

    //printf( "Total count:%d\n", count );

    save_image( &state, "gauss_blur",           "out.png", harris_data.gauss_blur,            0, NULL );
    save_image( &state, "ddx",                  "out.png", harris_data.ddx,                   0, NULL );
    save_image( &state, "ddy",                  "out.png", harris_data.ddy,                   0, NULL );
    save_image( &state, "xx",                   "out.png", harris_data.xx,                    0, NULL );
    save_image( &state, "xy",                   "out.png", harris_data.xy,                    0, NULL );
    save_image( &state, "yy",                   "out.png", harris_data.yy,                    0, NULL );

    save_image( &state, "harris_response",      "out.png", harris_data.harris_response,       0, NULL );
    save_image( &state, "harris_suppression",   "out.png", harris_data.harris_suppression,    0, NULL );

    save_image( &state, "ddxx",                 "out.png", harris_data.ddxx,                  0, NULL );
    save_image( &state, "ddxy",                 "out.png", harris_data.ddxy,                  0, NULL );
    save_image( &state, "ddyy",                 "out.png", harris_data.ddyy,                  0, NULL );
    //save_image( &state, "hessian",              "out.png", hessian_determinants,           0, NULL );


    for( int i = 0; i < buffer_count; i++ )
    {
        clReleaseMemObject( hessian_determinants[i] );
        clReleaseMemObject( corner_count[i] );
        clReleaseMemObject( strong_responses[i] );
    }

    free_harris_buffers( &state, &harris_data );
    opencl_fd_free( &state, 0, NULL );
    //free( strong_corner_counts );
}

void harris_hessian_close()
{

}
