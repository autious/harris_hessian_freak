
#include "opencl_handler.h"
#include "opencl_test.h"
#include "opencl_error.h"
#include "opencl_program.h"
#include "opencl_fd.h"
#include "log.h"
#include "util.h"

#include <errno.h>
#include <string.h>
#include <math.h>

//Decided for the H-H method.
static const float HHSIGMAS[] = { 0.7f, 2.0f, 4.0f, 6.0f, 8.0f, 12.0f, 16.0f, 20.0f, 24.0f, 0.0f, 0.0f };

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

struct HarrisHessianScale
{
    float sigma;
    int hessian_determinant_index;
    cl_mem hessian_determinant;
    cl_mem corner_count;
    cl_event execution_event;
};

void harris_hessian_init()
{
    opencl_loader_init();

    assert( 16 >= NELEMS( HHSIGMAS ) ); //Verification that the short we're using for keypoints is sufficient.

    opencl_program_add_compiler_flag( "-cl-fast-relaxed-math" );
    opencl_program_add_compiler_flag( "-cl-std=CL1.1" );
    opencl_program_add_define_integer( "SCALE_COUNT", NELEMS( HHSIGMAS ) );

    const char *programs[] = 
    {
       "kernels/derivate.cl",
       "kernels/gauss.cl",
       "kernels/harris.cl",
       "kernels/hessian.cl",
       "kernels/smme.cl",
       NULL
    };

    opencl_program_compile( programs );
}

void harris_hessian_close()
{
    opencl_program_close();
    opencl_loader_close();
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

static void save_keypoints( struct FD* state, const char* filename, cl_mem keypoints )
{
    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_int errcode_ret;

    clFinish( command_queue );

    /*
    cl_short* buf = clEnqueueMapBuffer( command_queue,
            keypoints,
            true,
            CL_MAP_READ,
            0,
            sizeof( cl_ushort ) * state->width * state->height,
            0,
            NULL,
            NULL,
            &errcode_ret
    );
    ASSERT_MAP( keypoints, errcode_ret );
    */

    cl_ushort *buf = malloc( sizeof( cl_ushort ) * state->width * state->height );
   
    if( buf )
    { 
        errcode_ret = clEnqueueReadBuffer( command_queue,
                keypoints,
                true,
                0,
                sizeof( cl_ushort ) * state->width * state->height,
                buf,
                0,
                NULL,
                NULL );
        ASSERT_READ( keypoints, errcode_ret );

        int filenamelen = strlen(filename)+strlen(".kpts")+1;
        char name[filenamelen];
        snprintf( name, filenamelen, "%s.kpts", filename );
        FILE* f = fopen( name, "w" );

        LOGV( "Outputting\n" );

        for( int y = 0; y < state->height; y++ )
        {
            for( int x = 0; x < state->width; x++ )
            {
                cl_ushort val = buf[y*state->width+x];
                for( int i = 0; i < NELEMS(HHSIGMAS); i++ )
                {
                    if( val & ( 1U << i ) )
                    {
                        fprintf( f, "x:%d,y:%d,i:%d\n",x,y,i);
                    }
                }
            }
        }
        fclose( f );
    }
    else
    {
        LOGE( "Failed malloc" );
    }

    /*
    clEnqueueUnmapMemObject( command_queue,
            keypoints,
            buf,
            0,
            NULL,
            NULL
    );
    */

    clFinish( command_queue );
}

void harris_hessian_detection( uint8_t *rgba_data, int width, int height )
{
    LOGV("Running desaturation");
    struct FD state;
    struct BufferMemory harris_data;
    size_t buffer_count = NELEMS(HHSIGMAS);

    cl_context context = opencl_loader_get_context();
    cl_command_queue command_queue = opencl_loader_get_command_queue();

    opencl_fd_load_rgba( rgba_data, width, height, &state );
    init_harris_buffers( &state, &harris_data );
    
    cl_event desaturate_event;
    opencl_fd_desaturate_image( &state, harris_data.gauss_blur, 0, NULL, &desaturate_event );
    
    cl_int errcode_ret;
    struct HarrisHessianScale harris_hessian_scales[buffer_count];

    cl_mem hessian_determinant_buffer = clCreateBuffer( context,
        CL_MEM_READ_WRITE,
        sizeof( cl_float ) * state.width * state.height * buffer_count,
        NULL,
        &errcode_ret
    );

    cl_mem strong_responses = clCreateBuffer( context,
                    CL_MEM_READ_WRITE,
                    sizeof( cl_uint ) * state.width * state.height,
                    NULL,
                    &errcode_ret
    );
    ASSERT_BUF( strong_arr_buf, errcode_ret );

    void* strong_responses_map = clEnqueueMapBuffer( command_queue,
        strong_responses,
        true,
        CL_MAP_WRITE,
        0,
        sizeof( cl_uint ) * state.width * state.height,
        0,
        NULL,
        NULL,
        &errcode_ret 
    );
    ASSERT_MAP( strong_responses, errcode_ret  );
    memset( strong_responses_map, 0, sizeof( cl_uint ) * state.width * state.height );
    cl_event strong_responses_unmap_event;
    errcode_ret = clEnqueueUnmapMemObject( command_queue,
        strong_responses,
        strong_responses_map,
        0,
        NULL,
        &strong_responses_unmap_event
    );
    ASSERT_MAP( strong_responses, errcode_ret  );

    cl_mem keypoints_buf = clCreateBuffer( context,
        CL_MEM_READ_WRITE,
        sizeof( cl_ushort ) * state.width * state.height, 
        NULL,
        &errcode_ret 
    ); 
    ASSERT_BUF( keypoints_buf, errcode_ret );

    for( int i = 0; i < buffer_count; i++ )
    {
        harris_hessian_scales[i].sigma = HHSIGMAS[i];

        cl_buffer_region det_region = { 
            .origin = (sizeof( cl_float ) * state.width * state.height)*i,
            .size = sizeof( cl_float ) * state.width * state.height
        };

        harris_hessian_scales[i].hessian_determinant_index = i; //Used later when the buffers change places.
        harris_hessian_scales[i].hessian_determinant = clCreateSubBuffer( 
            hessian_determinant_buffer,
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &det_region,
            &errcode_ret
        );
        ASSERT_BUF( strong_arr_buf, errcode_ret );
        
        cl_int value = 0;
        harris_hessian_scales[i].corner_count = clCreateBuffer( context,
                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        sizeof( cl_int ),
                        &value,
                        &errcode_ret
        );
        ASSERT_BUF( strong_arr_buf, errcode_ret );
    }

    //Run harris-hessian on all the standard scalespaces
    for( int i = 0; i < buffer_count-2; i++ )
    {
        printf( "dispatch Harris sigma:%f\n", HHSIGMAS[i] );
        do_harris( &state, 
                &harris_data, 
                harris_hessian_scales[i].hessian_determinant,
                strong_responses,
                harris_hessian_scales[i].corner_count,
                harris_hessian_scales[i].sigma,
                i == 0 ? 0 : 1, 
                i == 0 ? &strong_responses_unmap_event : &harris_hessian_scales[i-1].execution_event,
                &harris_hessian_scales[i].execution_event );
    }

    int max_corner_count = 0;
    int max_corner_count_index = 0;
    int total_corner_count = 0;
    //Find the characteristic scale-space
    //This becomes the first explicit sync-point between GPU and CPU
    for( int i = 0; i < buffer_count-2; i++ )
    {
        LOGV( "Reading corner..." );
        cl_int read_corner_count;
        cl_int errcode_ret = clEnqueueReadBuffer( 
            command_queue,
            harris_hessian_scales[i].corner_count,
            true,
            0,
            sizeof( cl_int ),
            &read_corner_count,
            1,
            &harris_hessian_scales[i].execution_event,
            NULL
        );
        ASSERT_READ( harris_hessian_scales[i].corner_count, errcode_ret );

        if( read_corner_count > max_corner_count )
        {
            max_corner_count = read_corner_count;
            max_corner_count_index = i;
        }
        total_corner_count += read_corner_count;
    }

    struct HarrisHessianScale scale_before  = harris_hessian_scales[buffer_count-1];
    struct HarrisHessianScale scale_after   = harris_hessian_scales[buffer_count-2];
    scale_before.sigma   = HHSIGMAS[max_corner_count_index] / sqrt(2.0);
    scale_after.sigma  = HHSIGMAS[max_corner_count_index] * sqrt(2.0);

    //insert the two extra scale-spaces
    for( int i = buffer_count-3; i > max_corner_count_index ; i-- )
    {
        harris_hessian_scales[i+2] = harris_hessian_scales[i];
    }

    harris_hessian_scales[max_corner_count_index+1] = harris_hessian_scales[max_corner_count_index];

    cl_event event_marker, event_before, event_after;
    clEnqueueMarker( command_queue, &event_marker );
    printf( "dispatch Harris sigma:%f\n", scale_before.sigma );
    do_harris( &state, 
            &harris_data, 
            scale_before.hessian_determinant,
            strong_responses,
            scale_before.corner_count,
            scale_before.sigma,
            1,
            &event_marker,
            &event_before 
    );
    printf( "dispatch Harris sigma:%f\n", scale_after.sigma );
    do_harris( &state, 
            &harris_data, 
            scale_after.hessian_determinant,
            strong_responses,
            scale_after.corner_count,
            scale_after.sigma,
            1,
            &event_before,
            &event_after 
    );

    //Insert the two new surrounding scales
    harris_hessian_scales[max_corner_count_index] = scale_before;
    harris_hessian_scales[max_corner_count_index+2] = scale_after;

    cl_int hessian_determinant_indices[buffer_count];

    for( int i = 0; i < buffer_count; i++ )
    {
        hessian_determinant_indices[i] = harris_hessian_scales[i].hessian_determinant_index;
    }

    cl_mem hessian_determinant_indices_buffer = clCreateBuffer( context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            sizeof( cl_int ) * buffer_count,
            hessian_determinant_indices,
            &errcode_ret
    );
    ASSERT_BUF( hessian_determinant_indices_buffer, errcode_ret );

    cl_event find_keypoints_event;
    printf( "dispatch keypoints search\n" );
    opencl_fd_find_keypoints( 
            &state, 
            hessian_determinant_buffer, 
            strong_responses, 
            keypoints_buf, 
            hessian_determinant_indices_buffer,  
            1, 
            &event_after, 
            &find_keypoints_event );

    clFinish( opencl_loader_get_command_queue() ); //Finish doing all the calculations before saving.

    for( int i = 0; i < buffer_count; i++ )
    {
        clReleaseMemObject( harris_hessian_scales[i].hessian_determinant );
        clReleaseMemObject( harris_hessian_scales[i].corner_count );
    }
    
    clReleaseMemObject( strong_responses );
    clReleaseMemObject( hessian_determinant_buffer );
    clReleaseMemObject( hessian_determinant_indices_buffer );

    save_keypoints( &state, "out.png", keypoints_buf );

    clReleaseMemObject( keypoints_buf );

    printf( "Total count:%d\n", total_corner_count );
    //printf( "Characteristic scale:%f\n",  );

    /*
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
    */


    free_harris_buffers( &state, &harris_data );
    opencl_fd_free( &state, 0, NULL );
    //free( strong_corner_counts );
}
