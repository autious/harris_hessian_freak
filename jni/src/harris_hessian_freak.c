
#include "opencl_loader.h"
#include "opencl_test.h"
#include "opencl_error.h"
#include "opencl_program.h"
#include "string_object.h"
#include "opencl_fd.h"
#include "log.h"
#include "util.h"
#include "freak.h"

#include <errno.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

//Decided for the H-H method.
static const float HHSIGMAS[] = { 0.7f, 2.0f, 4.0f, 6.0f, 8.0f, 12.0f, 16.0f, 20.0f, 24.0f, 0.0f, 0.0f };

static float HESSIAN_DETERMINANT_THRESHOLD = 0.00001f;

static float HARRIS_THRESHOLD = 0.00001f;

static float CORNER_RESPONSE_ALPHA  = 0.04f;
                    // I have found literally one lecture that explains that this alpha is 
                    // empirically measured to be [0.04 - 0.06] and another that just uses 
                    // 0.04 without ref

#define IMAGE_SIZE_MULTIPLE 32
struct BufferMemory
{
    cl_mem desaturated_image;
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

    cl_mem hessian_determinant_buffer;
    cl_mem strong_responses;
    cl_mem keypoints_buf;
    cl_mem hessian_determinant_indices_buffer;

    cl_mem hessian_determinants[NELEMS( HHSIGMAS )];
    cl_mem hessian_corner_counts[NELEMS( HHSIGMAS )];
};

struct HarrisHessianScale
{
    float sigma;
    int hessian_determinant_index;
    cl_event execution_event;
};

static struct BufferMemory mem;
static struct FD state;
static struct HarrisHessianScale harris_hessian_scales[NELEMS( HHSIGMAS )];

static void save_image( const char* path, const char* filename, float sigma, cl_mem mem, cl_uint count, cl_event *events )
{
    int sigmai = sigma*100;
    size_t sigmalen = sigmai > 1 ? (int)log10( sigmai ) : 0;

    size_t len = strlen( path ) 
        + strlen( "/" ) 
        + sigmalen + 1
        + strlen( "_" )
        + strlen( filename ) 
        + strlen( ".png" ) 
        + 2;
    char buf[len];
    snprintf( buf, len, "%s/%d_%s.png", path, sigmai, filename );
    opencl_fd_save_buffer_to_image( buf, &state, mem, count, events );
}

#define SI(path,name,sigma,mem,count,event) if( path ) save_image( path, name,sigma, mem, count, event )

static void init_harris_buffers( )
{
    cl_context context = opencl_loader_get_context();

    freak_buildPattern();

    mem.desaturated_image = opencl_fd_create_image_buffer( &state );
    mem.gauss_blur = opencl_fd_create_image_buffer( &state );
    mem.ddx = opencl_fd_create_image_buffer( &state );
    mem.ddy = opencl_fd_create_image_buffer( &state );
    mem.xx = opencl_fd_create_image_buffer( &state );
    mem.xy = opencl_fd_create_image_buffer( &state );
    mem.yy = opencl_fd_create_image_buffer( &state );

    mem.tempxx = opencl_fd_create_image_buffer( &state );
    mem.tempxy = opencl_fd_create_image_buffer( &state );
    mem.tempyy = opencl_fd_create_image_buffer( &state );
    mem.harris_response = opencl_fd_create_image_buffer( &state );
    mem.harris_suppression = opencl_fd_create_image_buffer( &state );

    //Double derivate
    mem.ddxx = opencl_fd_create_image_buffer( &state );
    mem.ddxy = opencl_fd_create_image_buffer( &state );
    mem.ddyy = opencl_fd_create_image_buffer( &state );

    cl_int errcode_ret;

    mem.hessian_determinant_buffer = clCreateBuffer( context,
        CL_MEM_READ_WRITE,
        sizeof( hh_float ) * state.width * state.height * NELEMS( HHSIGMAS ),
        NULL,
        &errcode_ret
    );
    ASSERT_BUF( hessian_determinant_buffer, errcode_ret );

    mem.strong_responses = clCreateBuffer( context,
        CL_MEM_READ_WRITE,
        sizeof( cl_uint ) * state.width * state.height,
        NULL,
        &errcode_ret
    );
    ASSERT_BUF( strong_arr_buf, errcode_ret );


    mem.keypoints_buf = clCreateBuffer( context,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        sizeof( cl_ushort ) * state.width * state.height, 
        NULL,
        &errcode_ret 
    ); 
    ASSERT_BUF( keypoints_buf, errcode_ret );

    mem.hessian_determinant_indices_buffer = clCreateBuffer( context,
        CL_MEM_READ_ONLY,
        sizeof( cl_int ) * NELEMS( HHSIGMAS ),
        NULL,
        &errcode_ret
    );
    ASSERT_BUF( hessian_determinant_indices_buffer, errcode_ret );

    for( int i = 0; i < NELEMS( HHSIGMAS ); i++ )
    {
        harris_hessian_scales[i].sigma = HHSIGMAS[i];

        cl_buffer_region det_region = { 
            .origin = (sizeof( hh_float ) * state.width * state.height)*i,
            .size = sizeof( hh_float ) * state.width * state.height
        };

        mem.hessian_determinants[i] = clCreateSubBuffer( 
            mem.hessian_determinant_buffer,
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &det_region,
            &errcode_ret
        );
        ASSERT_BUF( strong_arr_buf, errcode_ret );
        
        cl_int value = 0;
        mem.hessian_corner_counts[i] = clCreateBuffer( context,
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR | CL_MEM_ALLOC_HOST_PTR,
            sizeof( cl_int ),
            &value,
            &errcode_ret
        );
        ASSERT_BUF( strong_arr_buf, errcode_ret );
    }

}

static void free_harris_buffers( )
{
    opencl_fd_release_image_buffer( &state, mem.desaturated_image );
    opencl_fd_release_image_buffer( &state, mem.gauss_blur );
    opencl_fd_release_image_buffer( &state, mem.ddx );
    opencl_fd_release_image_buffer( &state, mem.ddy );
    opencl_fd_release_image_buffer( &state, mem.xx );
    opencl_fd_release_image_buffer( &state, mem.xy );
    opencl_fd_release_image_buffer( &state, mem.yy );
    opencl_fd_release_image_buffer( &state, mem.tempxx );
    opencl_fd_release_image_buffer( &state, mem.tempxy );
    opencl_fd_release_image_buffer( &state, mem.tempyy );

    opencl_fd_release_image_buffer( &state, mem.harris_response );
    opencl_fd_release_image_buffer( &state, mem.harris_suppression );

    opencl_fd_release_image_buffer( &state, mem.ddxx );
    opencl_fd_release_image_buffer( &state, mem.ddxy );
    opencl_fd_release_image_buffer( &state, mem.ddyy );

    for( int i = 0; i < NELEMS( HHSIGMAS ); i++ )
    {
        clReleaseMemObject( mem.hessian_determinants[i] );
        clReleaseMemObject( mem.hessian_corner_counts[i] );
    }

    clReleaseMemObject( mem.strong_responses );
    clReleaseMemObject( mem.hessian_determinant_buffer );
    clReleaseMemObject( mem.hessian_determinant_indices_buffer );
    clReleaseMemObject( mem.keypoints_buf );

    memset( &mem, 0, sizeof( struct BufferMemory ));
}

void harris_hessian_freak_init( int width, int height)
{
    state.width = (width / IMAGE_SIZE_MULTIPLE) * IMAGE_SIZE_MULTIPLE;
    state.height = (height / IMAGE_SIZE_MULTIPLE) * IMAGE_SIZE_MULTIPLE;
    state.source_width = width;
    state.source_height = height;

    LOGV( 
        "Cut off %d,%d pixels from picture", 
        state.source_width - state.width, 
        state.source_height - state.height 
    ); 

    opencl_loader_init();

    assert( 16 >= NELEMS( HHSIGMAS ) ); //Verification that the short we're using for keypoints is sufficient.

    struct StringObject cfo;
    string_object_init( &cfo );
    string_object_add_compiler_flag( &cfo, "-cl-fast-relaxed-math" );
    string_object_add_compiler_flag( &cfo, "-cl-std=CL1.1" );
    string_object_add_define_integer( &cfo, "SCALE_COUNT", NELEMS( HHSIGMAS ) );
#ifdef HH_USE_HALF
    string_object_add_define( &cfo, "HH_USE_HALF" );
#endif

    const char *programs[] = 
    {
       "optimized_gauss.cl",
       "ref_gauss.cl",
       "derivate.cl",
       "desaturate.cl",
       "gauss.cl",
       "harris.cl",
       "hessian.cl",
       "smme.cl",
       NULL
    };

    opencl_program_compile( programs, cfo.str );

    string_object_free( &cfo );

    init_harris_buffers();
}

void harris_hessian_freak_close()
{
    free_harris_buffers();
    opencl_program_close();
    opencl_loader_close();
}


static int calc_derivate_sample_width( float sigma )
{
    return 1;
    //return (int)(sigma/4.0f+1.0f) ;
}


static bool do_harris( 
        cl_mem hessian_determinant, 
        cl_mem strong_responses,
        cl_mem corner_count,
        float sigmaI, 
        const char* save_path,
        cl_int wait_for_event_count,
        cl_event* wait_for, 
        cl_event* event 
)
{
    float sigmaD = 0.7f * sigmaI;
    //float alpha  = 0.04f;
    
    cl_event gauss_event, derivate_event, second_moment_elements;

    //borrow ddx for temp storage
    opencl_fd_run_gaussxy( 
        sigmaD, 
        &state, 
        mem.desaturated_image, 
        mem.ddx, 
        mem.gauss_blur, 
        wait_for_event_count, 
        wait_for, 
        &gauss_event 
    );
    SI( save_path, "gauss_blur_x", sigmaI, mem.ddx, 1, &gauss_event );
    SI( save_path, "gauss_blur_xy", sigmaI, mem.gauss_blur, 1, &gauss_event );

    opencl_fd_derivate_image( 
        &state, 
        calc_derivate_sample_width( sigmaI ),
        mem.gauss_blur, 
        mem.ddx, 
        mem.ddy, 
        1, 
        &gauss_event, 
        &derivate_event 
    );
    SI( save_path, "derivate_ddx", sigmaI, mem.ddx, 1, &derivate_event );
    SI( save_path, "derivate_ddy", sigmaI, mem.ddy, 1, &derivate_event );

    opencl_fd_second_moment_matrix_elements( 
        &state, 
        mem.ddx, 
        mem.ddy, 
        mem.xx, 
        mem.xy, 
        mem.yy, 
        1, 
        &derivate_event, 
        &second_moment_elements 
    );

    SI( save_path, "second_moment_xx", sigmaI, mem.xx, 1, &second_moment_elements );
    SI( save_path, "second_moment_xy", sigmaI, mem.xy, 1, &second_moment_elements );
    SI( save_path, "second_moment_yy", sigmaI, mem.yy, 1, &second_moment_elements );

    cl_event moment_gauss[3];
    opencl_fd_run_gaussxy( 
        sigmaI, 
        &state, 
        mem.xx, 
        mem.tempxx, 
        mem.xx, 
        1, 
        &second_moment_elements, 
        &moment_gauss[0] 
    );

    opencl_fd_run_gaussxy( 
        sigmaI, 
        &state, 
        mem.xy, 
        mem.tempxy, 
        mem.xy, 
        1, 
        &second_moment_elements, 
        &moment_gauss[1] 
    );

    opencl_fd_run_gaussxy( 
        sigmaI, 
        &state, 
        mem.yy, 
        mem.tempyy, 
        mem.yy, 
        1, 
        &second_moment_elements, 
        &moment_gauss[2] 
    );

    cl_event harris_response_event;
    opencl_fd_run_harris_corner_response( 
        &state, 
        mem.xx, 
        mem.xy, 
        mem.yy, 
        mem.harris_response, 
        sigmaD, 
        CORNER_RESPONSE_ALPHA,
        3, 
        moment_gauss, 
        &harris_response_event 
    );

    SI( save_path, "harris_response", sigmaI, mem.harris_response, 1, &harris_response_event );

    cl_event harris_suppression_event;
    opencl_fd_run_harris_corner_suppression( 
        &state, 
        mem.harris_response, 
        mem.harris_suppression, 
        1, 
        &harris_response_event, 
        &harris_suppression_event 
    );

    SI( save_path, "harris_suppression", sigmaI, mem.harris_suppression, 1, &harris_suppression_event );

    //Get second derivates for hessian
    cl_event second_derivate_events[2];
    opencl_fd_derivate_image( 
        &state, 
        1,
        mem.ddx, 
        mem.ddxx, 
        mem.ddxy, 
        1, 
        &derivate_event, 
        &second_derivate_events[0] 
    );

    opencl_fd_derivate_image( 
        &state, 
        1,
        mem.ddy, 
        NULL, 
        mem.ddyy, 
        1, 
        &derivate_event, 
        &second_derivate_events[1] 
    );
    
    cl_event hessian_event;
    opencl_fd_run_hessian( 
        &state, 
        mem.ddxx, 
        mem.ddxy, 
        mem.ddyy, 
        hessian_determinant,
        sigmaD, 
        2, 
        second_derivate_events, 
        &hessian_event
    );

    cl_event harris_corner_count_event;
    opencl_fd_harris_corner_count( 
        &state, 
        mem.harris_suppression, 
        strong_responses, 
        corner_count, 
        HARRIS_THRESHOLD,
        1, 
        &harris_suppression_event, 
        &harris_corner_count_event
    );

    //Need a gathered event for output.
    cl_int errcode_ret = clEnqueueMarker( opencl_loader_get_command_queue(), event );
    ASSERT_ENQ( enq_marker, errcode_ret );

    return true;
}


keyPoint* harris_hessian_freak_generate_keypoint_list( 
    size_t* in_size, 
    cl_uint event_count, 
    cl_event *event_wait_list, 
    cl_event* event 
)
{

    int size = 0;
    int count = 0;
    const int step_size = 1024;
    keyPoint * kps = NULL;
    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_int errcode_ret;


    cl_short* buf = clEnqueueMapBuffer( command_queue,
            mem.keypoints_buf,
            true,
            CL_MAP_READ,
            0,
            sizeof( cl_ushort ) * state.width * state.height,
            event_count,
            event_wait_list,
            NULL,
            &errcode_ret
    );
    ASSERT_MAP( kps, errcode_ret );

    for( int y = 0; y < state.height; y++ )
    {
        for( int x = 0; x < state.width; x++ )
        {
            cl_ushort val = buf[y*state.width+x];
            for( int i = 0; i < NELEMS(HHSIGMAS); i++ )
            {
                if( val & ( 1U << i ) )
                {
                    if( size < count + 1 )
                    {
                        size += step_size;
                        kps = realloc( kps, sizeof( keyPoint ) * size );
                    } 
                    keyPoint kp = { .x = x, .y = y, .size = harris_hessian_scales[i].sigma };
                    kps[count++] = kp;
                }
            }
        }
    }

    clEnqueueUnmapMemObject( command_queue,
            mem.keypoints_buf,
            buf,
            0,
            NULL,
            event
    );

    *in_size = count;
    return kps;
}

void harris_hessian_freak_detection( 
    uint8_t *rgba_data, 
    const char* save_path,
    cl_uint event_count, 
    cl_event* event_wait_list, 
    cl_event *event 
)
{
    LOGV("Running desaturation");

    cl_command_queue command_queue = opencl_loader_get_command_queue();

    opencl_fd_load_rgba( rgba_data, &state );
    init_harris_buffers( );
    
    cl_event desaturate_event;
    opencl_fd_desaturate_image( &state, mem.desaturated_image, 0, NULL, &desaturate_event );

    SI( save_path, "desaturated_source", 0.0f, mem.desaturated_image, 1, &desaturate_event );
    
    cl_int errcode_ret;

    void* strong_responses_map = clEnqueueMapBuffer( command_queue,
        mem.strong_responses,
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
        mem.strong_responses,
        strong_responses_map,
        0,
        NULL,
        &strong_responses_unmap_event
    );
    ASSERT_MAP( strong_responses, errcode_ret  );

    cl_event hessian_write_null_events[NELEMS( HHSIGMAS )];

    //For potential debugging reasons.
    memset( &harris_hessian_scales, 0, sizeof( struct HarrisHessianScale ) * NELEMS( HHSIGMAS ) );

    for( int i = 0; i < NELEMS( HHSIGMAS ); i++ )
    {
        harris_hessian_scales[i].sigma = HHSIGMAS[i];
        harris_hessian_scales[i].hessian_determinant_index = i;

        cl_int zero_value = 0;
        errcode_ret = clEnqueueWriteBuffer( 
            command_queue,
            mem.hessian_corner_counts[i],
            false,
            0,
            sizeof( cl_int ),
            &zero_value,
            1,
            &strong_responses_unmap_event,
            &hessian_write_null_events[i]
        );
        ASSERT_BUF( hessian_corner_counts, errcode_ret );
    }


    const int buffer_count = NELEMS( HHSIGMAS );
    //Run harris-hessian on all the standard scalespaces
    for( int i = 0; i < buffer_count-2; i++ )
    {
        LOGV( "dispatch Harris sigma:%f\n", HHSIGMAS[i] );
        do_harris( 
                mem.hessian_determinants[harris_hessian_scales[i].hessian_determinant_index],
                mem.strong_responses,
                mem.hessian_corner_counts[harris_hessian_scales[i].hessian_determinant_index],
                harris_hessian_scales[i].sigma,
                save_path,
                i == 0 ? NELEMS( HHSIGMAS ) : 1, 
                i == 0 ? hessian_write_null_events : &harris_hessian_scales[i-1].execution_event,
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
            mem.hessian_corner_counts[harris_hessian_scales[i].hessian_determinant_index],
            true,
            0,
            sizeof( cl_int ), &read_corner_count, 1,
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
    for( int i = buffer_count-3; i > max_corner_count_index; i-- )
    {
        harris_hessian_scales[i+2] = harris_hessian_scales[i];
    }

    harris_hessian_scales[max_corner_count_index+1] = harris_hessian_scales[max_corner_count_index];

    cl_event event_marker, event_before, event_after;
    clEnqueueMarker( command_queue, &event_marker );
    LOGV( "dispatch Harris sigma:%f\n", scale_before.sigma );
    do_harris(
            mem.hessian_determinants[scale_before.hessian_determinant_index],
            mem.strong_responses,
            mem.hessian_corner_counts[scale_before.hessian_determinant_index],
            scale_before.sigma,
            save_path,
            1,
            &event_marker,
            &event_before 
    );
    LOGV( "dispatch Harris sigma:%f\n", scale_after.sigma );
    do_harris(
            mem.hessian_determinants[scale_after.hessian_determinant_index],
            mem.strong_responses,
            mem.hessian_corner_counts[scale_after.hessian_determinant_index],
            scale_after.sigma,
            save_path,
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

    cl_event write_indices_event;
    clEnqueueWriteBuffer( 
        command_queue,
        mem.hessian_determinant_indices_buffer,
        false,
        0,
        sizeof(cl_int) * buffer_count,
        hessian_determinant_indices,
        1,
        &event_after,
        &write_indices_event
    );
               

    LOGV( "dispatch keypoints search\n" );
    opencl_fd_find_keypoints( 
        &state, 
        mem.hessian_determinant_buffer, 
        mem.strong_responses, 
        mem.keypoints_buf, 
        mem.hessian_determinant_indices_buffer,  
        HESSIAN_DETERMINANT_THRESHOLD,
        1, 
        &write_indices_event, 
        event
    );

    LOGV( "Total count:%d\n", total_corner_count );
    //LOGV( "Characteristic scale:%f\n",  );

    

    opencl_fd_free( &state, 0, NULL );
    //free( strong_corner_counts );
}

descriptor* harris_hessian_freak_build_descriptor( 
    keyPoint* keypoints_list,
    size_t keypoints_count,
    size_t *desc_count,
    cl_uint event_count, 
    cl_event* event_wait_list, 
    cl_event* event 
)
{
#ifdef PROFILE
    PROFILE_MM( "build_descriptor" );
#endif
    cl_command_queue command_queue = opencl_loader_get_command_queue();
    cl_int errcode_ret;

    cl_event grayscale_map_event;
    hh_float *grayscale_data = clEnqueueMapBuffer( 
        command_queue,
        mem.desaturated_image,
        false,
        CL_MAP_READ,
        0,
        sizeof( hh_float ) * state.width * state.height,
        event_count,
        event_wait_list,
        &grayscale_map_event,
        &errcode_ret
    );
    ASSERT_MAP( grayscale_data, errcode_ret );

    errcode_ret = clWaitForEvents( 1, &grayscale_map_event );
    ASSERT_WAIT( grayscale_map_event, errcode_ret );  

    LOGV( "Starting computation of freak" );
    int _desc_count;
    descriptor* desc = freak_compute(  //This kills the keypoints_list
        grayscale_data, 
        state.width, 
        state.height, 
        keypoints_list, 
        keypoints_count, 
        &_desc_count
    );
    LOGV( "Done computing freak" );

    *desc_count = _desc_count;

    clEnqueueUnmapMemObject( command_queue,
        mem.desaturated_image,
        grayscale_data,
        0,
        NULL,
        event 
    );
#ifdef PROFILE
    PROFILE_MM( "build_descriptor" );
#endif
    return desc;
}
