#include <string.h>

#include "opencl_handler.h"
#include "opencl_test.h"
#include "opencl_error.h"
#include "opencl_program.h"
#include "opencl_fd.h"
#include "log.h"
#include "util.h"

#ifndef __ANDROID__

static void save_image( struct FD* state, const char* prefix, const char* filename, cl_mem mem, cl_uint count, cl_event *events )
{
    size_t len = strlen( filename ) + strlen( prefix ) + 2;
    char buf[len];
    snprintf( buf, len, "%s_%s", prefix, filename );
    opencl_fd_save_buffer_to_image( buf, state, mem, count, events );
}

int main( int argc, const char ** argv )
{
    opencl_loader_init();

    if( argc == 2 )
    {
        float sigmaI = 4.0f;
        float sigmaD = 0.7f * sigmaI;
        float alpha  = 0.04f;

        LOGV("Running desaturation");
        struct FD state;
        cl_event desaturate_event, gauss_event, derivate_event, second_moment_elements;

        opencl_fd_load_png_file( argv[1], &state );
        
        cl_mem gauss_blur = opencl_fd_create_image_buffer( &state );
        cl_mem ddx = opencl_fd_create_image_buffer( &state );
        cl_mem ddy = opencl_fd_create_image_buffer( &state );
        cl_mem xx = opencl_fd_create_image_buffer( &state );
        cl_mem xy = opencl_fd_create_image_buffer( &state );
        cl_mem yy = opencl_fd_create_image_buffer( &state );

        cl_mem tempxx = opencl_fd_create_image_buffer( &state );
        cl_mem tempxy = opencl_fd_create_image_buffer( &state );
        cl_mem tempyy = opencl_fd_create_image_buffer( &state );
        cl_mem harris_response = opencl_fd_create_image_buffer( &state );
        cl_mem harris_suppression = opencl_fd_create_image_buffer( &state );

        //Double derivate
        cl_mem ddxx = opencl_fd_create_image_buffer( &state );
        cl_mem ddxy = opencl_fd_create_image_buffer( &state );
        cl_mem ddyy = opencl_fd_create_image_buffer( &state );
        
        cl_mem hessian_out = opencl_fd_create_image_buffer( &state );

        opencl_fd_desaturate_image( &state, gauss_blur, 0, NULL, &desaturate_event );

        //borrow ddx for temp storage
        opencl_fd_run_gaussxy( sigmaD, &state, gauss_blur, ddx, gauss_blur, 1, &desaturate_event, &gauss_event );

        opencl_fd_derivate_image( &state, gauss_blur, ddx, ddy, 1, &gauss_event, &derivate_event );

        opencl_fd_second_moment_matrix_elements( &state, ddx, ddy, xx, xy, yy, 1, &derivate_event, &second_moment_elements );

        cl_event moment_gauss[3];
        opencl_fd_run_gaussxy( sigmaI, &state, xx, tempxx, xx, 1, &second_moment_elements, &moment_gauss[0] );
        opencl_fd_run_gaussxy( sigmaI, &state, xy, tempxy, xy, 1, &second_moment_elements, &moment_gauss[1] );
        opencl_fd_run_gaussxy( sigmaI, &state, yy, tempyy, yy, 1, &second_moment_elements, &moment_gauss[2] );

        cl_event harris_response_event;
        opencl_fd_run_harris_corner_response( &state, xx, xy, yy, harris_response, sigmaD, 3, moment_gauss, &harris_response_event );

        cl_event harris_suppression_event;
        opencl_fd_run_harris_corner_suppression( &state, harris_response, harris_suppression, 1, &harris_response_event, &harris_suppression_event );

        //Get second derivates for hessian
        cl_event second_derivate_events[2];
        opencl_fd_derivate_image( &state, ddx, ddxx, ddxy, 1, &derivate_event, &second_derivate_events[0] );
        opencl_fd_derivate_image( &state, ddy, NULL, ddyy, 1, &derivate_event, &second_derivate_events[1] );

        cl_event hessian_event;
        opencl_fd_run_hessian( &state, ddxx, ddxy, ddyy, hessian_out, sigmaD, 2, second_derivate_events, &hessian_event );

        cl_int* strong_corner_counts = calloc( state.width * state.height, sizeof( cl_int ) );
        cl_int corners; 

        cl_event harris_corner_count;
        opencl_fd_harris_corner_count( &state, harris_suppression, strong_corner_counts, &corners, 1, &harris_suppression_event, &harris_corner_count );

        clFinish( opencl_loader_get_command_queue() ); //Finish doing all the calculations before saving.

        for( int x = 0; x < state.width; x++ )
        {
            for( int y = 0; y < state.height; y++ )
            {
                int c = strong_corner_counts[x+y*state.width];
                if( c > 0 )
                    printf( "%d,%d:%d\n", x,y,c );
            }
        }

        printf( "Total count:%d\n", corners );

        save_image( &state, "gauss_blur",   argv[1], gauss_blur,      0, NULL );
        save_image( &state, "ddx",          argv[1], ddx,             0, NULL );
        save_image( &state, "ddy",          argv[1], ddy,             0, NULL );
        save_image( &state, "xx",           argv[1], xx,              0, NULL );
        save_image( &state, "xy",           argv[1], xy,              0, NULL );
        save_image( &state, "yy",           argv[1], yy,              0, NULL );

        save_image( &state, "harris_response", argv[1], harris_response, 0, NULL );
        save_image( &state, "harris_suppression", argv[1], harris_suppression, 0, NULL );

        save_image( &state, "ddxx", argv[1], ddxx, 0, NULL );
        save_image( &state, "ddxy", argv[1], ddxy, 0, NULL );
        save_image( &state, "ddyy", argv[1], ddyy, 0, NULL );
        save_image( &state, "hessian", argv[1], hessian_out, 0, NULL );

        opencl_fd_release_image_buffer( &state, gauss_blur );
        opencl_fd_release_image_buffer( &state, ddx );
        opencl_fd_release_image_buffer( &state, ddy );
        opencl_fd_release_image_buffer( &state, xx );
        opencl_fd_release_image_buffer( &state, xy );
        opencl_fd_release_image_buffer( &state, yy );
        opencl_fd_release_image_buffer( &state, tempxx );
        opencl_fd_release_image_buffer( &state, tempxy );
        opencl_fd_release_image_buffer( &state, tempyy );

        opencl_fd_release_image_buffer( &state, harris_response );
        opencl_fd_release_image_buffer( &state, harris_suppression );

        opencl_fd_release_image_buffer( &state, ddxx );
        opencl_fd_release_image_buffer( &state, ddxy );
        opencl_fd_release_image_buffer( &state, ddyy );
        opencl_fd_release_image_buffer( &state, hessian_out );

        opencl_fd_free( &state, 0, NULL );
        free( strong_corner_counts );
    }
    else
    {
        opencl_test_run();
    }

    opencl_program_close();
    opencl_loader_close();

    return 0;
}

#endif
