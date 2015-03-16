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
        struct FD process;
        cl_event desaturate_event, gauss_event, derivate_event, second_moment_elements;

        opencl_fd_load_png_file( argv[1], &process );
        
        cl_mem gauss_blur = opencl_fd_create_image_buffer( &process );
        cl_mem ddx = opencl_fd_create_image_buffer( &process );
        cl_mem ddy = opencl_fd_create_image_buffer( &process );
        cl_mem xx = opencl_fd_create_image_buffer( &process );
        cl_mem xy = opencl_fd_create_image_buffer( &process );
        cl_mem yy = opencl_fd_create_image_buffer( &process );

        cl_mem tempxx = opencl_fd_create_image_buffer( &process );
        cl_mem tempxy = opencl_fd_create_image_buffer( &process );
        cl_mem tempyy = opencl_fd_create_image_buffer( &process );
        cl_mem harris_response = opencl_fd_create_image_buffer( &process );

        opencl_fd_desaturate_image( &process, gauss_blur, 0, NULL, &desaturate_event );

        //borrow ddx for temp storage
        opencl_fd_run_gaussxy( sigmaD, &process, gauss_blur, ddx, gauss_blur, 1, &desaturate_event, &gauss_event );

        opencl_fd_derivate_image( &process, gauss_blur, ddx, ddy, 1, &gauss_event, &derivate_event );

        opencl_fd_second_moment_matrix_elements( &process, ddx, ddy, xx, xy, yy, 1, &derivate_event, &second_moment_elements );

        cl_event moment_gauss[3];
        opencl_fd_run_gaussxy( sigmaI, &process, xx, tempxx, xx, 1, &second_moment_elements, &moment_gauss[0] );
        opencl_fd_run_gaussxy( sigmaI, &process, xy, tempxy, xy, 1, &second_moment_elements, &moment_gauss[1] );
        opencl_fd_run_gaussxy( sigmaI, &process, yy, tempyy, yy, 1, &second_moment_elements, &moment_gauss[2] );

        cl_event harris_response_event;
        opencl_fd_run_harris_corner_response( &process, xx, xy, yy, harris_response, sigmaD, 3, moment_gauss, &harris_response_event );

        clFinish( opencl_loader_get_command_queue() ); //Finish doing all the calculations before saving.

        save_image( &process, "gauss_blur",   argv[1], gauss_blur,      0, NULL );
        save_image( &process, "ddx",          argv[1], ddx,             0, NULL );
        save_image( &process, "ddy",          argv[1], ddy,             0, NULL );
        save_image( &process, "xx",           argv[1], xx,              0, NULL );
        save_image( &process, "xy",           argv[1], xy,              0, NULL );
        save_image( &process, "yy",           argv[1], yy,              0, NULL );
        save_image( &process, "harris_response", argv[1], harris_response, 0, NULL );

        opencl_fd_release_image_buffer( &process, gauss_blur );
        opencl_fd_release_image_buffer( &process, ddx );
        opencl_fd_release_image_buffer( &process, ddy );
        opencl_fd_release_image_buffer( &process, xx );
        opencl_fd_release_image_buffer( &process, xy );
        opencl_fd_release_image_buffer( &process, yy );
        opencl_fd_release_image_buffer( &process, tempxx );
        opencl_fd_release_image_buffer( &process, tempxy );
        opencl_fd_release_image_buffer( &process, tempyy );
        opencl_fd_release_image_buffer( &process, harris_response );

        opencl_fd_free( &process, 0, NULL );
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
