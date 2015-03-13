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

        cl_mem ddx, ddy, gauss_blur,xx,xy,yy;
        
        gauss_blur = opencl_fd_create_image_buffer( &process );
        ddx = opencl_fd_create_image_buffer( &process );
        ddy = opencl_fd_create_image_buffer( &process );
        xx = opencl_fd_create_image_buffer( &process );
        xy = opencl_fd_create_image_buffer( &process );
        yy = opencl_fd_create_image_buffer( &process );

        opencl_fd_desaturate_image( &process, gauss_blur, 0, NULL, &desaturate_event );

        //borrow ddx for temp storage
        opencl_fd_run_gaussxy( 4.0f, &process, gauss_blur, ddx, gauss_blur, 1, &desaturate_event, &gauss_event );

        opencl_fd_derivate_image( &process, gauss_blur, ddx, ddy, 1, &gauss_event, &derivate_event );

        opencl_fd_second_moment_matrix_elements( &process, ddx, ddy, xx, xy, yy, 1, &derivate_event, &second_moment_elements );

        clFinish( opencl_loader_get_command_queue() ); //Finish doing all the calculations before saving.

        save_image( &process, "gauss_blur",   argv[1], gauss_blur,    0, NULL );
        save_image( &process, "ddx",          argv[1], ddx,           0, NULL );
        save_image( &process, "ddy",          argv[1], ddy,           0, NULL );
        save_image( &process, "xx",           argv[1], xx,           0, NULL );
        save_image( &process, "xy",           argv[1], xy,           0, NULL );
        save_image( &process, "yy",           argv[1], yy,           0, NULL );

        opencl_fd_release_image_buffer( &process, gauss_blur );
        opencl_fd_release_image_buffer( &process, ddx );
        opencl_fd_release_image_buffer( &process, ddy );
        opencl_fd_release_image_buffer( &process, xx );
        opencl_fd_release_image_buffer( &process, xy );
        opencl_fd_release_image_buffer( &process, yy );

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
