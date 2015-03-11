#include "opencl_handler.h"
#include "opencl_test.h"
#include "opencl_error.h"
#include "opencl_program.h"
#include "opencl_fd.h"
#include "log.h"
#include "util.h"

#ifndef __ANDROID__

int main( int argc, const char ** argv )
{
    opencl_loader_init();

    if( argc == 3 )
    {
        LOGV("Running desaturation");
        struct FD process;
        cl_event desaturate_event, gauss_event;

        opencl_fd_load_png_file( argv[1], &process );

        cl_mem t[2];
        opencl_fd_create_image_buffers( &process, t, NELEMS(t) );

        opencl_fd_desaturate_image( &process, t[0], 0, NULL, &desaturate_event );
        opencl_fd_run_gaussxy( 4.0f, &process, t[0], t[1], t[0], 1, &desaturate_event, &gauss_event );
        //opencl_fd_run_gauss2d( &process, 1.0f, 1, &desaturate_event, &gauss_event );
        opencl_fd_save_buffer_to_image( argv[2], &process, t[0], 1, &gauss_event );

        opencl_fd_release_image_buffers( &process, t, NELEMS(t) );
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
