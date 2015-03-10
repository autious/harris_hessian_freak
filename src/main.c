#include "opencl_handler.h"
#include "opencl_test.h"
#include "opencl_error.h"
#include "opencl_program.h"
#include "opencl_fd.h"
#include "log.h"

#ifndef __ANDROID__

int main( int argc, const char ** argv )
{
    opencl_loader_init();

    if( argc == 3 )
    {
        LOGV("Running desaturation");
        struct FD process;
        cl_event desaturate_event, gauss_event;

        opencl_fd_load_image( argv[1], &process );
        opencl_fd_desaturate_image( &process, 0, NULL, &desaturate_event );
        opencl_fd_run_gaussxy( &process, 4.0f, 1, &desaturate_event, &gauss_event );
        opencl_fd_save_buffer_to_image( argv[2], &process, 1, &gauss_event );
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
