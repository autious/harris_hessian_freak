#include "opencl_handler.h"
#include "opencl_test.h"
#include "opencl_error.h"
#include "log.h"

#ifndef __ANDROID__

int main( int argc, const char ** argv )
{
    opencl_loader_init();

    if( argc == 3 )
    {
        LOGV("Running desaturation");
        opencl_test_desaturate_image( argv[1], argv[2] );
    }
    else
    {
        opencl_test_run();
    }

    opencl_loader_close();

    return 0;
}

#endif
