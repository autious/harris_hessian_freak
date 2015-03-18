#include <string.h>

#include "opencl_handler.h"
#include "opencl_test.h"
#include "opencl_error.h"
#include "opencl_program.h"
#include "opencl_fd.h"
#include "harris_hessian.h"
#include "log.h"
#include "util.h"
#include "lodepng.h"

#ifndef __ANDROID__

int main( int argc, const char ** argv )
{
    opencl_loader_init();

    if( argc == 2 )
    {
        LOGV("Running harris hessian on %s\n", argv[1] );
        unsigned int lode_error;
        unsigned int width;
        unsigned int height;
        uint8_t *data;

        lode_error = lodepng_decode32_file( &data, &width, &height, argv[1] );

        if( lode_error == 0 )
        {
            LOGV( "Picture dimensions: (%u,%u)", width, height );

            harris_hessian_init(); 
            harris_hessian_detection(data, width, height);
            harris_hessian_close();
        }
        else
        {
            LOGE("Unable to load image: %s", argv[1] );
        }
    }
    else
    {
        LOGV("Doing test run");
        opencl_test_run();
    }

    opencl_program_close();
    opencl_loader_close();

    return 0;
}

#endif
