#include "opencl_handler.h"
#include "opencl_test.h"
#include "opencl_error.h"

#ifndef __ANDROID__

int main( int argc, const char ** argv )
{
    opencl_loader_init();

    opencl_test_run();

    opencl_loader_close();

    return 0;
}

#endif
