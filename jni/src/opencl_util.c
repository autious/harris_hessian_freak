#include "opencl_util.h"

#include "opencl_error.h"

#include <stdbool.h>

cl_ulong opencl_util_getduration( cl_event event )
{
#ifdef PROFILE
    cl_ulong start, end;
    cl_int err_ret = clGetEventProfilingInfo( event,
        CL_PROFILING_COMMAND_START,
        sizeof( cl_ulong ),
        &start,
        NULL );
    ASSERT_PROF( CL_PROFILING_COMMAND_START, err_ret );
    clGetEventProfilingInfo( event,
        CL_PROFILING_COMMAND_END,
        sizeof( cl_ulong ),
        &end,
        NULL );
    ASSERT_PROF( CL_PROFILING_COMMAND_END, err_ret );

    return end-start;
#else
    return 0;
#endif
}
