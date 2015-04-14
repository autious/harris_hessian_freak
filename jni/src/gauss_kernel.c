#include "gauss_kernel.h"

#include <math.h>
#include <stdlib.h>
#include <CL/opencl.h>

#include "ieeehalfprecision.h"

hh_float* generate_gauss_kernel_line( size_t* size, float sigma )
{
    *size = 2 * 3 * sigma;
    *size += (*size % 2) == 0 ? 1 : 0;  //Make uneven if even.
    cl_float* dest = (hh_float*)malloc( sizeof( cl_float ) * (*size) );

    for( int i = 0; i < *size; i++ )
    {
       int dist = i - *size/2;
       dest[i] = (exp(-0.5 * (dist * dist / (sigma * sigma)))) /(sqrt(2 * 3.141592) * sigma);
    }

#ifdef HH_USE_HALF
    cl_half* val = (cl_half*)malloc( sizeof( cl_half ) * (*size) );

    singles2halfp( val, dest, *size );

    free( dest );

    return val;
#else
    return dest;
#endif
}
