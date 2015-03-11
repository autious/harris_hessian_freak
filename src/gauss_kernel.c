#include "gauss_kernel.h"

#include <math.h>
#include <CL/opencl.h>

cl_float* generate_gauss_kernel_line( size_t* size, float sigma )
{
    *size = 2 * 3 * sigma;
    *size += (*size % 2) == 0 ? 1 : 0;  //Make uneven if even.
    cl_float* dest = (cl_float*)malloc( sizeof( cl_float ) * (*size) );
    for( int i = 0; i < *size; i++ )
    {
       int dist = i - *size/2;
       dest[i] = (exp(-0.5 * (dist * dist / (sigma * sigma)))) /(sqrt(2 * 3.141592) * sigma);
    }

    return dest;
}

/*
 * Lacking implementation of a decent 2d kernel
cl_float* generate_gauss_kernel_2D( size_t* diameter, float sigma )
{
    *diameter = 2 * 3 * sigma;
    *diameter += (*diameter % 2) == 0 ? 1 : 0;  //Make uneven if even.

    cl_float* dest = (cl_float*)malloc( sizeof( cl_float ) * (*diameter) * (*diameter) );

    for( int y = 0; y < *diameter; y++ )
    {
        int dy = y-*diameter/2;
        for( int x = 0; x < *diameter; x++ )
        {
            int dx = x-*diameter/2;
            //dest[y * *diameter+x] = (exp(-0.5 * ((dy * dy) / (sigma * sigma)+(dx * dx) / (sigma * sigma)))) /(sqrt(2 * 3.141592) * sigma);
        }
    }
    return dest;
}
*/
