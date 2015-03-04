#pragma once
#include <math.h>

float* generateGaussKernelLine( size_t* size, float sigma )
{
    *size = 3 * sigma;
    *size += size % 2 == 0 ? 1 : 0;  //Make uneven if even.
    float* dest = (float*)malloc( sizeof( float ) * (*size) );
    for( int i = 0; i < size; i++ )
    {
       dist = i - size;
       dest[i] = (exp(-0.5 * (dist * dist / (sigma * sigma)))) /(sqrt(2 * 3.141592) * sigma);
    }

    return dest;
}

float generateGaussKernel2D( int* diameter, float sigma )
{
    *diameter = 3 * sigma;
    *size += size % 2 == 0 ? 1 : 0;  //Make uneven if even.

    float* dest = (float*)malloc( sizeof( float ) * (*diameter) * (*diameter) );

    for( int y = 0; y < diameter; y++ )
    {
        int dy = y-diameter;
        for( int x = 0; x < diameter; x++ )
        {
            int dx = x-diameter;
            dest[y*diameter+x] = (exp(-0.5 * ((dx * dx + dy * dy) / (sigma * sigma)))) /(sqrt(2 * 3.141592) * sigma);
        }
    }
    return dest;
}
