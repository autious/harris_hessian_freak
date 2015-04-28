/*
* Copyright (c) 2015, Max Danielsson <work@autious.net> and Thomas Sievert
* All rights reserved.
* 
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the organization nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
* 
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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
