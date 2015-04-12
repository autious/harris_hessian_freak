__kernel void gaussx( __global hh_float* gauss_kernel, int kernel_radius, __global hh_float* input, __global hh_float* output, int width, __local cache_float* cached_source )
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    hh_float sum = 0;

    for( int i = -kernel_radius; i <= kernel_radius; i++ )
    {
        sum += LOAD_HHF(gauss_kernel,i+kernel_radius) * LOAD_HHF(input,min(width-1,max(coord.x + i,0))+coord.y*width);
    }

    STORE_HHF(output,coord.x+coord.y*width,sum);
}

__kernel void gaussy( __constant hh_float* gauss_kernel, int kernel_radius, __global hh_float* input, __global hh_float* output, int width, int height, __local cache_float* cached_source )
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    hh_float sum = 0;

    for( int i = -kernel_radius; i <= kernel_radius; i++ )
    {
        sum += LOAD_HHF(gauss_kernel,i+kernel_radius) * LOAD_HHF(input,min(height-1,max(coord.y + i,0))*width+coord.x);
    }

    STORE_HHF(output, coord.x+coord.y*width, sum );
}
