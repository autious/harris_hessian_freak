
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void desaturate( __global image2d_t image, __global float* desaturated_out, int width )
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    uint4 pixel = read_imageui( image, sampler, coord );
    desaturated_out[coord.x+coord.y*width] = (0.21f * pixel.x + 0.72f * pixel.y + 0.07f * pixel.z);
    //desaturated_out[coord.x+coord.y*width] = coord.x/(float)width * 255;
}

__kernel void gaussx( __global float* gauss_kernel, int kernel_radius, __global float* input, __global float* output, int width )
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    float sum = 0;

    for( int i = -kernel_radius; i <= kernel_radius; i++ )
    {
        sum += gauss_kernel[i+kernel_radius] * input[min(width-1,max(coord.x + i,0))+coord.y*width];
    }

    output[coord.x+coord.y*width] = sum;
}

__kernel void gaussy( __constant float* gauss_kernel, int kernel_radius, __global float* input, __global float* output, int width, int height)
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    float sum = 0;

    for( int i = -kernel_radius; i <= kernel_radius; i++ )
    {
        sum += gauss_kernel[i+kernel_radius] * input[min(height-1,max(coord.y + i,0))*width+coord.x];
    }

    output[coord.x+coord.y*width] = sum;
}

__kernel void gauss2d( __constant float* gauss_kernel, int kernel_radius, __global float* input, __global float* output, int width, int height, __global float* g_debug )
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));

    int kernel_diameter = kernel_radius*2+1;
    float sum = 0;
    if( coord.x == 0 && coord.y == 0 )
    {
        for( int y = -kernel_radius; y <= kernel_radius; y++ )
        {
            for( int x = -kernel_radius; x <= kernel_radius; x++ )
            {
                g_debug[(x+kernel_radius)+(y+kernel_radius)*kernel_diameter] = gauss_kernel[(x+kernel_radius)+(y+kernel_radius)*kernel_diameter];
            }
        }
    }

    for( int y = -kernel_radius; y <= kernel_radius; y++ )
    {
        for( int x = -kernel_radius; x <= kernel_radius; x++ )
        {
            //gauss_kernel[(x+kernel_radius)+(y+kernel_radius)*(kernel_radius*2)]
            sum += gauss_kernel[(x+kernel_radius)+(y+kernel_radius)*kernel_diameter] * input[min(width-1,max(coord.x + x,0))+min(height-1,max(coord.y + y,0))*width];
        }
    }

    output[coord.x+coord.y*width] = sum;
}
