
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void desaturate( __global image2d_t image, __global float* desaturated_out, int width )
{
    int2 coord = (int2)(get_global_id(0), get_global_id(1));
    float4 pixel = read_imagef( image, sampler, coord );
    desaturated_out[coord.x+coord.y*width] = (0.21f * pixel.x + 0.72f * pixel.y + 0.07f * pixel.z);
    //desaturated_out[coord.x+coord.y*width] = coord.x/(float)width * 255;
    //desaturated_out[coord.x+coord.y*width] = pixel.x;
}
