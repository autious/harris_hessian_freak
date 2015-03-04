
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel desaturate( __read_only image2d_t image, __global float* desaturated_out, int width )
{
    int2 coord = float2(get_global_id(0),get_global_id(1));

    uint4 pixel = read_imageui( image, sampler, coord );
    desaturated_out[x+y*width] = (0.21f * pixel.x + 0.72f * pixel.y + 0.07f * pixel.z);
}

__kernel gaussx( __constant float* kernel, __global float* input, __global float* output )
{
       
}

__kernel gaussy( __constant float* kernel, __global float* intpu, __global float* output, int width, int height )
{

}


