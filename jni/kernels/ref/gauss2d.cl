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
