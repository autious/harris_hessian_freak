
#define il(value,v_min,v_max) min(max(v_min,value),v_max)

__constant float alpha = 0.04; // I have found literally one lecture that explains that this alpha is 
                   //empirically measured to be [0.04 - 0.06] and another that just uses 
                   //0.04 without ref
__kernel void harris_corner_response( 
        __global float* xx, 
        __global float* xy, 
        __global float* yy, 
        __global float* output, 
        float sigmaD )
{
    int i = get_global_id(0);
    float A = xx[i] * pow(sigmaD, 2);
    float B = yy[i] * pow(sigmaD, 2);
    float C = xy[i] * pow(sigmaD, 2);

    output[i] = A * B - C * C - alpha * pow(A + B, 2);
}

#define SUP_HALFWIDTH 1

__kernel void harris_corner_suppression(
        __global float* in,
        __global float* out,
        int width,
        int height )
{
    int2 c = (int2)(get_global_id(0),get_global_id(1));

    float m = -HUGE_VALF;
    for( int y = -SUP_HALFWIDTH; y <= SUP_HALFWIDTH; y++ )
    {
        for( int x = -SUP_HALFWIDTH; x <= SUP_HALFWIDTH; x++ )
        {
            int2 lc = (int2)(il(x+c.x,0,width),il(y+c.y,0,height));
            m = fmax(m,in[lc.x+lc.y*width]);
        }
    }

    if( m > in[c.x+c.y*width] )
        in[c.x+c.y*width] = 0;
}
