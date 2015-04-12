#define il(value,v_min,v_max) min(max(v_min,value),v_max)

__constant hh_float CORNER_RESPONSE_ALPHA = 0.04; // I have found literally one lecture that explains that this alpha is 
                   //empirically measured to be [0.04 - 0.06] and another that just uses 
                   //0.04 without ref
__kernel void harris_corner_response( 
        __global hh_float* xx, 
        __global hh_float* xy, 
        __global hh_float* yy, 
        __global hh_float* output, 
        hh_float sigmaD )
{
    int i = get_global_id(0);
    hh_float A = xx[i] * pow(sigmaD, 2);
    hh_float B = yy[i] * pow(sigmaD, 2);
    hh_float C = xy[i] * pow(sigmaD, 2);

    output[i] = A * B - C * C - CORNER_RESPONSE_ALPHA * pow(A + B, 2);
}

#define SUP_HALFWIDTH 1

__kernel void harris_corner_suppression(
        __global hh_float* in,
        __global hh_float* out,
        int width,
        int height )
{
    int2 c = (int2)(get_global_id(0),get_global_id(1));

    hh_float m = -HUGE_VALF;
    for( int y = -SUP_HALFWIDTH; y <= SUP_HALFWIDTH; y++ )
    {
        for( int x = -SUP_HALFWIDTH; x <= SUP_HALFWIDTH; x++ )
        {
            int2 lc = (int2)(il(x+c.x,0,width),il(y+c.y,0,height));
            m = fmax(m,in[lc.x+lc.y*width]);
        }
    }

    hh_float in_value = in[c.x+c.y*width];
    if( m > in_value )
        out[c.x+c.y*width] = 0;
    else
        out[c.x+c.y*width] = in_value;
}

#define HARRIS_THRESHOLD 0.0f

__kernel void harris_count( __global hh_float* in, volatile __global uint* strong, volatile __global int* count ) 
{
    int i = get_global_id(0);

    hh_float value = in[i];

    if( value > 0 )
    {
        atomic_inc( count );
        if( value > HARRIS_THRESHOLD )
        {
           strong[i] += 1;
        }
    }
}
